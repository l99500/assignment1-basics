import os
import json
import mmap
import time
import pathlib
import resource
import multiprocessing
from collections import Counter, defaultdict
import regex as re  # 必须使用 regex 库以支持 \p{L}


# ====================================================
# 1. 独立的 Worker 函数 (定义在类外部，避免 Pickle 问题)
# ====================================================

def _worker_init(pattern_str, st_pattern_str):
    """
    子进程初始化函数：只在进程启动时运行一次。
    用于编译正则表达式，避免每次任务都传递编译好的对象。
    """
    global shared_regex, shared_st_pattern
    shared_regex = re.compile(pattern_str)
    if st_pattern_str:
        shared_st_pattern = re.compile(st_pattern_str)
    else:
        shared_st_pattern = None

def _process_chunk_task(args):
    """
    子进程的工作逻辑：读取文件块 -> 预分词 -> 统计频率
    """
    file_path, start_byte, end_byte, special_tokens = args
    local_counts = Counter()

    try:
        with open(file_path, "rb") as f:
            f.seek(start_byte)
            # 只读取分配给当前进程的字节块
            bytes_data = f.read(end_byte - start_byte)

        # 解码 (errors='replace' 防止切分点稍微切坏字节，虽然基于 endoftext 切分通常安全)
        text_chunk = bytes_data.decode("utf-8", errors="replace")

        # 1. 切分 Special Tokens (保护特殊标记不被后续正则打碎)
        if shared_st_pattern:
            parts = shared_st_pattern.split(text_chunk)
        else:
            parts = [text_chunk]

        for part in parts:
            # 跳过特殊标记或空字符串
            if part in special_tokens or not part:
                continue

            # 2. GPT-2 正则分词
            tokens = shared_regex.findall(part)
            
            # 3. 统计 (转为 bytes tuple)
            for token in tokens:
                token_bytes = token.encode("utf-8")
                # 将 b'abc' 转为 (97, 98, 99)
                ids = tuple(bytes([b]) for b in token_bytes)
                local_counts[ids] += 1

    except Exception as e:
        # 生产环境中建议使用 logging
        print(f"Worker Error processing {start_byte}-{end_byte}: {e}")
        return Counter()

    return local_counts


# ====================================================
# 2. 主类定义
# ====================================================

class BpeTrainMultiprocess:
    def __init__(self, 
                 input_path: str | os.PathLike,
                 vocab_size: int,
                 special_tokens: list[str],
                 chunk_size: int = 1024 * 1024 * 16, # 16MB per chunk
                 **kwargs):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.chunk_size = chunk_size
        
        # 初始化词表
        self.vocab_init()
        
        # 准备正则字符串 (注意：不在这里编译，而是传字符串给子进程)
        self.regex_pattern_str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        self.sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        if self.sorted_special_tokens:
            self.st_pattern_str = "(" + "|".join(re.escape(t) for t in self.sorted_special_tokens) + ")"
        else:
            self.st_pattern_str = None
            
        self.merges = []

    def vocab_init(self):
        """初始化字典: 0-255 bytes + special tokens"""
        self.vocab = {i: bytes([i]) for i in range(256)}
        # 建立倒排表：b'a' -> 97 (用于快速将 bytes 转为 ID)
        self.vocab_inv = {bytes([i]): i for i in range(256)}

        for st in self.special_tokens:
            st_bytes = st.encode("utf-8")
            if st_bytes not in self.vocab_inv:
                new_id = len(self.vocab)
                self.vocab[new_id] = st_bytes
                self.vocab_inv[st_bytes] = new_id

    def get_chunk_boundaries(self) -> list[tuple[int, int]]:
        """利用 mmap 快速寻找基于 special_token 的切分边界"""
        boundaries = []
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
            
        file_size = os.path.getsize(self.input_path)
        
        # 提前 encode，避免循环内重复 encode
        special_token_bytes = [st.encode("utf-8") for st in self.special_tokens]

        with open(self.input_path, "rb") as f:
            with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
                start = 0
                while start < file_size:
                    target_end = min(start + self.chunk_size, file_size)
                    
                    if target_end == file_size:
                        boundaries.append((start, file_size))
                        break

                    # 寻找 chunk_size 之后最近的特殊标记
                    next_delim_pos = -1
                    min_pos = float('inf')
                    
                    for st_bytes in special_token_bytes:
                        # 从 target_end 开始找
                        pos = mm.find(st_bytes, target_end)
                        if pos != -1 and pos < min_pos:
                            min_pos = pos
                    
                    if min_pos == float('inf'):
                        # 后面没有特殊标记了，这一块直接到文件末尾
                        boundaries.append((start, file_size))
                        break
                    
                    # 我们选择在特殊标记的 *起始位置* 切断
                    # Chunk 1: ... text ends here.
                    # Chunk 2: <|endoftext|> Next doc starts...
                    end = min_pos
                    boundaries.append((start, end))
                    start = end
        return boundaries

    def run_parallel_tokenization(self):
        """Step 1: 并行读取与预分词"""
        print(f"[1/3] Calculating chunk boundaries for {self.input_path}...")
        boundaries = self.get_chunk_boundaries()
        print(f"      Split into {len(boundaries)} chunks.")

        # 准备任务参数: (path, start, end, special_tokens_set)
        # 传递 set 查找更快
        st_set = set(self.special_tokens)
        tasks = [(self.input_path, start, end, st_set) for start, end in boundaries]

        print(f"[2/3] Parallel processing with {multiprocessing.cpu_count()} cores...")
        train_data_bytes = Counter()
        
        # 使用 initializer 初始化子进程的正则
        with multiprocessing.Pool(
            processes=multiprocessing.cpu_count(),
            initializer=_worker_init,
            initargs=(self.regex_pattern_str, self.st_pattern_str)
        ) as pool:
            # imap_unordered 性能更好，结果顺序不重要
            for i, local_counts in enumerate(pool.imap_unordered(_process_chunk_task, tasks)):
                train_data_bytes.update(local_counts)
                if (i + 1) % 10 == 0:
                    print(f"      Processed {i + 1}/{len(boundaries)} chunks...")

        print(f"      Pre-tokenization complete. Unique byte-tuples: {len(train_data_bytes)}")
        return train_data_bytes

    def train(self):
        """执行完整的 BPE 训练流程"""
        
        # 1. 获取预分词后的 byte tuples 统计
        train_data_bytes = self.run_parallel_tokenization()

        # 2. 转换数据格式: Bytes Tuple -> Int ID Tuple
        # 整数运算比字节对象运算快，且兼容 vocab 索引
        print("[2.5/3] Converting bytes to IDs...")
        train_data = Counter()
        for byte_tuple, count in train_data_bytes.items():
            try:
                # 利用初始化好的 vocab_inv 快速查找
                id_tuple = tuple(self.vocab_inv[b] for b in byte_tuple)
                train_data[id_tuple] = count
            except KeyError:
                # 理论上不会发生，除非正则切分出了 0-255 以外的字节（不可能）
                pass
        
        # 释放旧内存
        del train_data_bytes

        # 3. BPE 循环 (使用倒排索引优化)
        print("[3/3] Starting Fast BPE Loop (Inverted Index)...")
        
        # --- 构建倒排索引 ---
        # stats: 记录 pair 的频率 {(id1, id2): count}
        stats = Counter()
        # indices: 倒排索引 {pair: {word_tuple, ...}}
        # 记录每个 pair 出现在了哪些单词中，减少后续遍历范围
        indices = defaultdict(set)
        
        for ids, count in train_data.items():
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i+1])
                stats[pair] += count
                indices[pair].add(ids)

        # --- 循环合并 ---
        while len(self.vocab) < self.vocab_size:
            if not stats:
                break

            # Tie-breaking: 先比频率(高优)，再比字节序(字典序大优)
            # 注意：必须去 vocab 里查 bytes 内容来比较，不能直接比 ID
            best_pair = max(stats, key=lambda p: (stats[p], self.vocab[p[0]], self.vocab[p[1]]))
            
            # 如果最佳 pair 的频率已经归零（可能被其他合并破坏了），跳出
            if stats[best_pair] == 0:
                break

            # 执行 Merge
            new_id = len(self.vocab)
            part1 = self.vocab[best_pair[0]]
            part2 = self.vocab[best_pair[1]]
            self.vocab[new_id] = part1 + part2
            self.merges.append((part1, part2))

            # 打印进度
            if len(self.vocab) % 100 == 0:
                print(f"      Vocab size: {len(self.vocab)}/{self.vocab_size} | Merged: {best_pair} -> {new_id}")

            # --- 快速更新逻辑 (只更新相关单词) ---
            # 获取所有包含 best_pair 的单词列表
            words_to_update = list(indices[best_pair])
            changes = [] # 暂存变更：(old_ids, new_ids, count)

            for old_ids in words_to_update:
                count = train_data[old_ids]
                new_ids_list = []
                i = 0
                
                # 在当前单词中执行替换
                while i < len(old_ids):
                    if i < len(old_ids) - 1 and old_ids[i] == best_pair[0] and old_ids[i+1] == best_pair[1]:
                        # 找到匹配！执行合并
                        
                        # A. 维护左邻居的统计
                        if new_ids_list:
                            prev = new_ids_list[-1]
                            # 旧邻居 (prev, best_pair[0]) 频率减少
                            old_prev_pair = (prev, old_ids[i])
                            stats[old_prev_pair] -= count
                            if stats[old_prev_pair] == 0: del stats[old_prev_pair]
                            if old_prev_pair in indices: indices[old_prev_pair].discard(old_ids)
                            
                            # 新邻居 (prev, new_id) 频率增加
                            new_prev_pair = (prev, new_id)
                            stats[new_prev_pair] += count
                            indices[new_prev_pair].add(old_ids) # 注意：这里暂时还存的是 old_ids, 稍后批量清理

                        # B. 维护右邻居的统计 (如果合并打断了右边的 pair)
                        # 比如 A B C，合并 A B -> AB。原来的 (B, C) 就不存在了
                        if i + 2 < len(old_ids):
                            old_next_pair = (old_ids[i+1], old_ids[i+2])
                            stats[old_next_pair] -= count
                            if stats[old_next_pair] == 0: del stats[old_next_pair]
                            if old_next_pair in indices: indices[old_next_pair].discard(old_ids)

                        new_ids_list.append(new_id)
                        i += 2 # 跳过两个元素
                    else:
                        new_ids_list.append(old_ids[i])
                        i += 1
                
                new_ids_tuple = tuple(new_ids_list)
                changes.append((old_ids, new_ids_tuple, count))
            
            # --- 批量应用 Train Data 和 Indices 的变更 ---
            # 1. 彻底删除 best_pair 的记录
            del indices[best_pair]
            del stats[best_pair]

            for old_ids, new_ids, count in changes:
                # 2. 从 train_data 移除旧单词，添加新单词
                if old_ids in train_data:
                    del train_data[old_ids]
                train_data[new_ids] += count
                
                # 3. 修正 Indices 指向
                # 上面的循环中，我们仅仅是从 indices 中 discard 了 old_ids
                # 我们需要把 new_ids 加入到它包含的所有 pair 的索引中
                # 优化：其实只需要更新与 new_id 相关的 pair 即可，但全量更新更不容易出错
                # 实际上，在上面的 A 步骤中，我们向 indices 加了 old_ids (作为占位)。
                # 这里的逻辑稍微复杂，为了代码清晰，我们采用简单的“重新注册新词”策略：
                
                # 将新单词加入到它包含的所有 pair 的索引中
                for i in range(len(new_ids) - 1):
                    p = (new_ids[i], new_ids[i+1])
                    indices[p].add(new_ids)
                    # 同时要清理掉旧的 old_ids (如果之前没清干净)
                    if old_ids in indices[p]:
                        indices[p].discard(old_ids)

        return self.vocab, self.merges

# ====================================================
# 3. 辅助报告函数
# ====================================================

def save_and_report(vocab, merges, elapsed_time, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取内存峰值 (兼容 Linux/macOS)
    try:
        peak_memory_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux: KB, MacOS: Bytes (需要判断系统，这里简单按 KB 处理，通常服务器是 Linux)
        import platform
        if platform.system() == 'Darwin': # MacOS
             peak_memory_mb = peak_memory_kb / (1024 * 1024)
        else:
             peak_memory_mb = peak_memory_kb / 1024
    except:
        peak_memory_mb = 0.0

    print(f"\n📊 Performance Report:")
    print(f"----------------------")
    print(f"Time Taken  : {elapsed_time:.2f} seconds ({elapsed_time/3600:.4f} hours)")
    print(f"Peak Memory : {peak_memory_mb:.2f} MB")

    # 统计最长 Token
    if vocab:
        longest_token_bytes = max(vocab.values(), key=len)
        print(f"Longest Token Length: {len(longest_token_bytes)} bytes")
        try:
            print(f"Longest Token Content: {longest_token_bytes.decode('utf-8')}")
        except UnicodeDecodeError:
            print(f"Longest Token Content (repr): {repr(longest_token_bytes)}")

    print(f"\n💾 Saving to {output_dir}...")

    # 保存 Vocab
    json_vocab = {}
    for token_id, token_bytes in vocab.items():
        try:
            token_str = token_bytes.decode("utf-8")
        except UnicodeDecodeError:
            token_str = str(token_bytes)
        json_vocab[token_id] = token_str

    with open(os.path.join(output_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(json_vocab, f, indent=2, ensure_ascii=False)

    # 保存 Merges
    with open(os.path.join(output_dir, "merges.txt"), "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n") 
        for p1, p2 in merges:
            # 将 bytes 解码并用 Ġ 替换空格，方便可视化
            s1 = p1.decode("utf-8", errors="replace").replace(" ", "Ġ")
            s2 = p2.decode("utf-8", errors="replace").replace(" ", "Ġ")
            f.write(f"{s1} {s2}\n")

    print("Done.")

# ====================================================
# 4. 执行入口
# ====================================================

if __name__ == "__main__":
    # 配置区
    VOCAB_SIZE = 1000  # 测试用，实际可设为 32000 或更大
    SPECIAL_TOKENS = ["<|endoftext|>"]
    
    # 路径配置
    project_path = pathlib.Path(__file__).resolve().parent.parent.parent
    # 假设你的文件在这个位置，如果不存在，请修改路径
    input_path = os.path.join(project_path, "data/TinyStoriesV2-GPT4-train.txt")
    
    # 检查文件是否存在
    if not os.path.exists(input_path):
        print(f"❌ Error: Input file not found at {input_path}")
        # 为了演示，可以生成一个假的测试文件
        # print("Creating dummy file...")
        # with open("dummy_corpus.txt", "w") as f: f.write("Hello world " * 10000)
        # input_path = "dummy_corpus.txt"
    else:
        print(f"🚀 Starting BPE Training on {input_path}")
        print(f"   Target Vocab Size: {VOCAB_SIZE}")
        
        start_time = time.time()
        
        bpe = BpeTrainMultiprocess(
            input_path=input_path,
            vocab_size=VOCAB_SIZE,
            special_tokens=SPECIAL_TOKENS
        )
        
        # 开始训练
        bpe.train()

        end_time = time.time()
        
        save_and_report(
            vocab=bpe.vocab,
            merges=bpe.merges,
            elapsed_time=end_time - start_time
        )
import os
import regex as re
from collections import Counter

def run_train_bpe_naive(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # --- 1. 初始化 Vocab ---
    # 0-255 的基础字节
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    # 追加 Special Tokens
    for st in special_tokens:
        vocab[len(vocab)] = st.encode("utf-8")

    # 记录合并规则
    merges: list[tuple[bytes, bytes]] = []

    # --- 2. 预处理 (Pre-tokenization) ---
    word_counts = Counter()
    
    # 注意：标准 re 不支持 \p{L}，这里建议使用 regex 库，或者简化正则。
    # 为了保证代码能跑，我这里用标准 re 的近似写法。
    # 如果允许用 import regex，请保留你原来的写法。
    # GPT-2 的正则是针对 Unicode 字符设计的，这里简化为通用匹配。
    tokenizer_regex = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    # 构造 Special Token 的切分正则
    if special_tokens:
        # 对特殊字符进行转义，防止正则报错
        st_pattern = "(" + "|".join(re.escape(t) for t in special_tokens) + ")"
    else:
        st_pattern = None

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            # A. 先处理 Special Tokens (核心修正)
            if st_pattern:
                # 保留分隔符的 split
                parts = re.split(st_pattern, line)
            else:
                parts = [line]

            # B. 对非 Special Token 的部分跑正则
            for part in parts:
                if part in special_tokens:
                    # 如果是特殊标记，直接跳过，不用加入 BPE 统计
                    continue
                if not part: 
                    continue
                
                # 跑 GPT-2 正则
                word_list_per_line = re.findall(tokenizer_regex, part)

                # 转 bytes tuple 并统计
                for i_w in word_list_per_line:
                    i_w_bytes = i_w.encode("utf-8")
                    # (b'a', b'p', b'p', b'l', b'e')
                    word_tuple = tuple(bytes([i_]) for i_ in i_w_bytes)
                    word_counts[word_tuple] += 1    

    # --- 3. BPE 循环 ---
    # 注意：这里用 < 而不是 <=，因为一旦 len(vocab) == vocab_size 就该停了
    while len(vocab) < vocab_size:
        
        # A. 统计 Bigram (Pair)
        pair_counts = Counter()
        for word, count in word_counts.items():
            # 遍历当前单词里的所有相邻对
            for i in range(len(word) - 1):
                pair = (word[i], word[i+1]) # 修正：Key 必须是 Tuple
                pair_counts[pair] += count

        # 如果没有 pair 可合并了（比如所有词都变成了单个 token），提前退出
        if not pair_counts:
            break

        # B. 找最高频 Pair (Tie-breaking: 频率高优先 -> 字典序大优先)
        # max 的 key 逻辑：先比 count，再比 pair 本身
        # Python 的 tuple 比较是逐位的，符合字典序
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
        
        # C. 更新 Vocab 和 Merges
        new_token = best_pair[0] + best_pair[1] # b'e' + b's' -> b'es'
        vocab[len(vocab)] = new_token
        merges.append(best_pair) # 记录 (b'e', b's')

        # D. 更新 word_counts (Apply Merge) - 修正后的逻辑
        new_word_counts = Counter()
        for word, count in word_counts.items():
            new_word = []
            i = 0
            while i < len(word):
                # 检查是否匹配 best_pair
                if i < len(word) - 1 and word[i] == best_pair[0] and word[i+1] == best_pair[1]:
                    new_word.append(new_token) # 合并
                    i += 2 # 跳过两个元素
                else:
                    new_word.append(word[i]) # 照搬
                    i += 1
            new_word_counts[tuple(new_word)] = count
        
        word_counts = new_word_counts
        print(f"Merged: {best_pair} -> {new_token}, Vocab size: {len(vocab)}")

    return vocab, merges


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # ==========================================
    # 1. 初始化词表 (Vocabulary Initialization)
    # ==========================================
    # 词表：ID -> Bytes
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    
    # 倒排索引：Bytes -> ID (方便后续查找)
    vocab_inv: dict[bytes, int] = {bytes([i]): i for i in range(256)}

    # 处理 Special Tokens
    # 注意：Special Tokens 直接加入词表，不参与 BPE 合并运算，但在分词时要保留
    sorted_special_tokens = sorted(special_tokens, key=len, reverse=True) # 长词优先匹配
    for st in special_tokens:
        st_bytes = st.encode("utf-8")
        if st_bytes not in vocab_inv:
            new_id = len(vocab)
            vocab[new_id] = st_bytes
            vocab_inv[st_bytes] = new_id

    # ==========================================
    # 2. 预分词与统计 (Pre-tokenization & Loading)
    # ==========================================
    # 我们使用一个字典来存储所有单词的 Token ID 序列及其出现频率
    # 结构: { (id1, id2, id3...): count }
    # 这样可以极大压缩数据量，后续循环只需要遍历这个字典，不需要遍历原始文本
    train_data: dict[tuple[int, ...], int] = Counter()

    # GPT-2 的标准正则
    # 这里的正则将文本切分为：缩写、单词、数字、非空白符号、空白符
    pat_str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    regex = re.compile(pat_str)

    # 构造 Special Token 切分正则：(st1|st2|...)
    if sorted_special_tokens:
        st_pattern = "(" + "|".join(re.escape(t) for t in sorted_special_tokens) + ")"
    else:
        st_pattern = None

    print(f"Loading data from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            # 第一步：先用 Special Tokens 切开，保护它们不被后续正则打碎
            if st_pattern:
                parts = re.split(st_pattern, line)
            else:
                parts = [line]

            for part in parts:
                # 如果是 Special Token，跳过 BPE 统计（它们单独存在）
                if part in special_tokens:
                    continue
                if not part:
                    continue

                # 第二步：对普通文本运行 GPT-2 正则
                chunks = re.findall(regex, part)

                # 第三步：将每个 Chunk 转换为初始的 ID 列表 (Byte IDs)
                for chunk in chunks:
                    chunk_bytes = chunk.encode("utf-8")
                    # 将 bytes 转为整数 tuple: b'abc' -> (97, 98, 99)
                    ids = tuple(chunk_bytes) 
                    train_data[ids] += 1

    print(f"Data loaded. Unique words: {len(train_data)}")

    # ==========================================
    # 3. BPE 训练循环 (Training Loop)
    # ==========================================
    merges: list[tuple[bytes, bytes]] = []

    # 循环直到词表填满
    while len(vocab) < vocab_size:
        # A. 统计当前所有 pair 的频率
        stats = Counter()
        for ids, count in train_data.items():
            # 遍历当前单词内的所有相邻 pair
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i+1])
                stats[pair] += count

        # 如果没有 pair 了（所有词都合并完了），提前退出
        if not stats:
            break

        # B. 找到频率最高的 Pair
        # 评判标准：1. 频率最高 (stats[p])  2. 字典序最大 (p)
        # 注意：Python tuple 比较是逐位的，(100, 200) > (100, 199)，符合字节序要求
        best_pair = max(stats, key=lambda p: (stats[p], p))

        # C. 执行合并 (Merge)
        # 1. 在词表中注册新 Token
        new_id = len(vocab)
        token_bytes_0 = vocab[best_pair[0]]
        token_bytes_1 = vocab[best_pair[1]]
        new_token_bytes = token_bytes_0 + token_bytes_1
        
        vocab[new_id] = new_token_bytes
        merges.append((token_bytes_0, token_bytes_1))

        # 2. 更新训练数据 (将 best_pair 替换为 new_id)
        #    这里不重新扫描，而是构建一个新的字典
        new_train_data = {}
        for ids, count in train_data.items():
            # 如果这个单词里根本没有 best_pair 的这两个 ID，直接跳过处理，原样复制
            # 这是一个简单的加速检查
            if best_pair[0] not in ids:
                new_train_data[ids] = count
                continue

            # 开始替换逻辑
            new_ids = []
            i = 0
            while i < len(ids):
                # 检查是否匹配 best_pair
                if i < len(ids) - 1 and ids[i] == best_pair[0] and ids[i+1] == best_pair[1]:
                    new_ids.append(new_id)
                    i += 2 # 跳过两个
                else:
                    new_ids.append(ids[i])
                    i += 1
            new_train_data[tuple(new_ids)] = count
        
        train_data = new_train_data

        # 打印进度 (可选)
        if len(vocab) % 100 == 0:
            print(f"Vocab size: {len(vocab)} / {vocab_size}, Merged: {best_pair} -> {new_id}")

    return vocab, merges
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
    
    # 1. 初始化 Vocab
    vocab = {i: bytes([i]) for i in range(256)}
    vocab_inv = {bytes([i]): i for i in range(256)}
    
    sorted_special_tokens = sorted(special_tokens, key=len, reverse=True)
    for st in special_tokens:
        st_bytes = st.encode("utf-8")
        if st_bytes not in vocab_inv:
            new_id = len(vocab)
            vocab[new_id] = st_bytes
            vocab_inv[st_bytes] = new_id

    # 2. 准备正则 (GPT-2 标准正则)
    GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    tokenizer_regex = re.compile(GPT2_SPLIT_PATTERN)

    # Special Token 切分正则
    if sorted_special_tokens:
        st_pattern = "(" + "|".join(re.escape(t) for t in sorted_special_tokens) + ")"
    else:
        st_pattern = None

    train_data = Counter()

    # 3. 读取数据 (使用 f.read() 确保跨行匹配正确)
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
        
        # A. 先切分 Special Tokens
        if st_pattern:
            parts = re.split(st_pattern, text)
        else:
            parts = [text]

        for part in parts:
            if part in special_tokens:
                continue
            if not part:
                continue
            
            # B. 运行 GPT-2 正则
            chunks = re.findall(tokenizer_regex, part)

            for chunk in chunks:
                chunk_bytes = chunk.encode("utf-8")
                ids = tuple(chunk_bytes) 
                train_data[ids] += 1
    
    # 4. BPE 循环
    merges = []
    while len(vocab) < vocab_size:
        stats = Counter()
        for ids, count in train_data.items():
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i+1])
                stats[pair] += count

        if not stats:
            break

        # =========================================================
        # 关键修正：Tie-Breaking 使用 Bytes 内容比较，而不是 ID
        # =========================================================
        best_pair = max(stats, key=lambda p: (stats[p], vocab[p[0]], vocab[p[1]]))
        
        # Merge 操作
        new_id = len(vocab)
        part1 = vocab[best_pair[0]]
        part2 = vocab[best_pair[1]]
        vocab[new_id] = part1 + part2
        merges.append((part1, part2))

        # 更新 train_data
        new_train_data = {}
        for ids, count in train_data.items():
            if best_pair[0] not in ids:
                new_train_data[ids] = count
                continue
            
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == best_pair[0] and ids[i+1] == best_pair[1]:
                    new_ids.append(new_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            new_train_data[tuple(new_ids)] = count
        train_data = new_train_data

    return vocab, merges
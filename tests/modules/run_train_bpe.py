import os
import re
from collections import Counter

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
    # 1、初始化vocab, 0-255 bytes + special_tokens
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for i_st in special_tokens:
        vocab[len(vocab)] = i_st.encode("utf-8")

    # 2、从input_path读取训练数据并进行预切分,进行词频（转换为bytes）的统计
    word_counts = Counter()
    tokenizer_regex = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            word_list_per_line: list[str] = re.findall(tokenizer_regex, line)

            for i_w in word_list_per_line:
                i_w = i_w.encode("utf-8")
                i_w_bytes = tuple(bytes([i_]) for i_ in i_w)
                word_counts[i_w_bytes] += 1    

    # 3、进行merge，直到vocab里面的元素达到vocab_size
    while len(vocab) <= vocab_size:
        # 进行bigram的统计
        tmp_counter = Counter()
        for item, count in word_counts.items():
            for i_item in range(len(item)-1):
                tmp_counter[item[i_item]+item[i_item+1]] += count

        # 计算得到出现频率最高的bigram，如果有多个，则输出ASCII值最大的那个
        max_count = max(tmp_counter.values())
        max_bigram_list = [item for item, count in tmp_counter.items() if count == max_count]
        max_bigram = max(max_bigram_list)
        # 更新vocab
        vocab[len(vocab)] = max_bigram
        # 更新word_counts
        new_word_counts = Counter()
        for item, count in word_counts.items():
            tmp_list = []
            for i_new_item in range(len(item)-1):
                item_current, item_next = item[i_new_item], item[i_new_item+1]
                if item_current+item_next == max_bigram:
                    tmp_list.append(max_bigram)
                else:
                    tmp_list.extend([item_current, item_next])
            new_word_counts[tuple(tmp_list)] = count
        word_counts = new_word_counts

import json
import time

# from .adapters import run_train_bpe
from .modules.run_train_bpe import run_train_bpe
from .common import DATA_PATH, gpt2_bytes_to_unicode


def test_train_bpe_speed():
    """
    Ensure that BPE training is relatively efficient by measuring training
    time on this small dataset and throwing an error if it takes more than 1.5 seconds.
    This is a pretty generous upper-bound, it takes 0.38 seconds with the
    reference implementation on my laptop. In contrast, the toy implementation
    takes around 3 seconds.
    """
    input_path = DATA_PATH / "TinyStoriesV2-GPT4-train.txt"
    start_time = time.time()
    _, _ = run_train_bpe(
        input_path=input_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )
    end_time = time.time()
    print(end_time - start_time)
    assert end_time - start_time < 1800
"""GPT-2 BPE (via tiktoken) plus reserved chat tokens

The model's vocabulary is padded from 50257 to 50304 (a multiple of 64, which
matmuls prefer). The spare ids hold the chat-format tokens used for SFT, so the
pretrained model already has embedding rows for them.
"""

from functools import lru_cache

import tiktoken

VOCAB_SIZE = 50304
EOT = 50256  # <|endoftext|>, also put before every pretraining document
SPECIAL_TOKENS = {
    "<|endoftext|>": EOT,
    "<|system|>": 50257,
    "<|user|>": 50258,
    "<|assistant|>": 50259,
    "<|end|>": 50260,
}


@lru_cache(maxsize=1)
def get_tokenizer() -> tiktoken.Encoding:
    base = tiktoken.get_encoding("gpt2")
    return tiktoken.Encoding(
        name="gpt2-chat",
        pat_str=base._pat_str,
        mergeable_ranks=base._mergeable_ranks,
        special_tokens=SPECIAL_TOKENS,
    )

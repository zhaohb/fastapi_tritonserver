import os
from transformers.generation.utils import GenerationConfig


def parse_cfg(path: str):
    cfg = GenerationConfig.from_pretrained(path)
    if isinstance(cfg.eos_token_id, list):
        end_id = cfg.eos_token_id[0]
    else:
        end_id = cfg.eos_token_id
    return {
        "end_id": end_id,
        "pad_id": end_id,
        "top_k": cfg.top_k,
        "top_p": cfg.top_p,
        "temperature": cfg.temperature,
        "len_penalty": 1,
        "repetition_penalty": cfg.repetition_penalty
    }
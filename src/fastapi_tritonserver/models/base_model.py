import numpy as np
from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self):
        pass

    def encode(self, tokenizer, query, system_prompt, history_list: list):
        pass

    def decode(self, tokenizer, output_ids, inputs_token_lens, cutoff_len=0):
        new_ids = [[]]
        for id in output_ids[0]:
            new_ids[0].append(id[cutoff_len:])
        new_ids = np.array(new_ids)
        return tokenizer.batch_decode(new_ids[0], skip_special_tokens=True)

    def make_context(self):
        pass

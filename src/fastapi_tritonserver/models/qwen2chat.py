from .base_model import BaseModel
from fastapi_tritonserver.logger import _root_logger

logger = _root_logger

DEFAULT_PROMPT_TEMPLATES = {
    'InternLMForCausalLM':
    "<|User|>:{input_text}<eoh>\n<|Bot|>:",
    'qwen':
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n",
    'Qwen2ForCausalLM':
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n",
}


class Qwen2ChatModel(BaseModel):

    def __init__(self):
        pass

    def encode(
            self,
            tokenizer,
            query,
            system_prompt="You are a helpful assistant.",
            history_list=None
    ):
        # use make_content to generate prompt
        # print("input_id_list len", len(input_id_list))
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        for (old_query, old_response) in history_list:
            messages.append(
                {"role": "user", "content": old_query}
            )
            messages.append(
                {"role": "assistant", "content": old_response}
            )
        messages.append({"role": "user", "content": query})
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        encoded_outputs = tokenizer(
            text,
            # add_special_tokens=True,
            # truncation=True,
            # max_length=1024
        )
        return encoded_outputs


    def decode(self, tokenizer, output_ids, input_lengths, cutoff_len=0):

        return tokenizer.decode(output_ids[0][0])
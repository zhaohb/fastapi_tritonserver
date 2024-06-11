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
        if isinstance(query, str):
            messages.append({"role": "user", "content": query})
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        # print("prompt: ", prompt)
        # used in function call
        if not isinstance(query, str):
            im_end = "<|im_end|>"
            # right trip
            prompt = prompt[: -len("<|im_start|>assistant") - 1]
            prompt = prompt.rstrip()
            prompt = prompt[: -len(im_end)]
            # stop_words.append(im_end)
        encoded_outputs = tokenizer(
            prompt,
        )
        return encoded_outputs

    def decode(self, tokenizer, output_ids, input_lengths, cutoff_len=0):
        return tokenizer.decode(output_ids[0][0],  skip_special_tokens=True)
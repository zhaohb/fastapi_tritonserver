from .base_model import BaseModel
from transformers import AutoTokenizer
from fastapi_tritonserver.logger import _root_logger
from typing import List
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import requests
import torch
from typing import Tuple, List, Union
from fastapi_tritonserver.logger import _root_logger

logger = _root_logger

class QwenvlModel(BaseModel):

    def __init__(self):
        image_size = 448

        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)

        self.image_transform = transforms.Compose([
            transforms.Resize(
                (image_size,image_size),
                interpolation = InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,std=std),

        ])

    def encode(self, image_paths: List[str]):
        images = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if image_paths:
            for image_path in image_paths:
                if image_path.startswith("http://") or image_path.startswith("https://"):
                    try:
                        image = requests.get(image_path,stream=True, timeout = 3).raw
                    except requests.exceptions.Timeout as e:
                        logger.info(e)
                        return torch.tensor([])
                    except exceptions.MissingSchema as e:
                        logger.info(e)
                        return torch.tensor([])
                    image = Image.open(image)
                else:
                    image = Image.open(image_path)
                image = image.convert("RGB")
                images.append(self.image_transform(image))
            images = torch.stack(images, dim=0)
            return images
        else:
            return torch.tensor([])

    def decode(self, tokenizer, output_ids, input_lengths, cutoff_len=0):
        ### Fix me: For now, although the incoming images are in array format, only single images are supported
        new_ids = output_ids[0][0, input_lengths:]
        return tokenizer.decode(new_ids, skip_special_tokens=True)

    def make_context(self, tokenizer, prompt: str, images: List[str], visual_output):
        ## fix me: images is an array, but for now I'll just take the first one
        image = images[0]
        content_list = []
        content_list.append({'image': image})
        content_list.append({'text': prompt})
        query = tokenizer.from_list_format(content_list)

        def qwenvl_make_context(
            tokenizer,
            query: str,
            history: List[Tuple[str, str]] = None,
            system: str = "You are a helpful assistant.",
            max_window_size: int = 6144,
            chat_format: str = "chatml",
        ):
            if history is None:
                history = []
        
            if chat_format == "chatml":
                im_start, im_end = "<|im_start|>", "<|im_end|>"
                im_start_tokens = [tokenizer.im_start_id]#151644
                im_end_tokens = [tokenizer.im_end_id]#[151645]
                nl_tokens = tokenizer.encode("\n")
        
                def _tokenize_str(role, content):
                    return f"{role}\n{content}", tokenizer.encode(
                        role, allowed_special=set(tokenizer.IMAGE_ST)
                    ) + nl_tokens + tokenizer.encode(content, allowed_special=set(tokenizer.IMAGE_ST))
        
                system_text, system_tokens_part = _tokenize_str("system", system)
                system_tokens = im_start_tokens + system_tokens_part + im_end_tokens
        
                raw_text = ""
                context_tokens = []
        
                for turn_query, turn_response in reversed(history):
                    query_text, query_tokens_part = _tokenize_str("user", turn_query)
                    query_tokens = im_start_tokens + query_tokens_part + im_end_tokens
                    if turn_response is not None:
                        response_text, response_tokens_part = _tokenize_str(
                            "assistant", turn_response
                        )
                        response_tokens = im_start_tokens + response_tokens_part + im_end_tokens
        
                        next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
                        prev_chat = (
                            f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
                        )
                    else:
                        next_context_tokens = nl_tokens + query_tokens + nl_tokens
                        prev_chat = f"\n{im_start}{query_text}{im_end}\n"
        
                    current_context_size = (
                        len(system_tokens) + len(next_context_tokens) + len(context_tokens)
                    )
                    if current_context_size < max_window_size:
                        context_tokens = next_context_tokens + context_tokens
                        raw_text = prev_chat + raw_text
                    else:
                        break
        
                context_tokens = system_tokens + context_tokens
                raw_text = f"{im_start}{system_text}{im_end}" + raw_text
                context_tokens += (
                    nl_tokens
                    + im_start_tokens
                    + _tokenize_str("user", query)[1]
                    + im_end_tokens
                    + nl_tokens
                    + im_start_tokens
                    + tokenizer.encode("assistant")
                    + nl_tokens
                )
                raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"
            elif chat_format == "raw":
                raw_text = query
                context_tokens = tokenizer.encode(raw_text)
            else:
                raise NotImplementedError(f"Unknown chat format {chat_format!r}")
        
            return raw_text, context_tokens

        raw_text, context_tokens = qwenvl_make_context(tokenizer, query,history=None)

        input_ids = torch.tensor([context_tokens])
        bos_pos = torch.where(input_ids == 151857) ## self.config.visual['image_start_id']
        eos_pos = torch.where(input_ids == 151858) ## self.config.visual['image_start_id'] + 1
        assert (bos_pos[0] == eos_pos[0]).all()
        img_pos = torch.stack((bos_pos[0], bos_pos[1], eos_pos[1]), dim=1)

        vocab_size = 151936 ##self.config.vocab_size
        fake_prompt_id = torch.arange(vocab_size,
                                        vocab_size +
                                        visual_output.shape[0] * visual_output.shape[1])
        fake_prompt_id = fake_prompt_id.reshape(visual_output.shape[0],
                                                    visual_output.shape[1])
        for idx, (i, a, b) in enumerate(img_pos):
            input_ids[i][a + 1 : b] = fake_prompt_id[idx]
        input_ids = input_ids.contiguous().to(torch.int32)
        input_lengths = torch.tensor(input_ids.size(1), dtype=torch.int32)

        visual_output = torch.tensor(visual_output)
        prompt_table = visual_output.numpy()

        return input_ids, input_lengths, prompt_table

from typing import List
from triton_server_helper.client import StreamClient, AsyncStreamClient
import numpy as np
import time
import csv
from ..sampling_params import SamplingParams
from .engine import Engine, AsyncEngine
from tritonclient.utils import np_to_triton_dtype
import tritonclient.grpc.aio as grpcclient
from fastapi_tritonserver.logger import _root_logger
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import requests
import torch
from typing import Tuple, List, Union
from ..models.base_model import BaseModel
from ..models.qwenvl import QwenvlModel
from ..models.qwen2chat import Qwen2ChatModel
from fastapi_tritonserver.ctx import app_ctx
from fastapi_tritonserver.constants import app_constants

logger = _root_logger


def prepare_tensor(name, input):
    client_util = grpcclient
    t = client_util.InferInput(name, input.shape,
                               np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t


def to_word_list_format(word_dict: List[List[str]],
                        tokenizer=None,
                        add_special_tokens=False):
    """
    format of word_dict
        len(word_dict) should be same to batch_size
        word_dict[i] means the words for batch i
        len(word_dict[i]) must be 1, which means it only contains 1 string
        This string can contain several sentences and split by ",".
        For example, if word_dict[2] = " I am happy, I am sad", then this function will return
        the ids for two short sentences " I am happy" and " I am sad".
    """
    assert tokenizer is not None, "need to set tokenizer"

    flat_ids = []
    offsets = []
    for word_dict_item in word_dict:
        item_flat_ids = []
        item_offsets = []

        if isinstance(word_dict_item[0], bytes):
            word_dict_item = [word_dict_item[0].decode()]

        words = list(csv.reader(word_dict_item))[0]
        for word in words:
            ids = tokenizer.encode(word, add_special_tokens=add_special_tokens)

            if len(ids) == 0:
                continue

            item_flat_ids += ids
            item_offsets.append(len(ids))

        flat_ids.append(np.array(item_flat_ids))
        offsets.append(np.cumsum(np.array(item_offsets)))

    pad_to = max(1, max(len(ids) for ids in flat_ids))

    for i, (ids, offs) in enumerate(zip(flat_ids, offsets)):
        flat_ids[i] = np.pad(ids, (0, pad_to - len(ids)), constant_values=0)
        offsets[i] = np.pad(offs, (0, pad_to - len(offs)), constant_values=-1)

    return np.array([flat_ids, offsets], dtype="int32").transpose((1, 0, 2))


def create_inputs(
        model,
        tokenizer,
        query: str,
        system_prompt: str,
        history_list: list,
        params: SamplingParams,
        images: List[str],
        visual_output,
        model_type: str = None,
        streaming: bool = False
):

    if np.size(visual_output) != 0 and model_type.endswith('-vl'):
        input_start_ids, input_lens, prompt_table = model.make_context(
            tokenizer, query, images, visual_output
        )
        input_start_ids = input_start_ids.numpy()
        input_lens = input_lens.numpy()
    else:
        encoded_inputs = model.encode(
            tokenizer,
            query,
            system_prompt=system_prompt,
            history_list=history_list
        )
        input_start_ids = encoded_inputs['input_ids']
        input_lens = np.sum(encoded_inputs['attention_mask'], axis=-1).reshape(-1, 1)
   
    inputs_token_lens = len(input_start_ids)

    input_lens = input_lens.astype(np.int32).reshape(-1, 1)
    inputs_shape = np.ones_like(input_lens)
    output0_len = inputs_shape.astype(np.int32) * params.max_output_len
    streaming_data = np.array([[streaming]], dtype=bool)

    inputs = [
        prepare_tensor("input_ids", np.array([input_start_ids]).astype(np.int32)),
        prepare_tensor("input_lengths", input_lens),
        prepare_tensor("request_output_len", output0_len),
        prepare_tensor("streaming", streaming_data),
    ]

    if np.size(visual_output) != 0 and model_type.endswith('-vl'):
        inputs.append(prepare_tensor("prompt_embedding_table", prompt_table.astype(np.float16)))
        inputs.append(prepare_tensor("prompt_vocab_size", np.array([[256]]).astype(np.int32)))
        inputs.append(prepare_tensor("task", np.array([[0]]).astype(np.int32)))

    if params.beam_width:
        beam_width = (inputs_shape * params.beam_width).astype(np.int32)
        inputs.append(prepare_tensor("beam_width", beam_width))

    if params.temperature:
        temperature = (inputs_shape * params.temperature).astype(np.float32)
        inputs.append(prepare_tensor("temperature", temperature))

    if params.top_k:
        top_k = (inputs_shape * params.top_k).astype(np.int32)
        inputs.append(prepare_tensor("runtime_top_k", top_k))

    if params.top_p:
        top_p = (inputs_shape * params.top_p).astype(np.float32)
        inputs.append(prepare_tensor("runtime_top_p", top_p))

    if params.presence_penalty:
        presence_penalty = (inputs_shape * params.presence_penalty).astype(np.float32)
        inputs.append(prepare_tensor("presence_penalty", presence_penalty))

    if params.repetition_penalty:
        repetition_penalty = (inputs_shape * params.repetition_penalty).astype(np.float32)
        inputs.append(prepare_tensor("repetition_penalty", repetition_penalty))

    if params.len_penalty:
        len_penalty = (inputs_shape * params.len_penalty).astype(np.float32)
        inputs.append(prepare_tensor("len_penalty", len_penalty))

    if params.random_seed:
        random_seed = (inputs_shape * params.random_seed).astype(np.uint64)
        inputs.append(prepare_tensor("random_seed", random_seed))

    if params.end_id:
        end_id = (inputs_shape * params.end_id).astype(np.int32)
        inputs.append(prepare_tensor("end_id", end_id))

    if params.stop_words:
        stop_words = to_word_list_format([params.stop_words], tokenizer)
        inputs.append(prepare_tensor("stop_words_list", stop_words))

    return inputs, inputs_token_lens

def decode(tokenizer, output_ids: np.ndarray, cutoff_len=0):

    new_ids = [[]]
    for id in output_ids[0]:
        new_ids[0].append(id[cutoff_len:])
    new_ids = np.array(new_ids)
    return tokenizer.batch_decode(new_ids[0], skip_special_tokens=True)


class TritonServerAsyncEngine(AsyncEngine):
    def __init__(self, url: str, tokenizer_path: str, model_name: str, model_type: str):
        super().__init__(tokenizer_path)
        self._model_name = model_name
        self._client = AsyncStreamClient(url, [model_name])
        self._model_type = model_type
        if self._model_type:
            self._model = globals()[app_constants['model_dict'][self._model_type]]()
        else:
            self._model = BaseModel()

    async def is_server_live(self):
        return await self._client.is_server_live()

    async def wait_ready(self):
        await self._client.wait_server_ready()

    async def visual_infer(self, images: List[str], timeout: int = 60000, request_id: str = '', 
                           only_return_output=False):
        if self._model_type == 'qwen-vl':
            image_size = 448

            before_process_time = time.time()
            images = self._model.encode(images)
            #image_pre_obj = QwenvlPreprocess(image_size)
            #images = image_pre_obj.encode(images)

            if torch.numel(images) != 0:
                before_create_inputs_time = time.time()
                inputs = [
                    prepare_tensor("input", images.numpy()),
                ]

                before_infer_time = time.time()
                response_iterator = self._client.infer({
                    'model_name': self._model_name,
                    'inputs': inputs,
                    # 'request_id': request_id
                }, timeout)

                async for response in response_iterator:
                    result, error = response
                    if error:
                        raise Exception(error)
                    else:
                        after_infer_time = time.time()
                        logger.info('[%s] generate elapsed times preprocess_time: [%.4fms], '
                                    'infer_time: [%.4fms]',
                                    request_id,
                                    (before_create_inputs_time - before_process_time) * 1000,
                                    (after_infer_time - before_infer_time) * 1000)
                        return result.as_numpy('output')
            else:
                return np.array([])

    async def generate(
        self,
        query: str,
        system_prompt: str,
        history_list: list,
        params: SamplingParams,
        images: List[str] = [],
        visual_output=None,
        timeout: int = 60000,
        request_id: str = '',
        only_return_output=False
    ):
        params = self.merge_default_params(params)
        logger.info('[%s] req merged generate_params: [%s] timeout: [%s]', request_id, params.to_json(), timeout)
        before_create_inputs_time = time.time()
        inputs, inputs_token_lens = create_inputs(
            self._model,
            self._tokenizer,
            query=query,
            system_prompt=system_prompt,
            history_list=history_list,
            params=params,
            images=images,
            visual_output=visual_output,
            model_type=self._model_type,
            streaming=False
        )
        cutoff_len = inputs_token_lens if only_return_output else 0

        before_infer_time = time.time()
        response_iterator = self._client.infer({
            'model_name': self._model_name,
            'inputs': inputs,
            # 'request_id': request_id
        }, 6000)

        async for response in response_iterator:
            result, error = response
            if error:
                raise Exception(error)
            else:
                before_decode_time = time.time()
                decoded = self._model.decode(self._tokenizer, result.as_numpy('output_ids'), inputs_token_lens, cutoff_len)
                #if self._model_type == 'qwen-vl':
                #    decoded = qwenvl_decode(self._tokenizer, result, inputs_token_lens, cutoff_len)
                #else:
                #    decoded = decode(self._tokenizer, result, cutoff_len)
                logger.info('[%s] generate elapsed times create_input: [%.4fms], '
                            'infer_time: [%.3fs], decoded_input: [%.4fms]',
                            request_id,
                            (before_infer_time - before_create_inputs_time) * 1000,
                            (before_decode_time - before_infer_time),
                            (time.time() - before_decode_time) * 1000)
                return decoded

    async def generate_streaming(
        self,
        query: str,
        system_prompt: str,
        history: list,
        params: SamplingParams,
        timeout: int = 60000,
        request_id: str = ''
    ):
        params = self.merge_default_params(params)
        logger.info('[%s] req merged generate_params: [%s] timeout: [%s]', request_id, params.to_json(), timeout)
        before_create_inputs_time = time.time()
        # inputs, inputs_token_lens = create_inputs(self._tokenizer, prompt, params, True)
        inputs, inputs_token_lens = create_inputs(
            self._model,
            self._tokenizer,
            query=query,
            system_prompt=system_prompt,
            history_list=history,
            params=params,
            images=[],  # images,
            visual_output=None,  # visual_output,
            model_type=self._model_type,
            streaming=True
        )

        before_infer_time = time.time()
        response_iterator = self._client.infer({
            'model_name': self._model_name,
            'inputs': inputs,
            # 'request_id': request_id
        }, timeout)

        before_infer_time = time.time()
        queue_output_ids = []
        async for response in response_iterator:
            result, error = response
            if error:
                raise Exception(error)
            else:
                output_ids = result.as_numpy('output_ids')
                if len(queue_output_ids) > 0:
                    queue_output_ids.append(output_ids)
                    output_ids = np.concatenate(queue_output_ids, axis=-1)
                output_texts = decode(self._tokenizer, output_ids)
                is_ok = True
                for temp_text in output_texts:
                    if b"\xef\xbf\xbd" in temp_text.encode():
                        is_ok = False
                if is_ok:
                    yield output_texts
                    queue_output_ids = []
                else:
                    if len(queue_output_ids) == 0:
                        queue_output_ids.append(output_ids)

        logger.info('[%s] generate elapsed times create_input: [%.4fms], infer_time: [%.3fs]',
                    request_id,
                    (before_infer_time - before_create_inputs_time) * 1000,
                    time.time() - before_infer_time)


class TritonServerEngine(Engine):
    def __init__(self, url: str, tokenizer_path: str, model_name: str, model_type: str):
        super().__init__(tokenizer_path)
        self._model_name = model_name
        self._client = StreamClient(url, [model_name])
        self._model_type = model_type

    def is_server_live(self) -> bool:
        return self._client.is_server_live()

    def wait_ready(self):
        self._client.wait_server_ready()

    def generate(self, prompt: str, params: SamplingParams, timeout: int = 10000,
                 request_id: str = '', only_return_output=False, streaming: bool = False):
        params = self.merge_default_params(params)
        logger.info('[%s] req merged generate_params: [%s] timeout: [%s]', request_id, params.to_json(), timeout)
        inputs, inputs_token_lens = create_inputs(self._tokenizer, prompt, params, streaming)
        result = self._client.infer({
            'model_name': self._model_name,
            'inputs': inputs,
            'request_id': request_id
        }, timeout)
        cutoff_len = inputs_token_lens if only_return_output else 0
        return decode(self._tokenizer, result, cutoff_len)

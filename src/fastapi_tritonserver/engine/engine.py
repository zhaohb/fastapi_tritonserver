import os.path
from abc import ABC, abstractmethod
from fastapi_tritonserver.sampling_params import SamplingParams
from transformers import AutoTokenizer
from fastapi_tritonserver.logger import _root_logger
from fastapi_tritonserver.utils.generate_cfg import parse_cfg

logger = _root_logger


class BaseEngine(ABC):
    def __init__(self, tokenizer_path):
        logger.info("init tokenizer tokenizer_path: [%s]", tokenizer_path)
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self._default_params = parse_cfg(tokenizer_path)

    def merge_default_params(self, params: SamplingParams) -> SamplingParams:
        """
        overwrite SamplingParams with default generate config value
        """
        if params.end_id is None and "end_id" in self._default_params:
            params.end_id = self._default_params["end_id"]
        if params.pad_id is None and "pad_id" in self._default_params:
            params.pad_id = self._default_params["pad_id"]
        if params.top_k is None and "top_k" in self._default_params:
            params.top_k = self._default_params["top_k"]
        if params.top_p is None and "top_p" in self._default_params:
            params.top_p = self._default_params["top_p"]
        if params.temperature is None and "temperature" in self._default_params:
            params.temperature = self._default_params["temperature"]
        if params.len_penalty is None and "len_penalty" in self._default_params:
            params.len_penalty = self._default_params["len_penalty"]
        if params.repetition_penalty is None and "repetition_penalty" in self._default_params:
            params.repetition_penalty = self._default_params["repetition_penalty"]
        if (
                params.stop_words is None
                or (isinstance(params.stop_words, list) and len(params.stop_words) == 0)
        ) and "stop" in self._default_params:
            params.stop_words = self._default_params["stop"]
        return params


class Engine(BaseEngine):

    @abstractmethod
    def is_server_live(self) -> bool:
        pass

    @abstractmethod
    async def generate(
        self,
        query: str,
        system_prompt: str,
        history_list: list,
        params: SamplingParams,
        images: list = [],
        visual_output=None,
        timeout: int = 60000,
        request_id: str = '',
        only_return_output=False
    ):
        pass

    @abstractmethod
    async def wait_ready(self):
        pass


class AsyncEngine(BaseEngine):
    @abstractmethod
    async def is_server_live(self) -> bool:
        pass

    @abstractmethod
    async def generate(
        self,
        query: str,
        system_prompt: str,
        history_list: list,
        params: SamplingParams,
        images: list = [],
        visual_output=None,
        timeout: int = 60000,
        request_id: str = '',
        only_return_output=False
    ):
        pass

    @abstractmethod
    async def wait_ready(self):
        pass

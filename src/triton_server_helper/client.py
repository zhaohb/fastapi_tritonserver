import time, asyncio
from typing import List
import queue
import threading
from functools import partial
import tritonclient.grpc as grpcclient
import tritonclient.grpc.aio as asyncgrpcclient
from tritonclient.utils import InferenceServerException
from fastapi_tritonserver.logger import _root_logger

logger = _root_logger


class AsyncStreamClient:
    def __init__(self, url: str, readiness_models: List[str] = []):
        self.url = url
        self._readiness_models = readiness_models
        self._client = asyncgrpcclient.InferenceServerClient(url=url)

    async def is_server_live(self) -> bool:
        return await self._client.is_server_live()

    async def wait_server_ready(self):
        live = await self._client.is_server_live()
        while not live:
            logger.debug("wait_server_ready live: [%s]", live)
            await asyncio.sleep(1)
            try:
                live = await self._client.is_server_live()
            except Exception as e:
                logger.warn("wait server ready err: %s", str(e))

        await self.wait_models_ready()

    async def wait_models_ready(self):
        all_ready = False
        while not all_ready:
            curr_state = True
            for model_name in self._readiness_models:
                ready = await self._client.is_model_ready(model_name)
                if not ready:
                    curr_state = False
            all_ready = curr_state
            await asyncio.sleep(1)

    async def get_model_meta(self, model_name):
        return await self._client.get_model_metadata(model_name)

    def infer(self, request, timeout: int = 1000):
        """
        yield:
            - output_ids
            - sequence_length
            - cum_log_probs
            - output_log_probs
        """
        async def async_request_iterator():
            yield {
                **request,
                "sequence_start": True,
                "sequence_end": True,
            }

        # Start streaming
        response_iterator = self._client.stream_infer(
            inputs_iterator=async_request_iterator(),
            stream_timeout=timeout,
        )

        return response_iterator


class UserData:

    def __init__(self):
        self._completed_requests = queue.Queue()

    @property
    def completed_requests(self):
        return self._completed_requests


def callback(user_data, result, error):
    if error:
        user_data.completed_requests.put(error)
    else:
        user_data.completed_requests.put(result)


class StreamClient:
    def __init__(self, url: str, readiness_models: List[str] = []):
        self.url = url
        self._readiness_models = readiness_models
        self._client = grpcclient.InferenceServerClient(url=url)
        self._lock = threading.Lock()

    def is_server_live(self) -> bool:
        return self._client.is_server_live()

    def wait_server_ready(self):
        live = False
        while not live:
            time.sleep(1)
            try:
                live = self._client.is_server_live()
            except Exception as e:
                logger.warn("wait server ready err: %s", str(e))

        self.wait_models_ready()

    def wait_models_ready(self):
        all_ready = False
        while not all_ready:
            curr_state = True
            for model_name in self._readiness_models:
                ready = self._client.is_model_ready(model_name)
                if not ready:
                    curr_state = False
            all_ready = curr_state
            time.sleep(1)

    def get_model_meta(self, model_name):
        return self._client.get_model_metadata(model_name)

    # todo: add thread lock
    def infer(self, request, timeout: int = 1000):
        user_data = UserData()
        with self._lock:
            try:
                # Establish stream
                self._client.start_stream(callback=partial(callback, user_data))
                # Send request
                self._client.async_stream_infer(request['model_name'], request['inputs'], timeout=timeout)
            finally:
                # Wait for server to close the stream
                self._client.stop_stream()

            # Parse the responses
            while True:
                try:
                    result = user_data.completed_requests.get(block=False)
                except Exception:
                    break

                if type(result) == InferenceServerException:
                    raise result
                else:
                    return result

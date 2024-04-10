import argparse
import sys

import uvicorn
import uuid
import time
import asyncio
import logging
import os
from threading import Thread
from http import HTTPStatus
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi_tritonserver.logger import _root_logger, get_formatter
from fastapi_tritonserver.config import ServerConf
from fastapi_tritonserver.engine.tritonserver import TritonServerAsyncEngine, TritonServerEngine
from fastapi_tritonserver.protocols.fastapi import GenerateRequest
from fastapi_tritonserver.sampling_params import SamplingParams
from fastapi_tritonserver.ctx import app_ctx
from fastapi_tritonserver.entrypoints import openai_api
from tritonclient.utils import InferenceServerException

logger = _root_logger

TIMEOUT_KEEP_ALIVE = 60  # seconds.


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    init triton server client
    """
    # reset uvicorn.access logger format
    uvicorn_logger = logging.getLogger("uvicorn.access")
    uvicorn_logger.handlers[0].setFormatter(get_formatter())

    server_args = parse_args()
    server_conf = ServerConf.from_cli_args(server_args)
    logger.info("worker initiating engine. sever_conf: %s", server_conf)

    app_ctx['model_type'] = server_conf.model_type
    if server_conf.model_type.endswith('-vl'):
        llm_model_name = server_conf.model_name + '-llm'
        visual_model_name = server_conf.model_name + '-visual'

        app_ctx["asyncEngine"] = TritonServerAsyncEngine(server_conf.server_url, server_conf.tokenizer_path, llm_model_name, server_conf.model_type)
        app_ctx["asyncVisualEngine"] = TritonServerAsyncEngine(server_conf.server_url, server_conf.tokenizer_path, visual_model_name, server_conf.model_type)
    else:
        app_ctx["asyncEngine"] = TritonServerAsyncEngine(server_conf.server_url, server_conf.tokenizer_path,
                                                     server_conf.model_name, server_conf.model_type)
    logger.info("worker waiting engine ready")
    await app_ctx["asyncEngine"].wait_ready()
    yield
    logger.info("worker exited")


app = FastAPI(lifespan=lifespan)
app.include_router(openai_api.router)


def create_error_response(status_code: HTTPStatus,
                          message: str, type: str) -> JSONResponse:
    return JSONResponse({"message": message, "type": type},
                        status_code=status_code.value)


@app.exception_handler(InferenceServerException)
async def validation_exception_handler(request, exc):
    return create_error_response(HTTPStatus.BAD_REQUEST, str(exc), "infer_err")


@app.exception_handler(ValueError)
async def validation_exception_handler(request, exc):
    return create_error_response(HTTPStatus.BAD_REQUEST, str(exc), "param_err")


def get_request_id(request:GenerateRequest):
    if request.uuid is not None and len(request.uuid) > 0:
        return request.uuid
    else:
        return str(uuid.uuid4())


@app.post("/generate")
async def generate(raw_request: Request) -> Response:
    """
    Generate completion for the request.
    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - max_output_len: Maximum number of tokens to generate per output sequence.
    - num_beams: the beam width when use beam search.
    - repetition_penalty
    - top_k: Integer that controls the number of top tokens to consider.
        Set to -1 to consider all tokens.
    - top_p: Float that controls the cumulative probability of the top tokens
        to consider. Must be in (0, 1]. Set to 1 to consider all tokens.
    - temperature: Float that controls the randomness of the sampling. Lower
        values make the model more deterministic, while higher values make
        the model more random. Zero means greedy sampling.
    """
    request_dict = await raw_request.json()
    request = GenerateRequest(**request_dict)
    request_id = get_request_id(request)
    params = SamplingParams(**request_dict)
    # Non-streaming response
    if not request.stream:
        begin = time.time()
        try:
            if app_ctx['model_type'].endswith("-vl"):
                logger.info('[%s] req request: [%s] generate_params: [%s]', request_id, request_dict, params.to_json())
                visual_output = await app_ctx["asyncVisualEngine"].visual_infer(request.images, request.timeout, request_id, request.only_return_output)
                text = await app_ctx["asyncEngine"].generate(request.prompt, params, request.images, visual_output, request.timeout, request_id, request.only_return_output)
            else:
                logger.info('[%s] req request: [%s] generate_params: [%s]', request_id, request_dict, params.to_json())
                text = await app_ctx["asyncEngine"].generate(request.prompt, params, request.timeout, request_id, request.only_return_output)
        except Exception as e:
            logger.error('[%s] process fail msg: [%s]', request_id, str(e))
            raise e
        logger.info('[%s] resp elapsed: [%.4fs] result: [%s]', request_id, time.time() - begin, text)
        ret = {"text": text, "id": request_id}
        return JSONResponse(ret)
    else:
        app_ctx["asyncEngine"].generate_streaming(request.prompt, params, request.timeout, request_id)


@app.post("/batch_generate")
async def batch_generate(raw_request: Request) -> Response:
    request_dict = await raw_request.json()
    request = GenerateRequest(**request_dict)
    request_id = get_request_id(request)
    params = SamplingParams(**request_dict)
    begin = time.time()
    try:
        logger.info('[%s] req request: [%s] generate_params: [%s]', request_id, request_dict, params.to_json())
        futures = []
        for i, p in enumerate(request.prompts):
            futures.append(
                app_ctx["asyncEngine"].generate(p, params, request.timeout, request_id + '|' + str(i), request.only_return_output))

        texts = await asyncio.gather(*futures)
    except Exception as e:
        logger.error('[%s] process fail msg: [%s]', request_id, str(e))
        raise e
    logger.info('[%s] resp elapsed: [%.4fs] result: [%s]', request_id, time.time() - begin, texts)
    ret = {"texts": texts, "id": request_id}
    return JSONResponse(ret)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--model_type", type=str, default="qwen-vl")
    parser = ServerConf.add_cli_args(parser)
    return parser.parse_args()


def wait_engine_ready(engine:TritonServerEngine):
    engine.wait_ready()


def health_check(args, engine:TritonServerEngine):
    def server_live_job():
        err_cnt = 0
        while True:
            try:
                time.sleep(2)
                if not engine.is_server_live():
                    logger.warning('server is not live')
                    err_cnt = err_cnt + 1
            except Exception as e:
                logger.warning('sever_live_job err: %s', e)
                err_cnt = err_cnt + 1
            if err_cnt > 5:
                cmd = "lsof -i :" + str(args.port) + " | awk '{print $2}' | grep -v PID| xargs kill -9"
                logger.error('server is not live > 5 times, exit!, exec cmd: [%s]', cmd)
                os.system(cmd)

    thread = Thread(target=server_live_job)
    thread.start()
    return thread


if __name__ == "__main__":
    logger.info('fastapi-trt-llm-server start')
    args = parse_args()
    logger.info(f"args: {args}")
    logger.info("initiating engine.")

    server_conf = ServerConf.from_cli_args(args)
    model_type = server_conf.model_type
    logger.info(f"model_type: {server_conf.model_type}")

    if server_conf.model_type.endswith('-vl'):
        llm_model_name = server_conf.model_name + '-llm'
        visual_model_name = server_conf.model_name + '-visual'

        llm_engine = TritonServerEngine(server_conf.server_url, server_conf.tokenizer_path, llm_model_name, server_conf.model_type)
        wait_engine_ready(llm_engine)
        health_task = health_check(args, llm_engine)
        visual_engine = TritonServerEngine(server_conf.server_url, server_conf.tokenizer_path, visual_model_name, server_conf.model_type)
        wait_engine_ready(visual_engine)
        health_task = health_check(args, visual_engine)
    else:
        engine = TritonServerEngine(server_conf.server_url, server_conf.tokenizer_path, server_conf.model_name, server_conf.model_type)
        wait_engine_ready(engine)
        health_task = health_check(args, engine)

    logger.info("start http server")
    uvicorn.run("__main__:app",
                host=args.host,
                port=args.port,
                workers=args.workers,
                use_colors=False,
                reload=True,
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
    health_task.join()



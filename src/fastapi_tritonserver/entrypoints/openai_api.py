import argparse
import uvicorn
import time
import re
import logging
import os
import json
import copy
from threading import Thread
from http import HTTPStatus
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi_tritonserver.logger import _root_logger, get_formatter
from fastapi_tritonserver.config import ServerConf
from fastapi_tritonserver.engine.tritonserver import TritonServerAsyncEngine, TritonServerEngine
from fastapi_tritonserver.protocols.openai import (
    ModelCard, ModelList, ChatMessage, DeltaMessage,
    ChatCompletionRequest, ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionResponse, ChatCompletionStreamResponse,
    UsageInfo
)

from fastapi_tritonserver.sampling_params import SamplingParams
from fastapi_tritonserver.ctx import app_ctx
from tritonclient.utils import InferenceServerException
from fastapi_tritonserver.utils.tools import random_uuid
from typing import List, Literal, Optional, Union, Dict
from fastapi import APIRouter

_TEXT_COMPLETION_CMD = object()

logger = _root_logger
router = APIRouter(prefix="/openai", tags=["OpenAI API"])

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
app.include_router(router)


def trim_stop_words(response, stop_words):
    if stop_words:
        for stop in stop_words:
            idx = response.find(stop)
            if idx != -1:
                response = response[:idx]
    return response

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


TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters}"""

REACT_INSTRUCTION = """Answer the following questions as best you can. You have access to the following APIs:

{tools_text}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tools_name_text}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!"""


_TEXT_COMPLETION_CMD = object()


def parse_messages(messages, functions):
    if all(m.role != "user" for m in messages):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid request: Expecting at least one user message.",
        )

    messages = copy.deepcopy(messages)
    default_system = "You are a helpful assistant."
    system = ""
    if messages[0].role == "system":
        system = messages.pop(0).content.lstrip("\n").rstrip()
        if system == default_system:
            system = ""

    if functions:
        tools_text = []
        tools_name_text = []
        for func_info in functions:
            name = func_info.get("name", "")
            name_m = func_info.get("name_for_model", name)
            name_h = func_info.get("name_for_human", name)
            desc = func_info.get("description", "")
            desc_m = func_info.get("description_for_model", desc)
            tool = TOOL_DESC.format(
                name_for_model=name_m,
                name_for_human=name_h,
                # Hint: You can add the following format requirements in description:
                #   "Format the arguments as a JSON object."
                #   "Enclose the code within triple backticks (`) at the beginning and end of the code."
                description_for_model=desc_m,
                parameters=json.dumps(func_info["parameters"], ensure_ascii=False),
            )
            tools_text.append(tool)
            tools_name_text.append(name_m)
        tools_text = "\n\n".join(tools_text)
        tools_name_text = ", ".join(tools_name_text)
        system += "\n\n" + REACT_INSTRUCTION.format(
            tools_text=tools_text,
            tools_name_text=tools_name_text,
        )
        system = system.lstrip("\n").rstrip()

    dummy_thought = {
        "en": "\nThought: I now know the final answer.\nFinal answer: ",
        "zh": "\nThought: 我会作答了。\nFinal answer: ",
    }

    _messages = messages
    messages = []
    for m_idx, m in enumerate(_messages):
        role, content, func_call = m.role, m.content, m.function_call
        if content:
            content = content.lstrip("\n").rstrip()
        if role == "function":
            if (len(messages) == 0) or (messages[-1].role != "assistant"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid request: Expecting role assistant before role function.",
                )
            messages[-1].content += f"\nObservation: {content}"
            if m_idx == len(_messages) - 1:
                messages[-1].content += "\nThought:"
        elif role == "assistant":
            if len(messages) == 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid request: Expecting role user before role assistant.",
                )
            last_msg = messages[-1].content
            last_msg_has_zh = len(re.findall(r"[\u4e00-\u9fff]+", last_msg)) > 0
            if func_call is None:
                if functions:
                    content = dummy_thought["zh" if last_msg_has_zh else "en"] + content
            else:
                f_name, f_args = func_call["name"], func_call["arguments"]
                if not content:
                    if last_msg_has_zh:
                        content = f"Thought: 我可以使用 {f_name} API。"
                    else:
                        content = f"Thought: I can use {f_name}."
                content = f"\n{content}\nAction: {f_name}\nAction Input: {f_args}"
            if messages[-1].role == "user":
                messages.append(
                    ChatMessage(role="assistant", content=content.lstrip("\n").rstrip())
                )
            else:
                messages[-1].content += content
        elif role == "user":
            messages.append(
                ChatMessage(role="user", content=content.lstrip("\n").rstrip())
            )
        else:
            raise HTTPException(
                status_code=400, detail=f"Invalid request: Incorrect role {role}."
            )

    query = _TEXT_COMPLETION_CMD
    if messages[-1].role == "user":
        query = messages[-1].content
        messages = messages[:-1]

    if len(messages) % 2 != 0:
        print(376)
        raise HTTPException(status_code=400, detail="Invalid request")

    history = []  # [(Q1, A1), (Q2, A2), ..., (Q_last_turn, A_last_turn)]
    for i in range(0, len(messages), 2):
        if messages[i].role == "user" and messages[i + 1].role == "assistant":
            usr_msg = messages[i].content.lstrip("\n").rstrip()
            bot_msg = messages[i + 1].content.lstrip("\n").rstrip()
            if system and (i == len(messages) - 2):
                usr_msg = f"{system}\n\nQuestion: {usr_msg}"
                system = ""
            for t in dummy_thought.values():
                t = t.lstrip("\n")
                if bot_msg.startswith(t) and ("\nAction: " in bot_msg):
                    bot_msg = bot_msg[len(t) :]
            history.append([usr_msg, bot_msg])
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid request: Expecting exactly one user (or function) role before every assistant role.",
            )
    if system:
        assert query is not _TEXT_COMPLETION_CMD
        query = f"{system}\n\nQuestion: {query}"
    return query, history


async def get_gen_prompt(request):
    prev_messages = request.messages[:-1]
    if len(prev_messages) > 0 and prev_messages[0].role == "system":
        system = prev_messages.pop(0).content
    else:
        system = "You are a helpful assistant."

    query, history = parse_messages(request.messages, request.functions)

    return query, history, system


def parse_response(response):
    func_name, func_args = "", ""
    i = response.rfind("\nAction:")
    j = response.rfind("\nAction Input:")
    k = response.rfind("\nObservation:")
    if 0 <= i < j:  # If the text has `Action` and `Action input`,
        if k < j:  # but does not contain `Observation`,
            # then it is likely that `Observation` is omitted by the LLM,
            # because the output text may have discarded the stop word.
            response = response.rstrip() + "\nObservation:"  # Add it back.
        k = response.rfind("\nObservation:")
        func_name = response[i + len("\nAction:") : j].strip()
        func_args = response[j + len("\nAction Input:") : k].strip()
    if func_name:
        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(
                role="assistant",
                content=response[:i],
                function_call={"name": func_name, "arguments": func_args},
            ),
            finish_reason="function_call",
        )
        return choice_data
    z = response.rfind("\nFinal Answer: ")
    if z >= 0:
        response = response[z + len("\nFinal Answer: ") :]
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=response),
        finish_reason="stop",
    )
    return choice_data


@app.post("/v1/chat/completions")
async def create_chat_completion(raw_request: ChatCompletionRequest):

    request_dict = raw_request.json()
    request_json = json.loads(request_dict)
    request = ChatCompletionRequest(**request_json)

    begin = time.time()
    logger.info(f"Received chat completion request: {request}")

    stop_words = []
    if request.stop is not None and len(request.stop) > 0:
        if isinstance(request.stop, str):
            stop_words.append(request.stop)
        else:
            stop_words.append(','.join(request.stop))
    
    prev_messages = request.messages[:-1]
    if len(prev_messages) > 0 and prev_messages[0].role == "system":
        system = prev_messages.pop(0).content
    else:
        system = "You are a helpful assistant."

    query, history = parse_messages(request.messages, request.functions)

    print("query: ", query)
    print("history: ", history)
    if request.stream and request.functions:
        raise HTTPException(
            status_code=400,
            detail="Invalid request: Function calling is not yet implemented for stream mode.",
        )

    model_name = request.model
    request_id = f"{random_uuid()}"
    created_time = int(time.time())

    if request.messages[-1].role not in ["user", "function"]:
        print(454)
        raise HTTPException(status_code=400, detail="Invalid request")
    # query = request.messages[-1].content
    prev_messages = request.messages[:-1]
    if len(prev_messages) > 0 and prev_messages[0].role == "system":
        system = prev_messages.pop(0).content
    else:
        system = "You are a helpful assistant."

    if request.functions:
        stop_words = stop_words or []
        if "Observation:" not in stop_words:
            stop_words.append("Observation:")

    params = SamplingParams(
        max_output_len=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        beam_width=request.n,
        presence_penalty=request.presence_penalty,
        frequency_penalty=request.frequency_penalty,
        stop_words=stop_words,
    )

    def create_stream_response_json(
            index: int,
            text: str,
            finish_reason: Optional[str] = None,
    ) -> str:
        choice_data = ChatCompletionResponseStreamChoice(
            index=index,
            delta=DeltaMessage(content=text),
            finish_reason=finish_reason,
        )
        response = ChatCompletionStreamResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=[choice_data],
        )
        response_json = response.model_dump_json()

        return response_json

    async def completion_stream_generator() -> AsyncGenerator[str, None]:
        texts = []

        # first chunk do
        for i in range(request.n):
            texts.append("")

        async for res in app_ctx["asyncEngine"].generate_streaming(
            query=query,
            system_prompt=system,
            history=history,
            params=params,
            request_id=request_id,
            # output_accumulate=request.output_accumulate
        ):
            for i, output in enumerate(res):
                texts[i] += output
                response_json = create_stream_response_json(
                    index=i,
                    text=output,
                )
                yield f"data: {response_json}\n\n"

        # last chunk with role
        for i in range(request.n):
            choice_data = ChatCompletionResponseStreamChoice(
                index=i,
                delta=DeltaMessage(role="assistant"),
                finish_reason="stop",
            )
            chunk = ChatCompletionStreamResponse(
                id=request_id,
                choices=[choice_data],
                model=model_name,
            )
            data = chunk.model_dump_json()
            yield f"data: {data}\n\n"

        yield "data: [DONE]\n\n"
        logger.info('[%s] resp elapsed: [%.4fs] result: [%s]', request_id, time.time() - begin, texts)

    # Streaming response
    if request.stream:
        # background_tasks = BackgroundTasks()
        # Abort the request if the client disconnects.
        # background_tasks.add_task(abort_request)
        return StreamingResponse(completion_stream_generator(),
                                 media_type="text/event-stream")
    try:
        logger.info('[%s] req request: [%s] generate_params: [%s]', request_id, request_dict, params.to_json())
        response = await app_ctx["asyncEngine"].generate(
            query=query,
            system_prompt=system,
            history_list=history,
            params=params,
            timeout=6000,
            request_id=request_id
        )
    except Exception as e:
        logger.error('[%s] process fail msg: [%s]', request_id, str(e))
        raise e
    logger.info(
        '[%s] resp elapsed: [%.4fs] result: [%s]',
        request_id,
        time.time() - begin,
        response
    )
    response = trim_stop_words(response, stop_words)
    if request.functions:
        choice_data = parse_response(response)
    else:
        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role="assistant", content=response),
            finish_reason="stop",
        )
    response = ChatCompletionResponse(
        id=request_id,
        created=created_time,
        model=model_name,
        choices=[choice_data],
        usage=UsageInfo(),
    )
    if request.stream:
        resp = response.json(ensure_ascii=False)

        async def fake_stream_generator() -> AsyncGenerator[str, None]:
            yield f"data: {resp}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(fake_stream_generator(),
                                 media_type="text/event-stream")
    return response


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--model_type", type=str, default="qwen2-chat")
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

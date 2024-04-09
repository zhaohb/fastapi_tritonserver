#!/bin/bash
python3 -m fastapi_tritonserver.entrypoints.openai_api  \
      --port 9900 --host 0.0.0.0 \
      --model-name tensorrt_llm  \
      --tokenizer-path ${TOKENIZER_PATH} \
      --server-url ${TRITON_SERVER_HOST}:${TRITON_SERVER_PORT}  \
      --model_type ${MODEL_TYPE}
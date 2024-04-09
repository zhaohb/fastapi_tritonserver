FROM python:3.10-bullseye as builder

# install torch
RUN pip install --upgrade pip \
    && pip install torch

FROM python:3.10-bullseye as final
COPY --from=builder /usr/local /usr/local
RUN mkdir /app
WORKDIR /app
COPY . /app
# RUN chmod +x ./start_api.sh
RUN chmod +x ./start_openapi.sh

ENV TOKENIZER_PATH=""
ENV TRITON_SERVER_HOST="127.0.0.1"
ENV TRITON_SERVER_PORT="8001"
ENV MODEL_TYPE="qwen2-chat"

EXPOSE 9900

RUN python -m pip install --upgrade pip && \
    pip install .

# 启动程序
CMD ["./start_openapi.sh"]

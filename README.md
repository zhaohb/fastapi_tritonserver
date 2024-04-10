# 项目介绍
- 一个使用fastapi部署，封装tritonserver以支持openai API格式的项目。

# 准备工作
- 需要已部署tritonserver for trt-llm服务，[参考教程](./docs/deploy_triton.md)
- 目前仅支持tritonserver24.02 + tensorrt_llm 0.8.0部署的qwen1.5系列，其它未做测试。


## 安装本项目代码
- 源码安装
```shell
pip install .
```

- whl安装包安装（暂未提供）
```
pip install fastapi_tritonserver-0.0.1-py3-none-any.whl
```

- 从Pypi仓库安装（暂未实现）

## Api server  启动示例(普通的接口调用)
### 确保tritonserver已经启动，并切暴露在127.0.0.1:8001
- 假设你部署的qwen1.5-1.8b-chat模型，那么可以将tokenizer的本机离线路径或者在huggingface的在线路径提供给`--tokenizer-path`参考。
- workers可以根据你的tritonserver最大支持的batch_size来设置。
- 下面是一个简单示例
```shell
python3 -m fastapi_tritonserver.entrypoints.api_server  \
      --port 9900 \
      --host 0.0.0.0 \
      --model-name tensorrt_llm  \
      --tokenizer-path Qwen/Qwen1.5-1.8B-Chat \
      --server-url 127.0.0.1:8001  \
      --workers 4 \
      --model_type qwen2-chat
```


## 请求示例
```
curl -X POST  localhost:9100/generate -d '{
    "prompt": "who are you?"
}'
```
output:
```shell
{"text":"I am QianWen, a large language model created by Alibaba Cloud.","id":"89101ccc-d6d0-4cdf-a05c-8cbb7b466d66"}
```

## 参数说明
```
- prompt: 用于生成的提示。
- images: 只有vl模型需要这个输入
- max_output_len: 每个输出序列生成的最大令牌数。
- num_beams: 使用波束搜索时的波束宽度。
- repetition_penalty
- top_k: 控制要考虑的顶级令牌数的整数。设置为-1以考虑所有令牌。
- top_p: 控制要考虑的顶级令牌的累积概率的浮点数，必须在(0, 1]内。设置为1以考虑所有令牌。
- temperature: 控制采样随机性的浮点数。较低的值使模型更确定，而较高的值使模型更随机。零表示贪婪采样。
```

## Openai API 示例(符合openai api规范的调用)
### 确保tritonserver已经启动，并切暴露在127.0.0.1:8001, workers可以根据你的tritonserver最大支持的batch_size来设置。
```shell
python3 -m fastapi_tritonserver.entrypoints.openai_api  \
      --port 9900 \
      --host 0.0.0.0 \
      --model-name tensorrt_llm  \
      --tokenizer-path Qwen/Qwen1.5-1.8B-Chat \
      --server-url 127.0.0.1:8001  \
      --workers 4 \
      --model_type qwen2-chat
```

## 请求示例
```
curl -X POST localhost:9900/v1/chat/completions \ 
     -H "Content-Type: application/json" \
     -d '{
          "model": "gpt-3.5-turbo", \
          "messages": [{"role": "system", "content": "You are a helpful assistant."}, \
                       {"role": "user", "content": "who you are."}]
        }'
```
output:
```shell
{
    "model":"gpt-3.5-turbo",
    "object":"chat.completion",
    "choices":[{"index":0,"message":
                {"role":"assistant","content":"I am QianWen, a large language model created by Alibaba Cloud. I was trained on a vast amount of text data from the web, including books, articles, and other sources, to understand natural language and provide responses to various questions and tasks.\n\nMy primary function is to assist with a wide range of applications, including answering questions, generating text based on input prompts, summarizing long documents, translating languages, and even writing code. I can understand and generate human-like text in multiple languages, including English, Chinese, Spanish, French, German, Italian, Japanese, Korean, Russian, Portuguese, and more.\n\nQianW","function_call":null},
               "finish_reason":"stop"
              }],
    "created":1711955133}
```


## qwenvl 示例

```shell
# 本地启动
python3 -m fastapi_tritonserver.entrypoints.api_server --port 9000 --host 0.0.0.0 --model-name qwen-vl-test --tokenizer-path qwenvl_repo/qwen-vl-test/qwen-vl-test-llm/20240220104327/tokenizer/ --server-url localhost:6601 --workers 1 --model_type qwen-vl

# triton server启动
tritonserver --model-repository=qwenvl_repo/repo/ --strict-model-config=false --log-verbose=0 --metrics-port=6000 --http-port=6609 --grpc-port=6601

# 请求示例
curl -X POST  localhost:9000/generate -d '{"images": ["https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"], "prompt": "what it is"}'
```

### docker部署
1. 编译docker
```bash
docker build . -t fastapi_tritonserver
```

2. 运行docker
```bash
docker run -d --restart=always \
  -e TOKENIZER_PATH="Qwen/Qwen1.5-1.8B-Chat" \
  -e TRITON_SERVER_HOST="192.168.x.x" \
  -e TRITON_SERVER_PORT="8001" \
  -e MODEL_TYPE="qwen2-chat" \
  -e WORKERS=4 \
  --name fastapi_tritonserver \
  -p 9900:9900 \
  fastapi_tritonserver
```
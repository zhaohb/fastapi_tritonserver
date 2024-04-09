## 模型准备：
- 以Qwen1.5-1.8B-chat为例，其他规模的qwen1.5也是同样做法。
```shell
git clone https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat
```


## 模型转换：
```shell
git clone https://github.com/Tlntin/Qwen-TensorRT-LLM.git
cd example/qwen2
```
```shell
python build.py --hf_model_dir ./qwen/Qwen1.5-1.8B-Chat/      \
           --dtype float16      \
           --remove_input_padding    \
           --gpt_attention_plugin float16  \ 
           --enable_context_fmha    \
           --gemm_plugin float16   \
           --paged_kv_cache \
           --output_dir ./tmp/Qwen1.5
```
### 验证engine：
```shell
python3 run.py       \
       --tokenizer_dir ./qwen/Qwen1.5-1.8B-Chat/       \
       --engine_dir=./tmp/Qwen1.5
```
output:

```shell
[TensorRT-LLM] TensorRT-LLM version: 0.8.0Input [Text 0]: "<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
你好，请问你叫什么？<|im_end|>
<|im_start|>assistant
"
Output [Text 0 Beam 0]: "你好！我是来自阿里云的大规模语言模型，我叫通义千问。"
```
engine验证完成。

## 封装triton服务：
参考Qwen-TensorRT-LLM项目triton_model_rep目录，主要是修改config.obtxt文件中gpt_model_path字段，修改成engine的实际路径。
triton加载命令，triton镜像选择使用nvcr.io/nvidia/tritonserver:24.02-trtllm-python-py3 ：
```shell
CUDA_VISIBLE_DEVICES=0 tritonserver --model-repository=repo --strict-model-config=false --log-verbose=0 --metrics-port=6000
```
output:
```shell
I0328 03:16:20.315813 150 server.cc:634]
+-------------+-----------------------------------------------------------------+---------------------------------------------------------------------------------------------------+
| Backend     | Path                                                            | Config                                                                                            |
+-------------+-----------------------------------------------------------------+---------------------------------------------------------------------------------------------------+
| tensorrtllm | /opt/tritonserver/backends/tensorrtllm/libtriton_tensorrtllm.so | {"cmdline":{"auto-complete-config":"true","backend-directory":"/opt/tritonserver/backends","min-c |
|             |                                                                 | ompute-capability":"6.000000","default-max-batch-size":"4"}}                                      |
+-------------+-----------------------------------------------------------------+---------------------------------------------------------------------------------------------------+

I0328 03:16:20.315837 150 server.cc:677]
+--------------+---------+--------+
| Model        | Version | Status |
+--------------+---------+--------+
| tensorrt_llm | 1       | READY  |
+--------------+---------+--------+

I0328 03:16:20.929471 150 metrics.cc:877] Collecting metrics for GPU 0: NVIDIA A100-SXM4-80GB
I0328 03:16:20.996808 150 metrics.cc:770] Collecting CPU metrics
I0328 03:16:20.997046 150 tritonserver.cc:2508]
+----------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------+
| Option                           | Value                                                                                                                                          |
+----------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------+
| server_id                        | triton                                                                                                                                         |
| server_version                   | 2.43.0                                                                                                                                         |
| server_extensions                | classification sequence model_repository model_repository(unload_dependents) schedule_policy model_configuration system_shared_memory cuda_sha |
|                                  | red_memory binary_tensor_data parameters statistics trace logging                                                                              |
| model_repository_path[0]         | repo                                                                                                                                           |
| model_control_mode               | MODE_NONE                                                                                                                                      |
| strict_model_config              | 0                                                                                                                                              |
| rate_limit                       | OFF                                                                                                                                            |
| pinned_memory_pool_byte_size     | 268435456                                                                                                                                      |
| cuda_memory_pool_byte_size{0}    | 67108864                                                                                                                                       |
| min_supported_compute_capability | 6.0                                                                                                                                            |
| strict_readiness                 | 1                                                                                                                                              |
| exit_timeout                     | 30                                                                                                                                             |
| cache_enabled                    | 0                                                                                                                                              |
+----------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------+

I0328 03:16:20.998511 150 grpc_server.cc:2519] Started GRPCInferenceService at 0.0.0.0:8001
I0328 03:16:20.998732 150 http_server.cc:4637] Started HTTPService at 0.0.0.0:8000
I0328 03:16:21.039690 150 http_server.cc:320] Started Metrics Service at 0.0.0.0:6000
```
现实加载成功，grpc接口是0.0.0.0:8001，这个会作为模型的url被fastapi-tritonserver使用。
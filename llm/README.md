# application-genai-quantization-tool

config env

1. clone code

   ```
   git clone https://github.com/openvinotoolkit/openvino.genai.git
   ```

2. create virtual env

   ```
   virtualenv convert-llm
   ```

3. source virtual env and install python dependencies

   ```
   source convert-llm/bin/activate
   pip install -r tools/llm_bench/requirements.txt
   ```

4. convert model

   ```
   optimum-cli export openvino --framework pt -m $input_model_path --weight-format int4 --trust-remote-code --task text-generation-with-past $output_model_path
   ```

Convert records

| model name                 | convert command                                              | Weight compression mode | % all parameters (layers) | % ratio-defining parameters (layers) | status  | run     |
| -------------------------- | ------------------------------------------------------------ | ----------------------- | ------------------------- | ------------------------------------ | ------- | ------- |
| Baichuan2-7B-Chat          | optimum-cli export openvino --framework pt -m ./Baichuan2-7B-Chat --weight-format int4 --trust-remote-code --task text-generation-with-past ./Baichuan2-7B-Chat-IR/ | int8_asym               | 31% (31 / 162)            | 21% (29 / 160)                       |         |         |
|                            |                                                              | int4_sym                | 69% (131 / 162)           | 79% (131 / 160)                      | success | success |
| Baichuan-7B                | optimum-cli export openvino --framework pt -m ./Baichuan-7B --weight-format int4 --trust-remote-code --task text-generation-with-past ./Baichuan-7B-IR | int8_asym               | 7% (2 / 162)              | 0% (0 / 160)                         |         |         |
|                            |                                                              | int4_sym                | 93% (160 / 162)           | 100% (160 / 160)                     | success | success |
| Meta-Llama-3.1-8B          | optimum-cli export openvino --framework pt -m ./Meta-Llama-3.1-8B --weight-format int4 --trust-remote-code --task text-generation-with-past ./Meta-Llama-3.1-8B-IR | int8_asym               | 31% (94 / 226)            | 20% (92 / 224)                       |         |         |
|                            |                                                              | int4_sym                | 69% (132 / 226)           | 80% (132 / 224)                      | success | success |
| Meta-Llama-3.2-3B-Instruct | optimum-cli export openvino --framework pt -m ./Meta-Llama-3.2-3B-Instruct --weight-format int4 --trust-remote-code --task text-generation-with-past ./Meta-Llama-3.2-3B-Instruct-IR | int8_asym               | 12% (1 / 197)             | 0% (0 / 196)                         |         |         |
|                            |                                                              | int4_sym                | 88% (196 / 197)           | 100% (196 / 196)                     | success | success |
| Meta-Llama-3-8B            | optimum-cli export openvino --framework pt -m ./Meta-Llama-3-8B --weight-format int4 --trust-remote-code --task text-generation-with-past ./Meta-Llama-3-8B-IR | int8_asym               | 13% (2 / 226)             | 0% (0 / 224)                         |         |         |
|                            |                                                              | int4_sym                | 87% (224 / 226)           | 100% (224 / 224)                     | success | success |
| Phi-3-mini-128k-instruct   | optimum-cli export openvino --framework pt -m ./Phi-3-mini-128k-instruct --weight-format int4 --trust-remote-code --task text-generation-with-past ./Phi-3-mini-128k-instruct-IR | int8_asym               | 5% (2 / 130)              | 0% (0 / 128)                         |         |         |
|                            |                                                              | int4_sym                | 95% (128 / 130)           | 100% (128 / 128)                     | success | success |
| Qwen2-1.5B                 | optimum-cli export openvino --framework pt -m ./Qwen2-1.5B --weight-format int4 --trust-remote-code --task text-generation-with-past ./Qwen2-1.5B-IR | int8_asym               | 15% (1 / 197)             | 0% (0 / 196)                         |         |         |
|                            |                                                              | int4_sym                | 85% (196 / 197)           | 100% (196 / 196)                     | success | success |
| Qwen2.5-7B                 | optimum-cli export openvino --framework pt -m ./Qwen2.5-7B --weight-format int4 --trust-remote-code --task text-generation-with-past ./Qwen2.5-7B-IR | int8_asym               | 14% (2 / 198)             | 0% (0 / 196)                         |         |         |
|                            |                                                              | int4_sym                | 86% (196 / 198)           | 100% (196 / 196)                     | success | success |
| Qwen2.5-7B-Instruct        | optimum-cli export openvino --framework pt -m ./Qwen2.5-7B-Instruct --weight-format int4 --trust-remote-code --task text-generation-with-past ./Qwen2.5-7B-Instruct-IR | int8_asym               | 14% (2 / 198)             | 0% (0 / 196)                         |         |         |
|                            |                                                              | int4_sym                | 86% (196 / 198)           | 100% (196 / 196)                     | success | success |
| Qwen2-7B                   | optimum-cli export openvino --framework pt -m ./Qwen2-7B --weight-format int4 --trust-remote-code --task text-generation-with-past ./Qwen2-7B-IR | int8_asym               | 14% (2 / 198)             | 0% (0 / 196)                         |         |         |
|                            |                                                              | int4_sym                | 86% (196 / 198)           | 100% (196 / 196)                     | success | success |
| Qwen2-7B-Instruct          | optimum-cli export openvino --framework pt -m ./Qwen2-7B-Instruct --weight-format int4 --trust-remote-code --task text-generation-with-past ./Qwen2-7B-Instruct-IR | int8_asym               | 14% (2 / 198)             | 0% (0 / 196)                         |         |         |
|                            |                                                              | int4_sym                | 86% (196 / 198)           | 100% (196 / 196)                     | success | success |
| chatglm3-6b                | optimum-cli export openvino --framework pt -m ./chatglm3-6b --weight-format int4 --trust-remote-code --task text-generation-with-past ./chatglm3-6b-IR | int8_asym               | 9% (3 / 115)              | 0% (0 / 112)                         |         |         |
|                            |                                                              | int4_sym                | 91% (112 / 115)           | 100% (112 / 112)                     | success | success |
| glm-4-9b                   | optimum-cli export openvino --framework pt -m ./glm-4-9b --weight-format int4 --trust-remote-code --task text-generation-with-past ./glm-4-9b-IR | int8_asym               | 13% (3 / 163)             | 0% (0 / 160)                         |         |         |
|                            |                                                              | int4_sym                | 87% (160 / 163)           | 100% (160 / 160)                     | success | success |
| glm-4-9b-chat              | optimum-cli export openvino --framework pt -m ./glm-4-9b-chat --weight-format int4 --trust-remote-code --task text-generation-with-past ./glm-4-9b-chat-IR | int8_asym               | 13% (3 / 163)             | 0% (0 / 160)                         |         |         |
|                            |                                                              | int4_sym                | 87% (160 / 163)           | 100% (160 / 160)                     | success | success |




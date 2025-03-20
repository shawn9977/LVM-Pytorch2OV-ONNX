#!/bin/bash

PRECISION=int4
models=(
    "/mnt/storage/bruce-dir/models/LLM/baichuan-inc/Baichuan2-7B-Chat"
    "/mnt/storage/bruce-dir/models/LLM/baichuan-inc/Baichuan-7B"
    "/mnt/storage/bruce-dir/models/LLM/meta-llama/Llama-3.1-8B"
    "/mnt/storage/bruce-dir/models/LLM/meta-llama/Llama-3.2-3B-Instruct"
    "/mnt/storage/bruce-dir/models/LLM/meta-llama/Meta-Llama-3-8B"
    "/mnt/storage/bruce-dir/models/LLM/LLM-Research/Meta-Llama-3.1-8B"
    "/mnt/storage/bruce-dir/models/LLM/LLM-Research/Meta-Llama-3.2-3B-Instruct"
    "/mnt/storage/bruce-dir/models/LLM/LLM-Research/Meta-Llama-3-8B"
    "/mnt/storage/bruce-dir/models/LLM/microsoft/Phi-3-mini-128k-instruct"
    "/mnt/storage/bruce-dir/models/LLM/Qwen/Qwen2-1.5B"
    "/mnt/storage/bruce-dir/models/LLM/Qwen/Qwen2.5-7B"
    "/mnt/storage/bruce-dir/models/LLM/Qwen/Qwen2.5-7B-Instruct"
    "/mnt/storage/bruce-dir/models/LLM/Qwen/Qwen2-7B"
    "/mnt/storage/bruce-dir/models/LLM/Qwen/Qwen2-7B-Instruct"
    "/mnt/storage/bruce-dir/models/LLM/THUDM/chatglm3-6b"
    "/mnt/storage/bruce-dir/models/LLM/THUDM/glm-4-9b"
    "/mnt/storage/bruce-dir/models/LLM/THUDM/glm-4-9b-chat"
    )

for model in "${models[@]}"
do
    output_dir="$(dirname "$model")/$(basename "$model")-IR"

    if [ ! -d "$output_dir" ]; then
        mkdir -p "$output_dir"
    fi

    echo "-----Converting: $model to $output_dir-----"
    optimum-cli export openvino --framework pt -m $model --weight-format $PRECISION --trust-remote-code --task text-generation-with-past $output_dir
    echo "-----The $model conversion to $output_dir is complete-----"

    sudo sync; sudo bash -c "echo 3 > /proc/sys/vm/drop_caches"
done



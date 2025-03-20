#
#Author: Bruce Yin
#
#
#Install Moelscope
#pip install modelscope
##Model Download
#m1:
#modelscope download --model Qwen/Qwen2-Audio-7B-Instruct
#modelscope download --model Qwen/Qwen2-Audio-7B-Instruct README.md --local_dir ./dir
#m2:
# from modelscope import snapshot_download
# model_dir = snapshot_download('Qwen/Qwen2-Audio-7B-Instruct')
#
#
#Install Huggingface-cli
#pip install -U huggingface_hub
#
#Set PRC mirror
#export HF_ENDPOINT=https://hf-mirror.com
#
##Model Downlaod
#
#huggingface-cli download --resume-download stabilityai/stable-diffusion-2-1 --local-dir ./
#huggingface-cli download --resume-download lllyasviel/fav_models fav/realisticStockPhoto_v20.safetensors --local-dir ./

import paramiko
import sys
import os
import argparse
import subprocess
import time
import traceback
import stat

from modelscope import snapshot_download
from pathlib import Path

#model_id shakechen/Llama-2-7b cache_dir /home/$USER/Downloads/
def pullModelfrommodelscope(model_id, cache_dir, local_only=False):

    try:
        model_dir = snapshot_download(model_id=model_id, cache_dir=cache_dir, local_files_only=local_only)
    except Exception as e:
        print(f"Error executing command {cmd}: {e}")

#model-id madebyollin/taesdxl local-dir /home/$USER/Downloads/
def pullModelfromHuggingface(modelid, localdir, token):
    try:
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        if not token is None:
            os.system(f'huggingface-cli download --token {token} --resume-download {modelid} --local-dir {localdir}')
        else:
            os.system(f'huggingface-cli download --resume-download {modelid} --local-dir {localdir}')
    except Exception as e:
        print(f"Error executing command hugggingface client download: {e}")

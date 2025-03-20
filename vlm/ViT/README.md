Refer:


Step1: Create one new virtual python environment.
conda create -n ov_py310 python=3.10 -y
conda activate ov_py310

Step2: Install python dependencies
pip install -r requirement.txt

Step3: Convert
#Original model to FP32
python3 convert.py --checkpoint=B_16_imagenet1k --output=/home/genai-dev/Downloads --batch_size=1

#Original model to FP16
python3 convert.py --checkpoint=B_16_imagenet1k --output=/home/genai-dev/Downloads --batch_size=1 --fp16

Step4: Quantization

#FP16 to INT8
python3 quantization.py --ov_model=/home/genai-dev/Downloads/vit_b_16_imagenet1k_bs1_fp16.xml --output=/home/genai-dev/Downloads/ --batch_size=1 --data_dir=/home/genai-dev/.cache/coco128/images/train2017

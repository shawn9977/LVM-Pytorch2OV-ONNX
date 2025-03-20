Refer:

Step1:

Create one new virtual python environment.


Step2:

pip install -r requirement.txt

Step3: 

Download the model

| model name           | URL                                                          |
| -------------------- | ------------------------------------------------------------ |
| sam_vit_h_4b8939.pth | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth |
| sam_vit_b_01ec64.pth | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth |
| sam_vit_l_0b3195.pth | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth |

Step4: Convert
#Original model to FP32
python3 convert.py --checkpoint=/mnt/storage/bruce-dir/models/VLM/facebookresearch/SAM/sam_vit_h_4b8939.pth --output=/home/genai-dev/Downloads/temp

#Original model to FP16
python3 convert.py --checkpoint=/mnt/storage/bruce-dir/models/VLM/facebookresearch/SAM/sam_vit_h_4b8939.pth --output=/home/genai-dev/Downloads/temp --fp16


Step5: Quantization

#FP16 to INT8

python3 quantization.py --model_path=/home/genai-dev/Downloads/temp/sam_vit_h_encoder_fp16.xml  --output=/home/genai-dev/Downloads/temp --datasets_path=/mnt/storage/bruce-dir/datasets/coco128/images/train2017


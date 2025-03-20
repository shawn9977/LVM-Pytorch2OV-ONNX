# Refer:
    https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/clip-zero-shot-image-classification/clip-zero-shot-classification.ipynb
    https://github.com/wayfeng/ov_clip/blob/main/README.md

# Step1:
  Create one new virtual python environment.

  $sudo apt-get install python3-pip
  $pip install virtualenv

  $virtualenv CLIP

# Step2:

  Note: The open_clip opensource code had updated, so we need one specific open_clip version from github. please follow the step to install dependencies.

  $pip install -r requirement.txt

# Step3:

  Download the model first 

  $huggingface-cli download laion/CLIP-ViT-B-32-laion2B-s34B-b79K open_clip_pytorch_model.bin --local-dir models --local-dir-use-symlinks False
  $mv open_clip_pytorch_model.bin open_clip-vit-b-32.pth

  $huggingface-cli download laion/CLIP-ViT-H-14-laion2B-s32B-b79K open_clip_pytorch_model.bin --local-dir models --local-dir-use-symlinks False
  $mv open_clip_pytorch_model.bin open_clip-vit-h-14.pth

  $huggingface-cli download laion/CLIP-ViT-g-14-laion2B-s34B-b88K open_clip_pytorch_model.bin --local-dir models --local-dir-use-symlinks False
  $mv open_clip_pytorch_model.bin open_clip-vit-g-14.pth

  $huggingface-cli download laion/CLIP-ViT-L-14-laion2B-s32B-B82k open_clip_pytorch_model.bin --local-dir models --local-dir-use-symlinks False
  $mv open_clip_pytorch_model.bin open_clip-vit-l-14.pth

# Step4: Convert
  # original to FP32
  $python3 ./convert.py --model_id ViT-B-32 --checkpoint /mnt/storage/bruce-dir/models/VLM/laion/OpenCLIP/open_clip_vit_b_32.pth --output /mnt/storage/bruce-dir/models/VLM/laion/OpenCLIP-temp --batch 1 --seperated

  # original to FP16
  $python3 ./convert.py --model_id ViT-B-32 --checkpoint /mnt/storage/bruce-dir/models/VLM/laion/OpenCLIP/open_clip_vit_b_32.pth --output /mnt/storage/bruce-dir/models/VLM/laion/OpenCLIP-temp --fp16 --batch 1 --seperated

# Step5: Quantization
  # FP16 to INT8
  $python3 quantization.py --model_id=ViT-B-32 --checkpoint=/mnt/storage/bruce-dir/models/VLM/laion/OpenCLIP/open_clip_vit_b_32.pth --model=/mnt/storage/bruce-dir/models/VLM/laion/OpenCLIP-IR/FP16/vit_b_32_visual_fp16.xml --output=/mnt/storage/bruce-dir/models/VLM/laion/OpenCLIP-IR/INT8 --fp16 --data_dir /mnt/storage/bruce-dir/datasets/

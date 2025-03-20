# Refer:
    https://blog.openvino.ai/blog-posts/enable-openvino-tm-optimization-for-groundingdino

# Step1:
  $sudo apt-get install python3-pip
  $pip install virtualenv
  $virtual GroundingDINO
  $source GroundingDINO/bin/activate

# Step2:
  $pip install -r requirements.txt
  $cd ../../thirdparts/OV-GroundingDINO
  $pip install -r requirements.txt
  $pip install -e .
  $cd ../../vlm/GroundingDINO/

# Step3:
  $mkdir weights
  $cd weights/
  $wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
  $wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
  $cd ..

  if you already downloaded the models, ignore the step.

# Step4:
  #original to FP32
  ##swint
  $python3 convert.py --checkpoint=/mnt/storage/bruce-dir/models/VLM/IDEA-Research/GroundingDINO/groundingdino_swint_ogc.pth --config=../../thirdparts/OV-GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py --output=/mnt/storage/bruce-dir/models/VLM/IDEA-Research/GroundingDINO-IR/FP32

  ##swinb
  $python3 convert.py --checkpoint=/mnt/storage/bruce-dir/models/VLM/IDEA-Research/GroundingDINO/groundingdino_swinb_cogcoor.pth --config=../../thirdparts/OV-GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py --output=/mnt/storage/bruce-dir/models/VLM/IDEA-Research/GroundingDINO-IR/FP32

  #original to FP16
  ##swint
  $python3 convert.py --checkpoint=/mnt/storage/bruce-dir/models/VLM/IDEA-Research/GroundingDINO/groundingdino_swint_ogc.pth --config=../../thirdparts/OV-GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py --output=/mnt/storage/bruce-dir/models/VLM/IDEA-Research/GroundingDINO-IR/FP16 --fp16

  ##swinb
  $python3 convert.py --checkpoint=/mnt/storage/bruce-dir/models/VLM/IDEA-Research/GroundingDINO/groundingdino_swinb_cogcoor.pth --config=../../thirdparts/OV-GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py --output=/mnt/storage/bruce-dir/models/VLM/IDEA-Research/GroundingDINO-IR/FP16 --fp16

  #FP16 to INT8
  ##swint
  $python3 quantization.py --checkpoint=/mnt/storage/bruce-dir/models/VLM/IDEA-Research/GroundingDINO/groundingdino_swint_ogc.pth --config=../../thirdparts/OV-GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py --ov_model=/mnt/storage/bruce-dir/models/VLM/IDEA-Research/GroundingDINO-IR/FP16/groundingdino_swint_fp16.xml --output=/mnt/storage/bruce-dir/models/VLM/IDEA-Research/GroundingDINO-IR/INT8 --fp16 --data_dir=/mnt/storage/bruce-dir/datasets/coco-2017

  ##swinb
  $python3 quantization.py --checkpoint=/mnt/storage/bruce-dir/models/VLM/IDEA-Research/GroundingDINO/groundingdino_swinb_cogcoor.pth --config=../../thirdparts/OV-GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py --ov_model=/mnt/storage/bruce-dir/models/VLM/IDEA-Research/GroundingDINO-IR/FP16/groundingdino_swinb_fp16.xml --output=/mnt/storage/bruce-dir/models/VLM/IDEA-Research/GroundingDINO-IR/INT8 --fp16 --data_dir=/mnt/storage/bruce-dir/datasets/coco-2017

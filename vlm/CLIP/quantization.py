import argparse, os
import torch
import torch.nn.functional as F
import open_clip
import openvino as ov
import warnings
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

import logging
import cv2
import nncf
import numpy as np
import torch.utils.data as data
from torchvision.transforms.functional import to_pil_image
from zipfile import ZipFile
from pathlib import Path

def transform_fn(image_data):
    """
    Quantization transform function. Extracts and preprocess input data from dataloader item for quantization.
    Parameters:
        image_data: image data produced by DataLoader during iteration
    Returns:
        input_tensor: input data in Dict format for model quantization
    """
    return preprocess_image(to_pil_image(np.squeeze(image_data.numpy()))).unsqueeze(0)


class COCOLoader(data.Dataset):
    def __init__(self, images_path):
        self.images = list(Path(images_path).iterdir())

    def __getitem__(self, index):
        image_path = self.images[index]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __len__(self):
        return len(self.images)

def load_data(data_dir: Path):
    zipfile = data_dir/'coco128.zip'

    if not (data_dir / "coco128/images/train2017").exists():
        with ZipFile(zipfile, "r") as zip_ref:
            zip_ref.extractall(zipfile)
    else:
        print("File existed")

    coco_dataset = COCOLoader(data_dir / 'coco128/images/train2017')
    calibration_loader = torch.utils.data.DataLoader(coco_dataset)

    return nncf.Dataset(calibration_loader, transform_fn)

def quantize_model(args):
    print("CLIP: start to quantize the model")
    core = ov.Core()

    path_parts = args.model.rsplit('/', 1)
    filename = path_parts[1]
    filename_without_extension, extension = filename.rsplit('.', 1)
    new_filename = f"{filename_without_extension}_to_int8.{extension}"
    target_model_path = os.path.join(args.output, new_filename)

    nncf.set_log_level(logging.ERROR)

    ov_model = core.read_model(args.model)
    calibration_dataset = load_data(Path(args.data_dir))

    quantized_model = nncf.quantize(
        model=ov_model,
        calibration_dataset=calibration_dataset,
        model_type=nncf.ModelType.TRANSFORMER,
        preset=nncf.QuantizationPreset.PERFORMANCE,
    )

    fp16_compress = args.fp16
 
    print("CLIP: save model with compress_to_fp16 : ", args.fp16)
    print("CLIP: model is saved :", target_model_path)
    ov.save_model(quantized_model, target_model_path, compress_to_fp16=fp16_compress)
    print("CLIP: model quantization finished")

def get_parser():
    parser = argparse.ArgumentParser(description='convert OpenCLIP model to OpenVINO IR')

    parser.add_argument('--model_id', dest="model_id", default='ViT-B-32', help='the OpenCLIP model_id, example: ViT-B-32, ViT-L-14, ViT-H-14, ViT-g-14')
    parser.add_argument('--checkpoint', dest="checkpoint",  help='the OpenCLIP checkpoint path')
    parser.add_argument('--model', dest='model', type=str, help='the model which need to be quantized, IR format')
    parser.add_argument('--output', dest='output', type=str, help='the path where to save thedd quantized model')
    parser.add_argument('--fp16', dest='fp16', default=False,  action='store_true', help='whether save model as fp16 mode')
    parser.add_argument('--data_dir', type=str, help="Data folder to calibrate model, please select the root directory of the coco128 dataset")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    print("CLIP: ", args)

    model_id = args.model_id
    if model_id not in ['ViT-B-32', 'ViT-L-14', 'ViT-H-14', 'ViT-g-14']:
        print(f"{model_id} is not in the support list")
        exit(0)
    
    pretrained = args.checkpoint
    model, _, preprocess_image = open_clip.create_model_and_transforms(model_id, pretrained=pretrained)

    quantize_model(args)
   
    print("CLIP: Exiting CLIP IR quantization ......")
    exit(0)


import os
from pathlib import Path
import torch
import openvino as ov
from typing import Tuple
import torch.utils.data as data
import nncf
import numpy as np
import argparse, traceback
import cv2
from copy import deepcopy
from typing import Tuple
from torchvision.transforms.functional import resize, to_pil_image

class ResizeLongestSide:
    """
    Resizes images to longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming numpy arrays.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        return np.array(resize(to_pil_image(image), target_size))

    def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(original_size[0], original_size[1], self.target_length)
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array shape Bx4. Requires the original image size
        in (H, W) format.
        """
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

resizer = ResizeLongestSide(1024)

def preprocess_image(image: np.ndarray):
    resized_image = resizer.apply_image(image)
    resized_image = (resized_image.astype(np.float32) - [123.675, 116.28, 103.53]) / [
        58.395,
        57.12,
        57.375,
    ]
    resized_image = np.expand_dims(np.transpose(resized_image, (2, 0, 1)).astype(np.float32), 0)

    # Pad
    h, w = resized_image.shape[-2:]
    padh = 1024 - h
    padw = 1024 - w
    x = np.pad(resized_image, ((0, 0), (0, 0), (0, padh), (0, padw)))
    return x


def postprocess_masks(masks: np.ndarray, orig_size):
    size_before_pad = resizer.get_preprocess_shape(orig_size[0], orig_size[1], masks.shape[-1])
    masks = masks[..., : int(size_before_pad[0]), : int(size_before_pad[1])]
    masks = torch.nn.functional.interpolate(torch.from_numpy(masks), size=orig_size, mode="bilinear", align_corners=False).numpy()
    return masks

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

def transform_fn(image_data):
    """
    Quantization transform function. Extracts and preprocess input data from dataloader item for quantization.
    Parameters:
        image_data: image data produced by DataLoader during iteration
    Returns:
        input_tensor: input data in Dict format for model quantization
    """
    image = image_data.numpy()
    processed_image = preprocess_image(np.squeeze(image))
    return processed_image

def quantize_encoder(ov_encoder_path, output_path, datasets_path, fp16):
    core = ov.Core()

    coco_dataset = COCOLoader(datasets_path)
    calibration_loader = torch.utils.data.DataLoader(coco_dataset)

    calibration_dataset = nncf.Dataset(calibration_loader, transform_fn)

    model = core.read_model(ov_encoder_path)
    quantized_model = nncf.quantize(
        model,
        calibration_dataset,
        model_type=nncf.parameters.ModelType.TRANSFORMER,
        preset=nncf.QuantizationPreset.PERFORMANCE,
        subset_size=128,
    )
    print("SAM: save model with compress_to_fp16 : ", fp16)
    print("SAM: model is saved :", output_path)
    ov.save_model(quantized_model, output_path, compress_to_fp16=fp16)
    print("SAM: model quantization finished")

def get_new_filename(original_path, suffix="_to_int8"):
    path_parts = original_path.rsplit('/', 1)
    filename = path_parts[1]
    filename_without_extension, extension = filename.rsplit('.', 1)
    return f"{filename_without_extension}{suffix}.{extension}"

def get_args_parser():
    parser = argparse.ArgumentParser(description='Help information')
    parser.add_argument('--model_path', dest='model_path', type=str, help='The path of the model to be quantized, example: ./models/sam_vit_h_encoder_fp16.xml')
    parser.add_argument('--output', dest='output', type=str, help='Model output path, example: ./models/sam_vit_h_ir')
    parser.add_argument('--datasets_path', dest='datasets_path', type=str, help='datasets path, example: coco128/images/train2017')
    parser.add_argument('--fp16', dest='fp16', default=False,  action='store_true', help='whether save model as fp16 mode')
    return parser.parse_args()

def main():
    args = get_args_parser()

    new_filename = get_new_filename(args.model_path)
    encoder_model_output_path = os.path.join(args.output, new_filename)

    print("Start to quantize the model to INT8")
    quantize_encoder(args.model_path, encoder_model_output_path, args.datasets_path, args.fp16)

    print("SAM: Exiting SAM IR quantization ......")
    exit(0)

if __name__ == "__main__":
    main()



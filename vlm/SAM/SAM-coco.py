from segment_anything import sam_model_registry
import warnings
from pathlib import Path
import torch
import openvino as ov
from typing import Tuple
import torch.utils.data as data
import nncf
import numpy as np
import argparse, traceback
import cv2

class SamExportableModel(torch.nn.Module):
    def __init__(
        self,
        model,
        return_single_mask: bool,
        use_stability_score: bool = False,
        return_extra_metrics: bool = False,
    ) -> None:
        super().__init__()
        self.mask_decoder = model.mask_decoder
        self.model = model
        self.img_size = model.image_encoder.img_size
        self.return_single_mask = return_single_mask
        self.use_stability_score = use_stability_score
        self.stability_score_offset = 1.0
        self.return_extra_metrics = return_extra_metrics

    def _embed_points(self, point_coords: torch.Tensor, point_labels: torch.Tensor) -> torch.Tensor:
        point_coords = point_coords + 0.5
        point_coords = point_coords / self.img_size
        point_embedding = self.model.prompt_encoder.pe_layer._pe_encoding(point_coords)
        point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding)

        point_embedding = point_embedding * (point_labels != -1).to(torch.float32)
        point_embedding = point_embedding + self.model.prompt_encoder.not_a_point_embed.weight * (point_labels == -1).to(torch.float32)

        for i in range(self.model.prompt_encoder.num_point_embeddings):
            point_embedding = point_embedding + self.model.prompt_encoder.point_embeddings[i].weight * (point_labels == i).to(torch.float32)

        return point_embedding

    def t_embed_masks(self, input_mask: torch.Tensor) -> torch.Tensor:
        mask_embedding = self.model.prompt_encoder.mask_downscaling(input_mask)
        return mask_embedding

    def mask_postprocessing(self, masks: torch.Tensor) -> torch.Tensor:
        masks = torch.nn.functional.interpolate(
            masks,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )
        return masks

    def select_masks(self, masks: torch.Tensor, iou_preds: torch.Tensor, num_points: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Determine if we should return the multiclick mask or not from the number of points.
        # The reweighting is used to avoid control flow.
        score_reweight = torch.tensor([[1000] + [0] * (self.model.mask_decoder.num_mask_tokens - 1)]).to(iou_preds.device)
        score = iou_preds + (num_points - 2.5) * score_reweight
        best_idx = torch.argmax(score, dim=1)
        masks = masks[torch.arange(masks.shape[0]), best_idx, :, :].unsqueeze(1)
        iou_preds = iou_preds[torch.arange(masks.shape[0]), best_idx].unsqueeze(1)

        return masks, iou_preds

    @torch.no_grad()
    def forward(
        self,
        image_embeddings: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        mask_input: torch.Tensor = None,
    ):
        sparse_embedding = self._embed_points(point_coords, point_labels)
        if mask_input is None:
            dense_embedding = self.model.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                point_coords.shape[0], -1, image_embeddings.shape[0], 64
            )
        else:
            dense_embedding = self._embed_masks(mask_input)

        masks, scores = self.model.mask_decoder.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embedding,
            dense_prompt_embeddings=dense_embedding,
        )

        if self.use_stability_score:
            scores = calculate_stability_score(masks, self.model.mask_threshold, self.stability_score_offset)

        if self.return_single_mask:
            masks, scores = self.select_masks(masks, scores, point_coords.shape[1])

        upscaled_masks = self.mask_postprocessing(masks)

        if self.return_extra_metrics:
            stability_scores = calculate_stability_score(upscaled_masks, self.model.mask_threshold, self.stability_score_offset)
            areas = (upscaled_masks > self.model.mask_threshold).sum(-1).sum(-1)
            return upscaled_masks, scores, stability_scores, areas, masks

        return upscaled_masks, scores


#
#pre process the data input for image encoder
#

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


def convert_encoder2ir(model, filename) :
    sam = sam_model_registry[model['model_type']](checkpoint=model['model_path'])
    ov_encoder_path = Path(model['output'] + "/" + filename)

    if not ov_encoder_path.exists():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            warnings.filterwarnings("ignore", category=UserWarning)

            ov_encoder_model = ov.convert_model(
                sam.image_encoder,
                example_input=torch.zeros(1, 3, 1024, 1024),
                input=([1, 3, 1024, 1024],),
           )
        ov.save_model(ov_encoder_model, ov_encoder_path, compress_to_fp16=True)


def convert_predictor2ir(model, filename):
    sam = sam_model_registry[model['model_type']](checkpoint=model['model_path'])
    ov_predictor_path = Path(model['output'] + "/" + filename)

    if not ov_predictor_path.exists():
        exportable_model = SamExportableModel(sam, return_single_mask=True)
        embed_dim = sam.prompt_encoder.embed_dim
        embed_size = sam.prompt_encoder.image_embedding_size
        dummy_inputs = {
            "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
            "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
            "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
        }
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            ov_model = ov.convert_model(exportable_model, example_input=dummy_inputs)

        ov.save_model(ov_model, ov_predictor_path)

def quantize_encoder(ov_encoder_path, output_path):
    core = ov.Core()

    coco_dataset = COCOLoader("/home/benchmark/bruce-dir/nncf_convert/datasets/coco128/images/train2017")
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

    ov.save_model(quantized_model, output_path)

sam_models = [
        {"checkpoint": "sam_vit_h_4b8939.pth", "model_path":"./models/sam_vit_h_4b8939.pth", "output":"./models/sam_vit_h", "model_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth", "model_type": "vit_h"},
        {"checkpoint": "sam_vit_b_01ec64.pth", "model_path":"./models/sam_vit_b_01ec64.pth", "output":"./models/sam_vit_b", "model_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth", "model_type": "vit_b"},
        {"checkpoint": "sam_vit_l_0b3195.pth", "model_path":"./models/sam_vit_l_0b3195.pth", "output":"./models/sam_vit_l", "model_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth", "model_type": "vit_l"}
]

def get_args_parser():
    parser = argparse.ArgumentParser(description='Help information')
    parser.add_argument('--id', dest='id', type=str, help='choose the id of the models')

    return parser.parse_args()

def main():
    args = get_args_parser()
    
    try:
        model = sam_models[int(args.id)]
        
        #To prepare the model
        #To prepare the data

        print("Start to convert model from pth to IR")
        convert_encoder2ir(model, model['checkpoint'][:9]+"_encoder_FP16.xml")
        convert_predictor2ir(model, model['checkpoint'][:9]+"_predictor_FP16.xml")

        print("Start to quantize the model to INT8")
        encoder_model_path= model['output']+"/"+model['checkpoint'][:9]+"_encoder_FP16.xml"
        encoder_model_output_path = model['output']+"/"+ model['checkpoint'][:9]+"_encoder_FP16_INT8.xml"
        quantize_encoder(encoder_model_path, encoder_model_output_path)
    except Exception as e:
        print(f"An error occurred during quantization: {e}")
        traceback.print_exc()
    finally:
        print("Done")

if __name__ == "__main__":
    main()


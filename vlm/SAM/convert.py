from segment_anything import sam_model_registry
import warnings, os
from pathlib import Path
import torch
import openvino as ov
from typing import Tuple
import argparse


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

def convert_encoder2ir(model_type, model_path, output, filename, fp16) :
    sam = sam_model_registry[model_type](checkpoint=model_path)
    ov_encoder_path = Path(os.path.join(output, filename))

    if not ov_encoder_path.exists():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            warnings.filterwarnings("ignore", category=UserWarning)

            ov_encoder_model = ov.convert_model(
                sam.image_encoder,
                example_input=torch.zeros(1, 3, 1024, 1024),
                input=([1, 3, 1024, 1024],),
           )
        print(f"Model saved to {ov_encoder_path}")
        ov.save_model(ov_encoder_model, ov_encoder_path, compress_to_fp16=fp16)

def convert_predictor2ir(model_type, model_path, output, filename) :
    sam = sam_model_registry[model_type](checkpoint=model_path)
    ov_predictor_path = Path(os.path.join(output, filename))

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
        print(f"Model saved to {ov_predictor_path}")
        ov.save_model(ov_model, ov_predictor_path)

def get_model_type(file_name):
    if "vit_h" in file_name:
        return "vit_h"
    elif "vit_b" in file_name:
        return "vit_b"
    elif "vit_l" in file_name:
        return "vit_l"
    else:
        raise ValueError(f"Invalid model file name: {file_name}")

def generate_ir_file_names(original_file_name, use_fp16):
    base_name = original_file_name[:9]
    file_extension = "_fp16.xml" if use_fp16 else "_fp32.xml"
    ir_encoder_file_name = f"{base_name}_encoder{file_extension}"
    ir_predictor_file_name = f"{base_name}_predictor{file_extension}"
    return ir_encoder_file_name, ir_predictor_file_name

def get_args_parser():
    parser = argparse.ArgumentParser(description='Help information')
    parser.add_argument('--checkpoint', dest="checkpoint",  help='The path of the model to be converted, example: ./models/sam_vit_h_4b8939.pth')
    parser.add_argument('--output', dest='output', type=str, help='Model output path, example: ./models/sam_vit_h_ir')
    # parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='default batch size')
    parser.add_argument('--fp16', dest='fp16', default=False, action='store_true', help='whether save model as fp16 mode')
    args = parser.parse_args()
    return args

def main():
    args = get_args_parser()

    original_file_name = os.path.basename(args.checkpoint)
    model_type = get_model_type(original_file_name)

    ir_encoder_file_name, ir_predictor_file_name = generate_ir_file_names(original_file_name, args.fp16)

    print("Start to convert model from pth to IR")
    convert_encoder2ir(model_type, args.checkpoint, args.output, ir_encoder_file_name, args.fp16)
    convert_predictor2ir(model_type, args.checkpoint, args.output, ir_predictor_file_name)

    print("SAM: Exiting SAM IR convert ......")
    exit(0)

if __name__ == "__main__":
    main()

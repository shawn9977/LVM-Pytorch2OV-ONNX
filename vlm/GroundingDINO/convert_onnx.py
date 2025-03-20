import os
import argparse
import torch
from groundingdino.util.slconfig import SLConfig
from groundingdino.models import build_model
from groundingdino.util.utils import clean_state_dict
from groundingdino.util import get_tokenlizer

def load_pt_grounding_dino(model_config_path, model_checkpoint_path):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cpu"
    args.use_checkpoint = False
    args.use_transformer_ckpt = False
    
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    
    return model, args.max_text_len, get_tokenlizer.get_tokenlizer(args.text_encoder_type)

def export_to_onnx(model, output_path):
    caption = "the running dog ."
    input_ids = torch.randint(0, 1000, (1, 6))
    position_ids = torch.tensor([[0, 0, 1, 2, 3, 0]])
    token_type_ids = torch.tensor([[0, 0, 0, 0, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1]])
    text_token_mask = torch.randint(0, 2, (1, 6, 6), dtype=torch.bool)
    samples = torch.randn(1, 3, 512, 512)
    
    dynamic_axes = {
        "samples": {0: "batch_size"},
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "position_ids": {0: "batch_size", 1: "seq_len"},
        "token_type_ids": {0: "batch_size", 1: "seq_len"},
        "text_token_mask": {0: "batch_size", 1: "seq_len", 2: "seq_len"},
        "logits": {0: "batch_size"},
        "boxes": {0: "batch_size"}
    }
    
    torch.onnx.export(
        model,
        (samples, input_ids, attention_mask, position_ids, token_type_ids, text_token_mask),
        output_path,
        input_names=["samples", "input_ids", "attention_mask", "position_ids", "token_type_ids", "text_token_mask"],
        output_names=["logits", "boxes"],
        dynamic_axes=dynamic_axes,
        opset_version=16,
    )
    print(f"Model successfully exported to {output_path}")

def get_parser():
    parser = argparse.ArgumentParser(description='Export GroundingDINO model to ONNX')
    parser.add_argument('--checkpoint', required=True, help='Path to the checkpoint file')
    parser.add_argument('--config', required=True, help='Path to the model config file')
    parser.add_argument('--output', required=True, help='Path to save the ONNX model')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_parser()
    model, _, _ = load_pt_grounding_dino(args.config, args.checkpoint)
    export_to_onnx(model, args.output)

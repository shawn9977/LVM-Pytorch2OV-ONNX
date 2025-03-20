import argparse
import torch
import open_clip

class TextTransformer(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, text):
        return self.model.encode_text(text)

def export_onnx(model, dummy_input, output_path, dynamic_axes=None):
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes or {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print(f"Model saved to {output_path}")

def convert_separate_model(args, model, batch, text_batch):
    print("Exporting image encoder...")
    image_input = torch.randn(batch, 3, 224, 224, dtype=torch.float32)
    export_onnx(model.visual, image_input, f"{args.output}/{args.model_id}_visual.onnx")
    
    print("Exporting text encoder...")
    text_encoder = TextTransformer(model)
    text_input = torch.randint(low=0, high=49407, size=(text_batch, 77))
    export_onnx(text_encoder, text_input, f"{args.output}/{args.model_id}_text.onnx")

def convert_whole_model(args, model, batch, text_batch):
    print("Exporting full model...")
    dummy_input = {
        "image": torch.randn(batch, 3, 224, 224, dtype=torch.float32),
        "text": torch.randint(low=0, high=49407, size=(text_batch, 77))
    }
    export_onnx(model, (dummy_input["image"], dummy_input["text"]), f"{args.output}/{args.model_id}.onnx")

def get_parser():
    parser = argparse.ArgumentParser(description='Convert OpenCLIP model to ONNX')
    parser.add_argument('--model_id', default='ViT-B-32', help='Model ID (e.g., ViT-B-32, ViT-L-14)')
    parser.add_argument('--checkpoint', help='Path to OpenCLIP checkpoint')
    parser.add_argument('--batch', type=int, default=1, help='Batch size')
    parser.add_argument('--separate', action='store_true', help='Export image and text encoders separately')
    parser.add_argument('--static_text', action='store_true', help='Use static input size for text encoder')
    parser.add_argument('--output', required=True, help='Output directory')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_parser()
    model, _, _ = open_clip.create_model_and_transforms(args.model_id, pretrained=args.checkpoint)
    text_batch = args.batch if args.static_text else -1
    
    if args.separate:
        convert_separate_model(args, model, args.batch, text_batch)
    else:
        convert_whole_model(args, model, args.batch, text_batch)
    
    print("Export complete!")

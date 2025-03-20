import argparse, os
import torch
import openvino as ov
from pytorch_pretrained_vit import ViT
import torch.onnx


def convert_model(args):
    try:
        # Load model
        model = ViT(args.checkpoint, pretrained=True)
        model.eval()

        # Determine image size based on model ID
        model_id = args.checkpoint
        img_size = 384 if model_id.endswith("imagenet1k") else 224

        img = torch.randn([args.batch_size,3,img_size,img_size])
        ov_model = ov.convert_model(model, example_input=img, input=(args.batch_size, 3, img_size, img_size))

        # Construct output path
        output_dir = args.output
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        file_extension = "_fp16.xml" if args.fp16 else "_fp32.xml"
        ov_path = os.path.join(output_dir, f"vit_{model_id.lower()}_bs{args.batch_size}{file_extension}")

        # Save the model
        ov.save_model(ov_model, ov_path, compress_to_fp16=args.fp16)
        print(f"Model saved to {ov_path}")

        # Export ONNX if enabled
        if args.onnx:
            onnx_path = os.path.join(output_dir, f"vit_{model_id.lower()}_bs{args.batch_size}.onnx")
            torch.onnx.export(
                model, img, onnx_path, 
                export_params=True,
                opset_version=13, 
                do_constant_folding=True,
                input_names=['input'], 
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            print(f"ONNX model saved to {onnx_path}")

            # Convert ONNX to FP16 if requested
            if args.fp16:
                import onnx
                from onnxconverter_common import float16

                onnx_model = onnx.load(onnx_path)
                onnx_fp16_model = float16.convert_float_to_float16(onnx_model)
                
                onnx_fp16_path = onnx_path.replace(".onnx", "_fp16.onnx")
                onnx.save(onnx_fp16_model, onnx_fp16_path)
                print(f"ONNX FP16 model saved to {onnx_fp16_path}")


    except Exception as e:
        print(f"Error converting {args.checkpoint} to OpenVINO IR: {e}")

def get_parser():
    parser = argparse.ArgumentParser(description='convert OpenCLIP model to OpenVINO IR')
    parser.add_argument('--checkpoint', dest="checkpoint",  help='the checkpoint, example: B_16, B_32, L_16, L_32, B_16_imagenet1k, B_32_imagenet1k, L_16_imagenet1k, L_32_imagenet1k')
    parser.add_argument('--output', dest='output', type=str, help='the path where to save the converted model')
    parser.add_argument('--batch_size', dest='batch_size', type=int, help='the input batch size')
    parser.add_argument('--fp16', dest='fp16', default=False, action='store_true', help='whether save model as fp16 mode')
    parser.add_argument('--onnx', action='store_true', help='Export model to ONNX')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()

    convert_model(args)

    print("ViT: Exiting ViT IR convert ......")
    print("Conversion completed.")

    exit(0)


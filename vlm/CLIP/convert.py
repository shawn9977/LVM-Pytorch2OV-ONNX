import argparse
import torch
import torch.nn.functional as F
import open_clip
import openvino as ov
import warnings
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

class TextTransformer(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, text):
        return self.model.encode_text(text)

def convert_seperate_model(args, model, batch, text_batch):
    # convert visual transformer
    print("CLIP: start to convert visual model")
    image_input = {"x": torch.randn(batch, 3, 224, 224, dtype=torch.float32)}
    openclip_image_encoder = ov.convert_model(
            model.visual, example_input=image_input, input=(batch, 3, 224, 224))
    
    compress_to_fp16=True
    output_path=f"{args.output}/{model_id.lower().replace('-','_')}_visual_fp16.xml"
    if not args.fp16:
       compress_to_fp16=False
       output_path=f"{args.output}/{model_id.lower().replace('-','_')}_visual_fp32.xml"

    print("CLIP: model is saved : ", output_path)
    ov.save_model(openclip_image_encoder, output_path, compress_to_fp16=compress_to_fp16)

    # convert text transformer
    print("CLIP: start to convert text model")
    t = TextTransformer(model)
    token_input = {"text": torch.randint(low=0, high=49407, size=(1, 77))}
    openclip_text_encoder = ov.convert_model(
        t, example_input=token_input, input=(text_batch, 77))

    compress_to_fp16=True
    output_path=f"{args.output}/{model_id.lower().replace('-','_')}_text_fp16.xml"
    if not args.fp16:
       compress_to_fp16=False
       output_path=f"{args.output}/{model_id.lower().replace('-','_')}_text_fp32.xml"
   
    print("CLIP: model is saved : ", output_path)
    ov.save_model(openclip_text_encoder, output_path, compress_to_fp16=compress_to_fp16)

def convert_whole_model(args, model, batch, text_batch):
    dummy_inputs = {
        "image": torch.randn(1, 3, 224, 224, dtype=torch.float32),
        "text": torch.randint(low=0, high=49407, size=(text_batch, 77)),
    }
    ov_model = ov.convert_model(
        model, example_input=dummy_inputs, input=([batch, 3, 224, 224], [10, 77]))

    compress_to_fp16=True
    output_path=f"{args.output}/{args.model_id.lower().replace('-','_')}_fp16.xml"
    if not args.fp16:
       compress_to_fp16=False
       output_path=f"{args.output}/{args.model_id.lower().replace('-','_')}_fp32.xml"

    print("CLIP: model is saved : ", output_path)
    ov.save_model(ov_model, output_path, compress_to_fp16=compress_to_fp16)

def get_parser():
    parser = argparse.ArgumentParser(description='convert OpenCLIP model to OpenVINO IR')

    parser.add_argument('--model_id', dest="model_id", default='ViT-B-32', help='the OpenCLIP model_id, example: ViT-B-32, ViT-L-14, ViT-H-14, ViT-g-14')
    parser.add_argument('--checkpoint', dest="checkpoint",  help='the  OpenCLIP checkpoint path')
    parser.add_argument('--batch', dest="batch", type=int, default=1, help='default batch size')
    parser.add_argument('--seperated', dest="seperated",  action='store_true',help='whether sperate the OpenCLIP model to image encoder and text encoder')
    parser.add_argument('--static_text', dest="static_text",default=True,  action='store_true',help='whether convert text encoder with static input size')
    parser.add_argument('--output', dest='output', type=str, help='the directory where to save the converted model')
    parser.add_argument('--fp16', dest='fp16', default=False, action='store_true', help='whether save model as fp16 mode')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    print(args)

    model_id = args.model_id
    if model_id not in ['ViT-B-32', 'ViT-L-14', 'ViT-H-14', 'ViT-g-14']:
        print(f"CLIP: {model_id} is not in the support list")
        exit(0)
    
    pretrained = args.checkpoint
    model, _, preprocess_image = open_clip.create_model_and_transforms(model_id, pretrained=pretrained)

    batch = args.batch
    text_batch = -1
    if args.static_text:
        text_batch = batch
    
    if args.seperated:
        convert_seperate_model(args, model, batch, text_batch)
    else:
        convert_whole_model(args, model, batch, text_batch)

    print("CLIP: Exiting CLIP IR convert ......")
    exit(0)

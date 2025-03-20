import os, argparse
import torch
import openvino as ov
from groundingdino.util.slconfig import SLConfig
from groundingdino.models import build_model
from groundingdino.util.utils import clean_state_dict
from groundingdino.util import get_tokenlizer


def load_pt_grounding_dino(model_config_path, model_checkpoint_path):
    args = SLConfig.fromfile(model_config_path)

    # modified config
    PT_DEVICE = "cpu"
    args.device = PT_DEVICE
    args.use_checkpoint = False
    args.use_transformer_ckpt = False

    model = build_model(args)
    # print(model)
    checkpoint = torch.load(model_checkpoint_path, map_location=PT_DEVICE)
    # print(checkpoint)
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    _ = model.eval()

    return (
        model,
        args.max_text_len,
        get_tokenlizer.get_tokenlizer(args.text_encoder_type),
    )

# The method from openvino:
#     https://github.com/wenyi5608/GroundingDINO/blob/wenyi5608-openvino/demo/export_openvino.py
# Output:
#     image input is not dynamic
# GroudingDINO repo:
#     https://github.com/wenyi5608/GroundingDINO/ wenyi5608-openvino branch 
def convert_model_static(args):
    original_file_name = os.path.basename(args.checkpoint)
    base_name = original_file_name[:19]
    file_extension = "_fp16.xml" if args.fp16 else "_fp32.xml"
    ir_file_name = f"{base_name}{file_extension}"
    ov_dino_path = os.path.join(args.output, ir_file_name)

    caption =  "the running dog ." #". ".join(input_text)
    input_ids =  pt_grounding_dino_model.tokenizer([caption], return_tensors="pt")["input_ids"]
    position_ids = torch.tensor([[0, 0, 1, 2, 3, 0]])
    token_type_ids = torch.tensor([[0, 0, 0, 0, 0, 0]])
    attention_mask = torch.tensor([[True, True, True, True, True, True]])
    text_token_mask = torch.tensor([[[ True, False, False, False, False, False],
         [False,  True,  True,  True,  True, False],
         [False,  True,  True,  True,  True, False],
         [False,  True,  True,  True,  True, False],
         [False,  True,  True,  True,  True, False],
         [False, False, False, False, False,  True]]])

    samples = torch.randn(1, 3, 512, 512)

    dynamic_axes={
       "input_ids": {0: "batch_size", 1: "seq_len"},
       "attention_mask": {0: "batch_size", 1: "seq_len"},
       "position_ids": {0: "batch_size", 1: "seq_len"},
       "token_type_ids": {0: "batch_size", 1: "seq_len"},
       "text_token_mask": {0: "batch_size", 1: "seq_len", 2: "seq_len"},
       "logits": {0: "batch_size"},
       "boxes": {0: "batch_size"}
    }

    torch.onnx.export(
        pt_grounding_dino_model,
        f="./groundingdino.onnx",
        args=(samples, input_ids, attention_mask, position_ids, token_type_ids, text_token_mask), #, zeros, ones),
        input_names=["samples" , "input_ids", "attention_mask", "position_ids", "token_type_ids", "text_token_mask"],
        output_names=["logits", "boxes"],
        dynamic_axes=dynamic_axes,
        opset_version=16)

    #convert_model returns an openvino.runtime.Model object
    ov_model = ov.convert_model("./groundingdino.onnx")

    print("GroundDINO: model is saved:", ov_dino_path)
    # then model can be serialized to *.xml & *.bin files

    ov.save_model(ov_model, ov_dino_path, compress_to_fp16=args.fp16)


# The method from openvino notebook:
#     https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/grounded-segment-anything/grounded-segment-anything.ipynb
# Output: 
#     All dynamic inputs
# GroudingDINO repo: 
#     https://github.com/wenyi5608/GroundingDINO/ main branch
def convert_model(args):
    ov_dino_path = args.output

    tokenized = pt_grounding_dino_model.tokenizer(["the running dog ."], return_tensors="pt")
    #{'input_ids': tensor([[ 101, 1996, 2770, 3899, 1012,  102]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1]])}
    #print(tokenized)
    
    input_ids = tokenized["input_ids"]
    token_type_ids = tokenized["token_type_ids"]
    attention_mask = tokenized["attention_mask"]
    position_ids = torch.arange(input_ids.shape[1]).reshape(1, -1)
    text_token_mask = torch.randint(0, 2, (1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.bool)
    img = torch.randn(1, 3, *ground_dino_img_size)

    dummpy_inputs = (
        img,
        input_ids,
        attention_mask,
        position_ids,
        token_type_ids,
        text_token_mask,
    )

    # without disabling gradients trace error occurs: "Cannot insert a Tensor that requires grad as a constant"
    for par in pt_grounding_dino_model.parameters():
        par.requires_grad = False
    # If we don't trace manually ov.convert_model will try to trace it automatically with default check_trace=True, which fails.
    # Therefore we trace manually with check_trace=False, despite there are warnings after tracing and conversion to OpenVINO IR
    # output boxes are correct.
    traced_model = torch.jit.trace(
        pt_grounding_dino_model,
        example_inputs=dummpy_inputs,
        strict=False,
        check_trace=False,
    )

    ov_dino_model = ov.convert_model(traced_model, example_input=dummpy_inputs)

    print("GroundDINO: model is saved:", ov_dino_path)
    if args.fp16 :
        ov.save_model(ov_dino_model, ov_dino_path, compress_to_fp16=True)
    else:
        ov.save_model(ov_dino_model, ov_dino_path, compress_to_fp16=False)


def get_parser():
    parser = argparse.ArgumentParser(description='convert Groudingdino model to OpenVINO IR')
    parser.add_argument('--checkpoint', dest="checkpoint",  help='the checkpoint path')
    parser.add_argument('--config', dest="config",  help='the config path of grounding DINO')
    parser.add_argument('--output', dest='output', type=str, help='the path where to save the converted model')
    parser.add_argument('--fp16', dest='fp16', default=False, action='store_true', help='whether save model as fp16 mode')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    print(args)

    pt_grounding_dino_model, max_text_len, dino_tokenizer = load_pt_grounding_dino(args.config, args.checkpoint)

    print("GroundDINO: start to convert the model")
    convert_model_static(args)

    print("GroundDINO: Exiting groudingdino IR convert ......")
    exit(0)

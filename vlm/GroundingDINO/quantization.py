from pathlib import Path
import os, argparse
import json
import torch
import nncf
from torchvision.transforms.functional import resize
import numpy as np
import openvino
from sklearn.metrics import accuracy_score
import openvino as ov
import torchvision
from torchvision.datasets import CocoDetection
from groundingdino.util.slconfig import SLConfig
from groundingdino.models import build_model
from groundingdino.util.utils import clean_state_dict
from groundingdino.util import get_tokenlizer
from groundingdino.models.GroundingDINO.bertwarper import (
    generate_masks_with_special_tokens_and_transfer_map,
)


def load_pt_grounding_dino(model_config_path, model_checkpoint_path):
    args = SLConfig.fromfile(model_config_path)

    # modified config
    PT_DEVICE = "cpu"
    args.device = PT_DEVICE
    args.use_checkpoint = False
    args.use_transformer_ckpt = False

    model = build_model(args)
    #print(model)
    checkpoint = torch.load(model_checkpoint_path, map_location=PT_DEVICE)
    #print(checkpoint)
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    _ = model.eval()

    return (
        model,
        args.max_text_len,
        get_tokenlizer.get_tokenlizer(args.text_encoder_type),
    )

def get_labels(annotations):
    labels = []

    size = len(annotations)
    for i in range(size):
        category_id = int(annotations[i]['category_id'])
        cc = len(jsondata['categories'])
        for c in range(cc):
            if int(jsondata['categories'][c]['id']) == category_id :
                labels.append(jsondata['categories'][c]['name'])

    return labels

def transform_fn(data_item):
    images, ann = data_item
    #print(ann)

    caption = get_labels(ann)
    if isinstance(caption, list):
        caption = ". ".join(caption)
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    captions = [caption]
    
    tokenized = dino_tokenizer(captions, padding="longest", return_tensors="pt")
    specical_tokens = dino_tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])
    
    (text_self_attention_masks, position_ids, cate_to_token_mask_list,) = generate_masks_with_special_tokens_and_transfer_map(tokenized, specical_tokens, dino_tokenizer)
    
    if text_self_attention_masks.shape[1] > max_text_len:
        text_self_attention_masks = text_self_attention_masks[:, :max_text_len, :max_text_len]

        position_ids = position_ids[:, :max_text_len]
        tokenized["input_ids"] = tokenized["input_ids"][:, :max_text_len]
        tokenized["attention_mask"] = tokenized["attention_mask"][:, :max_text_len]
        tokenized["token_type_ids"] = tokenized["token_type_ids"][:, :max_text_len]

    inputs = {}
    inputs["attention_mask"] = tokenized["attention_mask"]
    inputs["text_token_mask"] = text_self_attention_masks
    inputs["input_ids"] = tokenized["input_ids"]
    inputs["position_ids"] = position_ids
    inputs["token_type_ids"] = tokenized["token_type_ids"]
    inputs["samples"] = images    

    return inputs

def validate(model: openvino.runtime.CompiledModel,
             validation_loader: torch.utils.data.DataLoader) -> float:
    predictions = []
    references = []

    output = model.outputs[0]

    for images, target in validation_loader:
        pred = model(images)[output]
        predictions.append(np.argmax(pred, axis=1))
        references.append(target)

    predictions = np.concatenate(predictions, axis=0)
    references = np.concatenate(references, axis=0)
    return accuracy_score(predictions, references)

def get_ov_model(args):
    ov_dino_path = Path(args.ov_model)

    core = ov.Core()
    if not ov_dino_path.exists():
        print("GroundDINO: ov model doesn't exist.", ov_dino_path) 
    else:
        ov_dino_model = core.read_model(ov_dino_path)

    return ov_dino_model

def get_new_filename(original_path, suffix="_to_int8"):
    path_parts = original_path.rsplit('/', 1)
    filename = path_parts[1]
    filename_without_extension, extension = filename.rsplit('.', 1)
    return f"{filename_without_extension}{suffix}.{extension}"

def quantize_model(args):
    model = get_ov_model(args)
    new_filename = get_new_filename(args.ov_model)
    model_output_path = os.path.join(args.output, new_filename)

    if args.accuracy_control:
        print("GroundDINO: start to model quantization with out accuracy control......")
        quantized_model = nncf.quantize(
                        model,
                        calibration_dataset,
                        preset=nncf.QuantizationPreset.PERFORMANCE,
                        model_type=nncf.parameters.ModelType.TRANSFORMER,
                        subset_size=128,)
    else:
        print("GroundDINO: start to model quantization with accuracy control......")
        quantized_model = nncf.quantize_with_accuracy_control(
                        model,
                        calibration_dataset=calibration_dataset,
                        validation_dataset=validation_dataset,
                        validation_fn=validate,
                        preset=nncf.QuantizationPreset.PERFORMANCE,
                        model_type=nncf.parameters.ModelType.TRANSFORMER,
                        max_drop=0.01,)

    fp16_compress = args.fp16
    print("GroundDINO: save model with compress_to_fp16 : ", args.fp16)
    ov.save_model(quantized_model, model_output_path, compress_to_fp16=fp16_compress)
    print("GroundDINO: model quantization finished")

def get_parser():
    parser = argparse.ArgumentParser(description='convert OpenCLIP model to OpenVINO IR')
    parser.add_argument('--checkpoint', dest="checkpoint",  help='the  OpenCLIP checkpoint path')
    parser.add_argument('--config', dest="config",  help='the config path of grounding DINO')
    parser.add_argument('--ov_model', dest='ov_model', type=str, help='the model which need to be quantized, IR format')
    parser.add_argument('--output', dest='output', type=str, help='the path where to save thedd quantized model')
    parser.add_argument('--fp16', dest='fp16', default=False,  action='store_true', help='whether save model as fp16 mode')
    parser.add_argument('--data_dir', default='./data', help="Data folder to calibrate model, please select coco-2017, example: /mnt/storage/bruce-dir/datasets/coco-2017")
    parser.add_argument('--accuracy_control', dest='accuracy_control', default=True,  action='store_true', help='default mean without accuracy control')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    print(args)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((512, 512)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    ann_file = args.data_dir + "/validation/labels.json"
    with open(ann_file) as f:
       jsondata = json.load(f)

    data_dir = args.data_dir + "/validation/data"
    validation = CocoDetection(root=data_dir, annFile=ann_file, transform=transform)
    train = CocoDetection(root=data_dir, annFile=ann_file, transform=transform)

    calibration_loader = torch.utils.data.DataLoader(validation)
    validation_loader = torch.utils.data.DataLoader(validation)

    calibration_dataset = nncf.Dataset(calibration_loader, transform_fn)
    validation_dataset = nncf.Dataset(validation_loader, transform_fn)

    pt_grounding_dino_model, max_text_len, dino_tokenizer = load_pt_grounding_dino(args.config, args.checkpoint)

    quantize_model(args)

    print("GroundDINO: Exiting GroudingDINO quantization ......")
    exit(0)


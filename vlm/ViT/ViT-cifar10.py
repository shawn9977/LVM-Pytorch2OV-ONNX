from pathlib import Path
from zipfile import ZipFile
import cv2, os, argparse, subprocess
import torch
import torch.utils.data as data
import nncf
from copy import deepcopy
from typing import Tuple
from torchvision.transforms.functional import resize, to_pil_image
import numpy as np
import openvino
from sklearn.metrics import accuracy_score
import openvino as ov
import torchvision
from torchvision.datasets import CIFAR10
from pytorch_pretrained_vit import ViT


def transform_fn(data_item):
    images, _ = data_item
    return images

def predict_mapping(pred):
    pass    

def validate(model: openvino.runtime.CompiledModel,
             validation_loader: torch.utils.data.DataLoader) -> float:
    predictions = []
    references = []

    output = model.outputs[0]
    for images, target in validation_loader:
        pred = model(images)[output]

        #TODO mapping ViT label with CIFAR10 label        
        #predict = predict_mapping(pred)

        predictions.append(np.argmax(pred, axis=1))
        references.append(target)
        
    predictions = np.concatenate(predictions, axis=0)
    references = np.concatenate(references, axis=0)

    # print("predictions: ", predictions)
    # print("references: ", references)
    return accuracy_score(predictions, references)

def quantize_model(args):
    original_path = args.ov_model
    path_parts = original_path.rsplit('/', 1)
    filename = path_parts[1]
    filename_without_extension, extension = filename.rsplit('.', 1)
    new_filename = f"{filename_without_extension}_to_int8.{extension}"
    save_model_path = f"{args.output}/{new_filename}"

    core = ov.Core()

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((384, 384)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    validation = CIFAR10(root=os.path.expanduser(args.dataset), transform=transform, download=True, train=False)
    calibration = CIFAR10(root=os.path.expanduser(args.dataset), transform=transform, download=True, train=True)

    calibration_loader = torch.utils.data.DataLoader(calibration, batch_size=args.batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation, batch_size=args.batch_size, shuffle=True)

    calibration_dataset = nncf.Dataset(calibration_loader, transform_fn)
    validation_dataset = nncf.Dataset(validation_loader, transform_fn)

    ov_model_path = Path(args.ov_model)
    model = core.read_model(ov_model_path)

    if not args.accuracy_control:
        print("quantizing without accuracy control")
        quantized_model = nncf.quantize(
                    model=model,
                    calibration_dataset=calibration_dataset,
                    model_type=nncf.ModelType.TRANSFORMER,
                    preset=nncf.QuantizationPreset.PERFORMANCE,)
    else :
        print("quantizing with accuracy control")
        quantized_model = nncf.quantize_with_accuracy_control(model,
                    calibration_dataset=calibration_dataset,
                    validation_dataset=validation_dataset,
                    validation_fn=validate,
                    model_type=nncf.parameters.ModelType.TRANSFORMER,
                    preset=nncf.QuantizationPreset.PERFORMANCE,
                    max_drop=0.01)

    ov.save_model(quantized_model, save_model_path, compress_to_fp16=True)

def convert_model(args):
    # Load model
    model = ViT(args.checkpoint, pretrained=True)
    model.eval()

    # Convert and save OV model
    model_id = args.checkpoint
    img_size = 384 if model_id.endswith("imagenet1k") else 224
    
    img = torch.randn([args.batch_size,3,img_size,img_size])
    ov_model = ov.convert_model(model, example_input=img, input=(args.batch_size,3,img_size,img_size))
    
    path = args.output
    if args.fp16 : 
        ov_path = f"{path}/vit_{model_id.lower()}_bs{args.batch_size}_fp16.xml"
        ov.save_model(ov_model, ov_path, compress_to_fp16=True)
    else:
        ov_path = f"{path}/vit_{model_id.lower()}_bs{args.batch_size}_fp32.xml"
        ov.save_model(ov_model, ov_path, compress_to_fp16=False)

def get_parser():
    parser = argparse.ArgumentParser(description='convert OpenCLIP model to OpenVINO IR')

    subparsers = parser.add_subparsers(dest='command')
    parser_add = subparsers.add_parser('convert', help='convert model to IR')
    parser_add.add_argument('--checkpoint', dest="checkpoint",  help='the checkpoint, example: B_16, B_32, L_16, L_32, B_16_imagenet1k, B_32_imagenet1k, L_16_imagenet1k, L_32_imagenet1k')
    parser_add.add_argument('--output', dest='output', type=str, help='the path where to save the converted model')
    parser_add.add_argument('--batch_size', dest='batch_size', type=int, help='the input batch size')
    parser_add.add_argument('--fp16', dest='fp16', default=False, action='store_true', help='whether save model as fp16 mode')

    parser_add = subparsers.add_parser('quantize', help='nncf quantize model to INT8')
    parser_add.add_argument('--ov_model', dest='ov_model', type=str, help='the model which need to be quantized, IR format')
    parser_add.add_argument('--output', dest='output', type=str, help='the path where to save thedd quantized model')
    parser_add.add_argument('--batch_size', dest='batch_size', type=int, help='the input batch size')
    parser_add.add_argument('--data_dir', dest='dataset', default='./data', help="Data folder to calibrate model")
    parser_add.add_argument('--accuracy_control', dest='accuracy_control', default=False,  action='store_true', help='default mean without accuracy control')

    parser_add = subparsers.add_parser('reshape', help='reshape the model layer')
    parser_add.add_argument('--layer', dest="layer", default=str, help='choose the right layer')
    parser_add.add_argument('--new_shpae', dest='new_shape', type=str, help='the new shape you want to set')
    parser_add.add_argument('--ov_model', dest='ov_model', default=False, action='store_true', help='input the model')
    parser_add.add_argument('--output', dest='output', type=str, help="prepare the output path")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    print(args)

    if args.command == 'convert':
        convert_model(args)

    if args.command == 'quantize':
        quantize_model(args)

    if args.command == 'reshape':
        print("please waiting for the feature")
        #reshape_inputs(args)

    clean_system_buffer_and_cache(args.password)

    print("Done, Exiting......")
    exit(0)



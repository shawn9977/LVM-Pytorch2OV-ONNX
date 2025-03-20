from segment_anything import sam_model_registry
from segment_anything.utils.onnx import SamOnnxModel
import torch

# Load SAM model
sam = sam_model_registry["vit_b"](checkpoint="/home/shawn/project/application-genai-tools/models/SAM/sam_vit_b_01ec64.pth")

# Export images encoder from SAM model to ONNX
torch.onnx.export(
    f="./vit_b_encoder.onnx",
    model=sam.image_encoder,
    args=torch.randn(1, 3, 1024, 1024),
    input_names=["images"],
    output_names=["embeddings"],
    export_params=True,
    keep_initializers_as_inputs=True,
    external_data_format=False,
    opset_version=13
)

# Export mask decoder from SAM model to ONNX
onnx_model = SamOnnxModel(sam, return_single_mask=True)
embed_dim = sam.prompt_encoder.embed_dim
embed_size = sam.prompt_encoder.image_embedding_size
mask_input_size = [4 * x for x in embed_size]
dummy_inputs = {
    "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
    "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
    "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
    "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
    "has_mask_input": torch.tensor([1], dtype=torch.float),
    "orig_im_size": torch.tensor([1500, 2250], dtype=torch.int32),
}
output_names = ["masks", "iou_predictions", "low_res_masks"]
torch.onnx.export(
    f="./vit_b_decoder.onnx",
    model=onnx_model,
    args=tuple(dummy_inputs.values()),
    input_names=list(dummy_inputs.keys()),
    output_names=output_names,
    dynamic_axes={
        "point_coords": {1: "num_points"},
        "point_labels": {1: "num_points"}
    },
    export_params=True,
    opset_version=17,
    do_constant_folding=True
)


import onnxruntime
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic

out_onnx_model_quantized_path = "vit_b_16_encoder_quantized.onnx"
input_onnx_encoder_path = "vit_b_encoder.onnx" # Path to the encoder onnx file, all the other blocks files must be in the same dir!

quantize_dynamic(
    model_input=input_onnx_encoder_path,
    model_output=out_onnx_model_quantized_path,
    # optimize_model=True,
    per_channel=False,
    reduce_range=False,
    weight_type=QuantType.QUInt8,
)


import onnx
import onnx_graphsurgeon as gs

# 载入 ONNX 模型
model = onnx.load("vit_b_16_encoder_quantized.onnx")
graph = gs.import_onnx(model)

# 替换 ConvInteger -> Conv
for node in graph.nodes:
    if node.op == "ConvInteger":
        node.op = "Conv"

# 保存修改后的 ONNX
onnx.save(gs.export_onnx(graph), "vit_b_16_encoder_fixed.onnx")

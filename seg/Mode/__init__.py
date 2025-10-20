from seg.Mode.model import *
from seg.Mode.unet import UNet
from seg.Mode.segnet import SegNet
from seg.Mode.attention_unet import AttentionUNet
from seg.Mode.DeepLabV3 import DeepLabV3
from seg.Mode.nnunet import nnUNet2D
from seg.Mode.TransUnet import TransUNet
from seg.Mode.CE_Net import CE_Net

from seg.Mode.EfficientUNet_Lite import EfficientUNet_Lite
from seg.Mode.UNet_Lite import UNetPlusPlus_Lite
MODELS = {
    "EA_Unet": EA_Unet,
    "unet": UNet,
    "segnet": SegNet,
    "attention_unet": AttentionUNet,
    "DeepLabV3": DeepLabV3,
    "nnunet": nnUNet2D,
    "TransUNet": TransUNet,
    "CE_Net": CE_Net,
    "UNetPlusPlus_Lite": UNetPlusPlus_Lite,
    "EfficientUNet_Lite": EfficientUNet_Lite
}

def get_model(model_name, in_channels=4, out_channels=4):
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")
    return MODELS[model_name](in_channels, out_channels)
# config.py
import torch
# 导入所有模型
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
# 配置参数
DATA_DIR = r'D:\data\output'
MODALITIES = ['FLAIR', 'T1', 'T1GD', 'T2']
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_CLASSES = 4  # 背景 + 3个类别
CLASS_WEIGHTS = torch.tensor([0.1, 1.0, 1.0, 1.0])  # 背景、坏死、水肿、肿瘤


# 模型映射字典
MODEL_MAP = {
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

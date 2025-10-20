# config.py
import torch

from seg.Mode.model import *

# 配置参数
DATA_DIR = 'dir'
MODALITIES = ['FLAIR', 'T1', 'T1GD', 'T2']
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_CLASSES = 4
CLASS_WEIGHTS = torch.tensor([0.1, 1.0, 1.0, 1.0])


MODEL_MAP = {
    "EA_Unet": EA_Unet,
}

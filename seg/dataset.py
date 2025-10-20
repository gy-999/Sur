# dataset.py

import os
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from torchvision import transforms
from seg.config import MODALITIES, DATA_DIR


class BrainTumorDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        base_name = self.file_list[idx]

        # 加载多模态图像
        images = []
        for mod in MODALITIES:
            img_path = os.path.join(DATA_DIR, mod, base_name)
            img = nib.load(img_path).get_fdata().astype(np.float32)

            img = (img - img.mean()) / (img.std() + 1e-8)
            images.append(img)


        # 创建字典样本
        image_stack = np.stack(images, axis=0)  # (4, H, W)
        mask_path = os.path.join(DATA_DIR, 'mask', base_name)
        mask = nib.load(mask_path).get_fdata().astype(np.uint8)

        # 标签映射
        remapped_mask = np.zeros_like(mask)
        remapped_mask[mask == 1] = 1
        remapped_mask[mask == 2] = 2
        remapped_mask[mask == 4] = 3

        sample = {'image': image_stack, 'label': remapped_mask}

        # 应用变换
        if self.transform:
            sample = self.transform(sample)

        # 返回图像和标签张量
        return sample['image'], sample['label'], base_name


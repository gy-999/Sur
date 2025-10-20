# utils.py
import math
import numpy as np
from sklearn.model_selection import train_test_split
import os
from torch.utils.data import DataLoader
from seg.dataset import *
from seg.config import *
from torchvision import transforms
from seg.preprocess import *
import argparse
import os
import pandas as pd
import json
from datetime import datetime
from seg.config import *



# 获取指定模型
def get_model(model_name, in_channels=4, out_channels=4):
    if model_name not in MODEL_MAP:
        raise ValueError(f"未知模型: {model_name}. 可用模型: {list(MODEL_MAP.keys())}")
    return MODEL_MAP[model_name](in_channels, out_channels)


def load_model_weights(model, checkpoint_path):
    """安全加载模型权重，处理不匹配的键"""
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # 过滤掉不需要的键
    filtered_state_dict = {}
    for key, value in state_dict.items():
        if 'total_ops' not in key and 'total_params' not in key:
            filtered_state_dict[key] = value

    # 获取当前模型的状态字典
    model_state_dict = model.state_dict()

    # 检查键的匹配情况
    missing_keys = []
    unexpected_keys = []

    for key in filtered_state_dict:
        if key not in model_state_dict:
            unexpected_keys.append(key)

    for key in model_state_dict:
        if key not in filtered_state_dict:
            missing_keys.append(key)

    if missing_keys:
        print(f"警告: 缺少以下键: {missing_keys}")
    if unexpected_keys:
        print(f"警告: 意外的键: {unexpected_keys}")

    # 加载权重
    model.load_state_dict(filtered_state_dict, strict=False)
    return checkpoint


def save_results_to_csv(results, csv_file='result/model_results.csv'):
    """将简化结果保存到CSV文件"""
    try:
        # 准备数据 - 只包含平均值
        data = {
            'model': results['model'],
            'params': results['params'],
            'gflops': results['gflops'],
            'test_loss': results['test_loss'],
            'avg_dice': results['avg_dice'],
            'avg_jaccard': results['avg_jaccard'],
            'dice_necrosis': results.get('dice_necrosis', 0),
            'dice_edema': results.get('dice_edema', 0),
            'dice_enhancing': results.get('dice_enhancing', 0),
            'timestamp': results.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        }

        # 如果文件存在，读取并追加
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            # 检查是否已有相同模型的记录
            mask = df['model'] == results['model']
            if mask.any():
                # 更新现有记录
                for key, value in data.items():
                    df.loc[mask, key] = value
            else:
                # 添加新记录
                df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
        else:
            # 创建新文件
            df = pd.DataFrame([data])

        # 保存到CSV
        df.to_csv(csv_file, index=False)
        print(f"简化结果已保存到: {csv_file}")

    except Exception as e:
        print(f"保存结果到CSV时出错: {e}")


def load_previous_results(csv_file='model_results.csv'):
    """加载之前的结果"""
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            return df
        except Exception as e:
            print(f"加载历史结果时出错: {e}")
    return None


def save_detailed_results(results, output_dir='results'):
    """保存详细结果到JSON文件"""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{results['model']}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    try:
        # 转换numpy数组为列表以便JSON序列化
        serializable_results = {}
        for key, value in results.items():
            if hasattr(value, 'tolist'):
                serializable_results[key] = value.tolist()
            else:
                serializable_results[key] = value

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        print(f"详细结果已保存到: {filepath}")
    except Exception as e:
        print(f"保存详细结果时出错: {e}")
def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    scheduler = []
    total_steps = epochs * niter_per_ep
    warmup_steps = warmup_epochs * niter_per_ep

    for step in range(total_steps):
        if step < warmup_steps:
            lr = start_warmup_value + (base_value - start_warmup_value) * step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            lr = final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * progress))
        scheduler.append(lr)

    return np.array(scheduler)


# 更新预处理流程
def preprocess_data():

    mask_dir = os.path.join(DATA_DIR, 'mask')
    all_files = [f for f in os.listdir(mask_dir) if f.endswith('.nii')]

    train_files, test_files = train_test_split(all_files, test_size=0.3, random_state=42)
    val_files, test_files = train_test_split(test_files, test_size=0.5, random_state=42)

    # 定义2D预处理流程
    patch_size = (180, 180)
    train_transform = transforms.Compose([
        RandomRotFlip2D(),
        RandomCrop2D(patch_size),
        GaussianNoise(p=0.1),
        ContrastAugmentationTransform(p_per_sample=0.5),
        BrightnessTransform(mu=0.0, sigma=0.1, p_per_sample=0.5),
        EnsureContiguous(),  # 添加连续性确保
        ToTensorDict()
    ])

    val_transform = transforms.Compose([
        CenterCrop2D(patch_size),
        EnsureContiguous(),  # 添加连续性确保
        ToTensorDict()
    ])

    test_transform = transforms.Compose([
        CenterCrop2D(patch_size),
        EnsureContiguous(),  # 添加连续性确保
        ToTensorDict()
    ])

    train_dataset = BrainTumorDataset(train_files, transform=train_transform)
    val_dataset = BrainTumorDataset(val_files, transform=val_transform)
    test_dataset = BrainTumorDataset(test_files, transform=test_transform)


    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
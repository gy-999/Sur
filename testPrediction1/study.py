# main.py
import sys
import torch

torch.cuda.empty_cache()

sys.path.append(r'D:\python\cancer_survival\testPrediction1\model')
import utils
from lifelines.utils import concordance_index
from testPrediction1.model.data_loader import MyDataset, preprocess_clinical_data
from testPrediction1.model.model import (
    MultimodalModel,
    DeepSurvModel
)
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import os

# 设置必要的参数
m_length = 16
BATCH_SIZE = 16
EPOCH = 100
lr = 0.003
data_path = utils.DATA_PATH

# 设置随机种子并检测CUDA
utils.setup_seed(24)
device = utils.test_gpu()

# 只测试两个模型来简化问题'only_cox''Edema', 'Necrosis''Tumor'
model_configs = [
    {
        'name': 'Multimodal Model',
        'modalities': ['clinical', 'Edema', 'Necrosis', 'Tumor'],
        'model_class': MultimodalModel,
        'kwargs': {
            'fusion_method': 'attention',
            'trade_off': 0.3,
            'mode': 'total'
        }
    },
    {
        'name': 'DeepSurv',
        'modalities': ['clinical'],
        'model_class': DeepSurvModel,
        'kwargs': {
            'hidden_dims': [64, 32],  # 减小网络规模
            'dropout_rate': 0.3
        }
    }
]

# 存储所有模型的结果
all_results = {}

# 创建数据集, 'Edema', 'Tumor', 'Necrosis'
mydataset = MyDataset(['clinical','Edema','Necrosis','Tumor'], data_path)

# 预处理数据
prepro_clin_data_X, _, prepro_clin_data_y, _ = preprocess_clinical_data(data_path['clinical'])
prepro_clin_data_X.reset_index(drop=True, inplace=True)
prepro_clin_data_y.reset_index(drop=True, inplace=True)

# 定义分割比例
train_size = 0.6
val_size = 0.3
test_size = 0.1

# 使用分层抽样确保事件分布
x_train_val, x_test, y_train_val, y_test = train_test_split(
    prepro_clin_data_X,
    prepro_clin_data_y[11],
    test_size=test_size,
    random_state=24,
    stratify=prepro_clin_data_y[11]
)
x_train, x_val, y_train, y_val = train_test_split(
    x_train_val,
    y_train_val,
    test_size=val_size / (train_size + val_size),
    random_state=24,
    stratify=y_train_val
)

# 获取索引
train_indices = x_train.index.tolist()
val_indices = x_val.index.tolist()
test_indices = x_test.index.tolist()

print(f"\n{'#' * 80}")
print("开始模型对比实验")
print(f"{'#' * 80}")

# 训练和评估每个模型
for config in model_configs:
    print(f"\n{'=' * 60}")
    print(f"训练模型: {config['name']}")
    print(f"使用模态: {config['modalities']}")
    print(f"{'=' * 60}")

    try:
        # 为每个模型创建数据加载器
        dataloaders = utils.get_dataloaders(mydataset, train_indices, val_indices, test_indices, BATCH_SIZE)

        # 调试：检查数据维度
        print("检查数据维度...")
        for phase in ['train']:
            for i, (data, data_label) in enumerate(dataloaders[phase]):
                if i == 0:  # 只检查第一个batch
                    print(f"{phase} 数据形状:")
                    for modality, tensor in data.items():
                        print(f"  {modality}: {tensor.shape}")
                    print(f"  label: {data_label['label'].shape}")
                    break

        # 创建模型实例
        if config['model_class'] == MultimodalModel:
            model = config['model_class'](
                modalities=config['modalities'],
                m_length=m_length,
                dataloaders=dataloaders,
                device=device,
                **config['kwargs']
            )
        else:
            # 单模态模型只需要临床数据
            model = config['model_class'](
                m_length=m_length,
                dataloaders=dataloaders,
                device=device,
                **config['kwargs']
            )

        # 设置训练参数
        run_tag = utils.compose_run_tag(
            model=model, lr=lr, dataloaders=dataloaders,
            log_dir='.training_logs/', suffix=f"_{config['name'].replace(' ', '_')}"
        )

        fit_args = {
            'num_epochs': EPOCH,
            'lr': lr,
            'info_freq': 3,
            'log_dir': os.path.join('.training_logs/', run_tag),
            'lr_factor': 0.01,
            'scheduler_patience': 5,
        }

        # 训练模型
        model.fit(**fit_args)

        # 输出最佳验证性能
        if hasattr(model, 'best_c_index') and 'best_score' in model.best_c_index:
            best_val_c_index = model.best_c_index['best_score']
            print(f"{config['name']} 最佳验证 C-index: {best_val_c_index:.4f}")
        else:
            best_val_c_index = 0.0
            print(f"{config['name']} 无法获取验证 C-index")

        # 测试模型
        print(f"测试 {config['name']}...")
        model.test()  # 加载最佳权重

        # 收集所有测试集的预测结果
        all_hazards = []
        all_events = []
        all_times = []

        model.model.eval()
        with torch.no_grad():
            for data_test, data_label_test in dataloaders['test']:
                out_test, event_test, time_test = model.predict(data_test, data_label_test)
                hazard_test, representation_test = out_test

                all_hazards.extend(hazard_test['hazard'].detach().cpu().numpy())
                all_events.extend(event_test.cpu().numpy())
                all_times.extend(time_test.cpu().numpy())

        # 在整个测试集上计算C-index
        if len(all_hazards) > 0:
            test_c_index = concordance_index(
                event_times=np.array(all_times),
                predicted_scores=-np.array(all_hazards),
                event_observed=np.array(all_events)
            )
            print(f"{config['name']} 测试集 C-index: {test_c_index:.4f}")

            # 存储结果
            all_results[config['name']] = {
                'test_c_index': test_c_index,
                'val_c_index': best_val_c_index,
                'modalities': config['modalities']
            }
        else:
            print(f"警告：{config['name']} 测试集为空")
            all_results[config['name']] = {
                'test_c_index': 0.0,
                'val_c_index': best_val_c_index,
                'modalities': config['modalities']
            }

    except Exception as e:
        print(f"训练 {config['name']} 时出现错误: {e}")
        import traceback

        traceback.print_exc()
        all_results[config['name']] = {
            'test_c_index': 0.0,
            'val_c_index': 0.0,
            'modalities': config['modalities'],
            'error': str(e)
        }

# 打印比较结果
print(f"\n{'#' * 80}")
print("模型性能比较结果")
print(f"{'#' * 80}")

print(f"\n{'模型名称':<20} {'使用模态':<30} {'验证集C-index':<15} {'测试集C-index':<15}")
print('-' * 80)

# 按测试集C-index排序
sorted_results = sorted(all_results.items(), key=lambda x: x[1].get('test_c_index', 0), reverse=True)

for model_name, results in sorted_results:
    modalities_str = ', '.join(results['modalities'])
    val_c_index = results.get('val_c_index', 0.0)
    test_c_index = results.get('test_c_index', 0.0)

    print(f"{model_name:<20} {modalities_str:<30} {val_c_index:<15.4f} {test_c_index:<15.4f}")

print(f"\n{'#' * 80}")
print("所有模型训练完成")
print(f"{'#' * 80}")
"""Utils"""
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader


DATA_PATH = {
'clinical': 'clinical1.csv',
'Edema': 'ED2',
'Necrosis': 'NET2',
'Tumor': '\ET2'
}
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def test_gpu():
    print('GPU？', torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('The device is ', device)
    return device

def evaluate_model(c_index_arr):
    m = np.sum(c_index_arr, axis=0) / len(c_index_arr)
    s = np.std(c_index_arr)
    return m, s

def get_dataloaders(mydataset, train_indices, val_indices, test_indices, batch_size):
    # 创建采样器
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    dataloaders = {}
    dataloaders['train'] = DataLoader(mydataset, batch_size=batch_size, sampler=train_sampler)
    dataloaders['val'] = DataLoader(mydataset, batch_size=batch_size, sampler=val_sampler)
    dataloaders['test'] = DataLoader(mydataset, batch_size=batch_size, sampler=test_sampler)

    print('Dataset sizes (# patients):')
    print('train: ', len(train_indices))
    print('  val: ', len(val_indices))
    print(' test: ', len(test_indices))
    print()
    print('Batch size: ', batch_size)

    return dataloaders

def compose_run_tag(model, lr, dataloaders, log_dir, suffix=''):
    def add_string(string, addition, sep='_'):
        if not string:
            return addition
        else:
            return string + sep + addition

    data = None

    if hasattr(model, 'data_modalities'):
        for modality in model.data_modalities:
            data = add_string(data, modality)
    else:

        data = 'clinical'

    run_tag = f'{data}_lr{lr}'
    run_tag += suffix
    print(f'Run tag: "{run_tag}"')
    tb_log_dir = os.path.join(log_dir, run_tag)
    return run_tag

def save_5fold_results(c_index_arr_test, c_index_arr_combine, run_tag):
	"""
	Save the results after 5 fold cross validation.
	"""
	m, s = evaluate_model(c_index_arr_test)
	p, q = evaluate_model(c_index_arr_combine)
	with open(f'proposed_{run_tag}.txt', 'w') as file:
		file.write(f'\ntest_c_index: {str(c_index_arr_test)}')
		file.write(f"\n test_Mean: {m}")
		file.write(f"\n test_Std: {s}")
		file.write(f'\ncombine_c_index: {str(c_index_arr_combine)}')
		file.write(f"\n combine_Mean: {p}")
		file.write(f"\n combine_Std: {q}")
	file.close()


def save_5fold_results1(c_index_arr_test, all_fold_average_brier_scores, run_tag):
	"""
	Save the results after 5 fold cross validation.
	"""
	a=np.mean(all_fold_average_brier_scores)
	b=np.std(all_fold_average_brier_scores)
	m, s = evaluate_model(c_index_arr_test)
	with open(f'No cross-A/proposed_{run_tag}.txt', 'w') as file:
		file.write(f'\ntest_c_index: {str(c_index_arr_test)}')
		file.write(f'\nAverage Brier Score: {a}')
		file.write(f'\n Brier Score_Std: {b}')
		file.write(f"\n test_Mean: {m}")
		file.write(f"\n test_Std: {s}")
	file.close()























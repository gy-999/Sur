"""data_loader.py"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
import os
def load_nifti_image(file_path):
    """Load NIfTI image from file and return as numpy array."""
    img = nib.load(file_path)
    img_data = img.get_fdata()
    return img_data
def find_out_index(data):
    out_idx = []
    for i, row in enumerate(data.values.tolist()):
        try:
            np.array(row, dtype='int')
        except ValueError:
            out_idx.append(i)
    return out_idx
def preprocess_clinical_data(clinical_path):
    data_clinical = pd.read_csv(clinical_path, header=None, delimiter='\t')
    target_data = data_clinical.iloc[:, [11, 12]]
    out_idx = find_out_index(target_data)
    clin_variables = data_clinical.iloc[:, 1:12]
    idx_minus_one = clin_variables[(clin_variables.iloc[:, 1] == -1) | (clin_variables.iloc[:, 2] == -1)].index.tolist()
    delete_indices = list(set(out_idx + idx_minus_one))
    data_clinical.drop(index=delete_indices, inplace=True)
    target_data = data_clinical.iloc[:, [11, 12]]
    clin_data_categorical = data_clinical.iloc[:, [1, 3, 4, 5, 7, 8, 9, 10]]
    # print(clin_data_categorical)
    clin_data_continuous = data_clinical.iloc[:, [2,6]]
    return clin_data_categorical, clin_data_continuous, target_data, delete_indices
class MyDataset(Dataset):
    def __init__(self, modalities, data_path):
        super(MyDataset, self).__init__()
        self.data_modalities = modalities
        clin_data_categorical, clin_data_continuous, target_data, remove_idx = preprocess_clinical_data(data_path['clinical'])
        # print('data', clin_data_categorical)
        # print('data', clin_data_continuous)
        # print('data', target_data)
        self.target = target_data.values.tolist()
        if 'clinical' in self.data_modalities:
            self.clin_cat = clin_data_categorical.values.tolist()
            self.clin_cont = clin_data_continuous.values.tolist()
        if 'Edema' in self.data_modalities:
            edema_files = [f for f in sorted(os.listdir(data_path['Edema'])) if f.endswith('.nii')]
            self.edema_files = [edema_files[i] for i in range(len(edema_files)) if i not in remove_idx]
        if 'Necrosis' in self.data_modalities:
            necrosis_files = [f for f in sorted(os.listdir(data_path['Necrosis'])) if f.endswith('.nii')]
            self.necrosis_files = [necrosis_files[i] for i in range(len(necrosis_files)) if i not in remove_idx]
        if 'Tumor' in self.data_modalities:
            tumor_files = [f for f in sorted(os.listdir(data_path['Tumor'])) if f.endswith('.nii')]
            self.tumor_files = [tumor_files[i] for i in range(len(tumor_files)) if i not in remove_idx]
        self.data_path = data_path
    def __len__(self):
        return len(self.target)
    def __getitem__(self, index):
        data = {}
        data_label = {}
        target_y = np.array(self.target[index], dtype='int')
        target_y = torch.from_numpy(target_y)
        data_label['label'] = target_y.type(torch.LongTensor)
        if 'clinical' in self.data_modalities:
            clin_cate = np.array(self.clin_cat[index], dtype=np.int64)
            clin_cate = torch.from_numpy(clin_cate)
            data['clinical_categorical'] = clin_cate
            clin_conti = np.array(self.clin_cont[index], dtype=np.float32)
            clin_conti = torch.from_numpy(clin_conti)
            data['clinical_continuous'] = clin_conti
        if 'Edema' in self.data_modalities:
            edema_path = os.path.join(self.data_path['Edema'], self.edema_files[index])
            edema_img = load_nifti_image(edema_path)
            edema_tensor = torch.from_numpy(edema_img).float()
            data['Edema'] = edema_tensor
        if 'Necrosis' in self.data_modalities:
            necrosis_path = os.path.join(self.data_path['Necrosis'], self.necrosis_files[index])
            necrosis_img = load_nifti_image(necrosis_path)
            necrosis_tensor = torch.from_numpy(necrosis_img).float()
            data['Necrosis'] = necrosis_tensor
        if 'Tumor' in self.data_modalities:
            tumor_path = os.path.join(self.data_path['Tumor'], self.tumor_files[index])
            tumor_img = load_nifti_image(tumor_path)
            tumor_tensor = torch.from_numpy(tumor_img).float()
            data['Tumor'] = tumor_tensor
        return data, data_label






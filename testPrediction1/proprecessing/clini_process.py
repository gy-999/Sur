import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

# 第一步：读取临床数据
def read_clinical_data(path):
    return pd.read_csv(path)

CLINICAL_PATH = r'D:\UPENN\tumor_segmentation\UPENN-GBM_clinical_info_v2.1.csv'
data = read_clinical_data(CLINICAL_PATH)

# 第二步：处理数据，找到并删除不需要的行和文件
def find_rows_to_delete(data, column, value):
    return data[data[column] == value]

def delete_files(ids_to_delete, directories):
    for directory in directories:
        for id_name in ids_to_delete:
            file_path = os.path.join(directory, f"{id_name}.nii")
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted file: {file_path}")

rows_to_delete = find_rows_to_delete(data, 'GTR_over90percent', 'Not Available')
ids_to_delete = rows_to_delete.iloc[:, 0]
directories = [
    r'D:\UPENN\tumor_segmentation\Edema1',
    r'D:\UPENN\tumor_segmentation\Necrosis1',
    r'D:\UPENN\tumor_segmentation\Tumor1'
]
delete_files(ids_to_delete, directories)

# 从临床数据中删除这些行
clinical_df = data[data['GTR_over90percent'] != 'Not Available']

# 第三步：对'event'列进行映射
def map_event_column(df, mapping):
    df['event'] = df['event'].map(mapping)
    return df

co_deletion_mapping = {'Deceased': 1, 'Alive': 0}
clinical_df = map_event_column(clinical_df, co_deletion_mapping)

# 第四步：选择需要使用的特征并处理缺失值
def select_and_process_features(df, features):
    data_used = df[features]
    data_used.columns = ['ID', 'Gender', 'Age_at_scan_years', 'IDH1', 'GTR_over90percent', 'event', 'event_time']
    data_used.dropna(subset=['event', 'event_time'], inplace=True)
    return data_used

features = ['ID', 'Gender', 'Age_at_scan_years', 'IDH1', 'GTR_over90percent', 'event', 'event_time']
data_used = select_and_process_features(clinical_df, features)

# 第五步：分类数据和连续数据的处理
def process_categorical_and_numerical_data(df):
    df_cate = df[['Gender', 'IDH1', 'GTR_over90percent']].apply(LabelEncoder().fit_transform)
    df_cate = df_cate.astype('category')
    df_num = df[['Age_at_scan_years']].fillna(-1)
    return df[['ID']], df_cate, df_num, df[['event', 'event_time']]

data_id, data_cate, data_num, target = process_categorical_and_numerical_data(data_used)

# 第六步：定义嵌入大小并合并数据
def define_embedding_sizes(data_cate):
    return [(len(col.cat.categories), min(50, (len(col.cat.categories) + 1) // 2)) for _, col in data_cate.items()]

embedding_sizes = define_embedding_sizes(data_cate)
print(embedding_sizes)

def merge_data(data_id, data_cate, data_num, target):
    return pd.concat([data_id, data_cate, data_num, target], axis=1)

df = merge_data(data_id, data_cate, data_num, target)

print('Sample size: ', len(set(df['ID'])))
print(set(len(i) for i in df['ID']))

# 第七步：获取图像文件名称并过滤临床数据
def get_file_names(directory):
    return [os.path.splitext(f)[0] for f in os.listdir(directory) if f.endswith('.nii')]

def filter_images_by_clinical_data(clinical_ids, directories):
    for directory in directories:
        for file_name in os.listdir(directory):
            if file_name.endswith('.nii') and os.path.splitext(file_name)[0] not in clinical_ids:
                file_path = os.path.join(directory, file_name)
                os.remove(file_path)
                print(f"Deleted file: {file_path}")

directories = [
    r'D:\UPENN\tumor_segmentation\Edema1',
    r'D:\UPENN\tumor_segmentation\Necrosis1',
    r'D:\UPENN\tumor_segmentation\Tumor1'
]

clinical_patient_ids = df['ID'].tolist()
filter_images_by_clinical_data(clinical_patient_ids, directories)

# 获取图像文件名称
# structural_files = get_file_names(r"D:\UPENN\images_structural")
edema_files = get_file_names(r"D:\UPENN\tumor_segmentation\Edema1")

file_data = {
    # "images_structural": structural_files,
    "tumor_segmentation_Edema": edema_files
}

file_df = pd.DataFrame({k: pd.Series(v) for k, v in file_data.items()})
first_patient_names = file_df["tumor_segmentation_Edema"].dropna().tolist()

# 过滤临床数据，只保留患者名称在first_patient_names中的行
filtered_second_df = df[df['ID'].isin(first_patient_names)]

# 第八步：将过滤后的结果保存为新的 CSV 文件，不包含第一行 index=False, header=False, sep='\t' header=False, sep='\t'
filtered_second_df.to_csv(r'D:\UPENN\tumor_segmentation\clinical.csv', index=False,)

print("Filtered file has been saved as clinical3.csv without header")
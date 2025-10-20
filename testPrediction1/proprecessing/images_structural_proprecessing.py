import os
import shutil
import os
import nibabel as nib
import numpy as np
import csv
import os
import shutil
import numpy as np
import nibabel as nib
import csv

def find_zero_files(directory):
    zero_files = []

    # 遍历目录中的所有文件
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.nii'):
                file_path = os.path.join(root, file)

                # 读取NIfTI文件
                img = nib.load(file_path)
                data = img.get_fdata()

                # 检查文件数据是否全为零
                if np.all(data == 0):
                    zero_files.append(file)

    return zero_files

def save_to_csv(file_list, csv_path):
    # 将文件名称存储在CSV文件的第一列
    with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for file_name in file_list:
            writer.writerow([file_name])

def move_files(file_list, source_dir, target_dir):
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file in file_list:
                source_path = os.path.join(root, file)
                target_path = os.path.join(target_dir, file)
                shutil.move(source_path, target_path)
                print(f"已移动文件: {source_path} 到 {target_path}")

# 目录路径
source_directories = {
    'D:\\xunlei\\PKG - UCSF-PDGM-v3-20230111\\Feature_image\\Edema2': 'D:\\xunlei\\PKG - UCSF-PDGM-v3-20230111\\Feature_image\\Edema3',
    'D:\\xunlei\\PKG - UCSF-PDGM-v3-20230111\\Feature_image\\Necrosis2': 'D:\\xunlei\\PKG - UCSF-PDGM-v3-20230111\\Feature_image\\Necrosis3',
    'D:\\xunlei\\PKG - UCSF-PDGM-v3-20230111\\Feature_image\\Tumor2': 'D:\\xunlei\\PKG - UCSF-PDGM-v3-20230111\\Feature_image\\Tumor3'
}

all_zero_files = []

# 查找每个目录中的全为零文件
for source_dir in source_directories.keys():
    zero_files = find_zero_files(source_dir)
    all_zero_files.extend(zero_files)

# 去重
all_zero_files = list(set(all_zero_files))

# CSV文件路径
csv_path = r'D:\xunlei\PKG - UCSF-PDGM-v3-20230111\_files.csv'

# 保存到CSV文件
if all_zero_files:
    save_to_csv(all_zero_files, csv_path)
    print(f"全为零的文件已存储在 {csv_path}")

    # 移动全为零的文件
    for source_dir, target_dir in source_directories.items():
        move_files(all_zero_files, source_dir, target_dir)
else:
    print("没有全为零的文件。")
# def remove_folders_with_suffix(directory, suffix):
#     try:
#         for folder_name in os.listdir(directory):
#             folder_path = os.path.join(directory, folder_name)
#             if os.path.isdir(folder_path) and folder_name.endswith(suffix):
#                 shutil.rmtree(folder_path)
#                 print(f'Removed folder: {folder_path}')
#     except Exception as e:
#         print(f'Error: {e}')
#
# # 目录路径和后缀
# directory = r'D:\UPENN\images_structural'
# suffix = '_21'
#
# # 调用函数
# remove_folders_with_suffix(directory, suffix)






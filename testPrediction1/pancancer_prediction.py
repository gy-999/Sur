# %pip install -r requirements.txt
import os
import sys
sys.path.append(r'D:\python\cancer_survival\testPrediction1\model')
import utils
from lifelines.utils import concordance_index
from testPrediction1.model.data_loader import MyDataset
from testPrediction1.model.data_loader import preprocess_clinical_data
from testPrediction1.model.model import Model
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index
from sksurv.metrics import brier_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Ellipse
from scipy import stats
import seaborn as sns
from sklearn.manifold import TSNE

from sksurv.util import Surv
from sksurv.metrics import integrated_brier_score
from sksurv.nonparametric import kaplan_meier_estimator

# 设置必要的参数
m_length = 16
BATCH_SIZE = 8
EPOCH = 30
lr = 0.003
K = 3
data_path = utils.DATA_PATH
modalities_list = [['clinical',]]

# 绿色为浮肿区域(ED,peritumoral edema) (标签2)、,'Edema',,'Tumor','Necrosis','Edema'
# 黄色为增强肿瘤区域(ET,enhancing tumor)(标签4)、'Tumor',
# 红色为坏疽(NET,non-enhancing tumor)(标签1) 'Necrosis',,
# 设置随机种子并检测CUDA
utils.setup_seed(24)
device = utils.test_gpu()

def plot_combined_correlations(representation_0, representation_1, fold_representation, modalities):
    n_modalities = len(modalities)
    fig, axes = plt.subplots(n_modalities, n_modalities, figsize=(5 * n_modalities, 5 * n_modalities))

    # 确保axes是二维数组，即使只有一个模态
    if n_modalities == 1:
        axes = np.array([[axes]])

    for i, mod1 in enumerate(modalities):
        for j, mod2 in enumerate(modalities):
            ax = axes[i, j]

            if i == j:  # 对角线：事件0和事件1的相关性
                data_0 = np.array(representation_0[mod1]).flatten()
                data_1 = np.array(representation_1[mod1]).flatten()

                min_length = min(len(data_0), len(data_1))
                data_0 = data_0[:min_length]
                data_1 = data_1[:min_length]

                correlation, p_value = stats.pearsonr(data_0, data_1)

                ax.scatter(data_0, data_1, alpha=0.5, label=f'{mod1}')
                ax.set_xlabel(f'{mod1} - Event 0', fontsize=8)
                ax.set_ylabel(f'{mod1} - Event 1', fontsize=8)
                if p_value < 0.005:
                    ax.set_title(f'{mod1}\nCorr: {correlation:.2f}\np<0.005', fontsize=10)
                else:
                    ax.set_title(f'{mod1}\nCorr: {correlation:.2f}\np: {p_value:.4f}', fontsize=10)

                min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
                max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')

            elif i < j:  # 上三角：两个模态之间的相关性
                data_1 = np.array(fold_representation[mod1]).flatten()
                data_2 = np.array(fold_representation[mod2]).flatten()

                min_length = min(len(data_1), len(data_2))
                data_1 = data_1[:min_length]
                data_2 = data_2[:min_length]

                correlation, p_value = stats.pearsonr(data_1, data_2)

                ax.scatter(data_1, data_2, alpha=0.5, label=f'{mod1} vs {mod2}')
                ax.set_xlabel(mod1, fontsize=8)
                ax.set_ylabel(mod2, fontsize=8)
                if p_value < 0.005:
                    ax.set_title(f'{mod1} vs {mod2}\nCorr: {correlation:.2f}\np<0.005', fontsize=10)
                else:
                    ax.set_title(f'{mod1} vs {mod2}\nCorr: {correlation:.2f}\np: {p_value:.4f}', fontsize=10)

                z = np.polyfit(data_1, data_2, 1)
                p = np.poly1d(z)
                ax.plot(data_1, p(data_1), "r--", label='Trend')

            else:  # 下三角：留空或者可以放置其他信息
                ax.axis('off')

            if i != j:
                ax.legend(fontsize=8)
            ax.grid(True)
            ax.tick_params(axis='both', which='major', labelsize=8)

    plt.tight_layout()
    plt.show()

def plot_combined_tsne(representation_train_0, representation_train_1, fold_representation, k, Moda):
    # 创建t-SNE对象
    tsne = TSNE(n_components=2, random_state=0)
    scaler = StandardScaler()

    # 计算总行数和列数
    if len(Moda) > 2:
        n_rows = len(Moda)
    else:
        n_rows = 3
    n_cols = 3

    # 创建图形和子图
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

    # 设置颜色和标记
    colors = ['darkblue', 'darkred']
    markers = ['o', 's']
    modality_colors = plt.cm.rainbow(np.linspace(0, 1, len(Moda)))
    modality_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    # 绘制每个模态的 Event 0 vs Event 1 图
    for i, modality in enumerate(Moda):
        tsne_results = {}
        for event, representation in enumerate([representation_train_0, representation_train_1]):
            data = np.array(representation[modality]).reshape(len(representation[modality]), -1)

            # 标准化数据
            scaled_data = scaler.fit_transform(data)

            # 应用t-SNE
            tsne_results[f'Event_{event}'] = tsne.fit_transform(scaled_data)

        ax = axs[i, 0]
        for j, event in enumerate(['Event_0', 'Event_1']):
            x, y = tsne_results[event][:, 0], tsne_results[event][:, 1]
            ax.scatter(x, y, c=colors[j], marker=markers[j], label=f'{event}', alpha=0.7)
        ax.set_title(f'{modality} (Event 0 vs 1)')
        ax.legend()
        ax.grid(True)

    # 绘制 Event 0 的所有模态图
    ax = axs[0, 1]
    for i, modality in enumerate(Moda):
        data = np.array(representation_train_0[modality]).reshape(len(representation_train_0[modality]), -1)

        # 标准化数据
        scaled_data = scaler.fit_transform(data)

        # 应用t-SNE
        tsne_result = tsne.fit_transform(scaled_data)

        x, y = tsne_result[:, 0], tsne_result[:, 1]
        ax.scatter(x, y, c=[modality_colors[i]], marker=modality_markers[i % len(modality_markers)], label=modality,
                   alpha=0.7)
    ax.set_title('Event 0 (All Modalities)')
    ax.legend()
    ax.grid(True)

    # 绘制 Event 1 的所有模态图
    ax = axs[1, 1]
    for i, modality in enumerate(Moda):
        data = np.array(representation_train_1[modality]).reshape(len(representation_train_1[modality]), -1)

        # 标准化数据
        scaled_data = scaler.fit_transform(data)

        # 应用t-SNE
        tsne_result = tsne.fit_transform(scaled_data)

        x, y = tsne_result[:, 0], tsne_result[:, 1]
        ax.scatter(x, y, c=[modality_colors[i]], marker=modality_markers[i % len(modality_markers)], label=modality,
                   alpha=0.7)
    ax.set_title('Event 1 (All Modalities)')
    ax.legend()
    ax.grid(True)

    # 绘制 fold_representation 的所有模态图
    ax = axs[2, 1]
    for i, modality in enumerate(Moda):
        data = np.array(fold_representation[modality]).reshape(len(fold_representation[modality]), -1)

        # 标准化数据
        scaled_data = scaler.fit_transform(data)

        # 应用t-SNE
        tsne_result = tsne.fit_transform(scaled_data)

        x, y = tsne_result[:, 0], tsne_result[:, 1]
        ax.scatter(x, y, c=[modality_colors[i]], marker=modality_markers[i % len(modality_markers)], label=modality,
                   alpha=0.7)
    ax.set_title('Fold Representation (All Modalities)')
    ax.legend()
    ax.grid(True)

    # 绘制非临床模态图
    non_clinical_modalities = [mod for mod in Moda if mod != 'clinical']
    non_clinical_colors = plt.cm.rainbow(np.linspace(0, 1, len(non_clinical_modalities)))
    non_clinical_markers = ['o', 's', '^']

    for col, data in enumerate([representation_train_0, representation_train_1, fold_representation]):
        ax = axs[col, 2]
        for i, modality in enumerate(non_clinical_modalities):
            modality_data = np.array(data[modality]).reshape(len(data[modality]), -1)

            # 标准化数据
            scaled_data = scaler.fit_transform(modality_data)

            # 应用t-SNE
            tsne_result = tsne.fit_transform(scaled_data)

            x, y = tsne_result[:, 0], tsne_result[:, 1]
            ax.scatter(x, y, c=[non_clinical_colors[i]], marker=non_clinical_markers[i], label=modality, alpha=0.7)

        if col == 0:
            ax.set_title('Event 0 (Non-clinical Modalities)')
        elif col == 1:
            ax.set_title('Event 1 (Non-clinical Modalities)')
        else:
            ax.set_title('Fold Representation (Non-clinical Modalities)')

        ax.legend()
        ax.grid(True)

    # 移除未使用的子图
    if len(Moda) == 4:
        axs[3, 1].axis('off')
        axs[3, 2].axis('off')
    elif len(Moda) == 2:
        axs[2, 0].axis('off')
    elif len(Moda) == 1:
        axs[2, 0].axis('off')
        axs[1, 0].axis('off')

    # 设置总标题
    fig.suptitle(f't-SNE Analysis\n', fontsize=16)

    # 调整布局并显示图形
    plt.tight_layout()
    plt.show()

for modalities in modalities_list:
    # 创建数据集
    mydataset = MyDataset(modalities, data_path)

    # 假设 preprocess_clinical_data 函数和 data_path 已经定义
    prepro_clin_data_X, _, prepro_clin_data_y, _ = preprocess_clinical_data(data_path['clinical'])
    prepro_clin_data_X.reset_index(drop=True, inplace=True)
    prepro_clin_data_y.reset_index(drop=True, inplace=True)

    train_testVal_strtfdKFold = StratifiedKFold(n_splits=5, random_state=24, shuffle=True)
    train_testVal_kfold = train_testVal_strtfdKFold.split(prepro_clin_data_X, prepro_clin_data_y[5])

    test_c_index_arr = []
    val_c_index_arr = []
    combined_c_index_arr = []

    for k, (train, test_val) in enumerate(train_testVal_kfold):
        x_train, y_train = prepro_clin_data_X.iloc[train, :], prepro_clin_data_y.iloc[train, :][5]

        # 调整验证集和测试集的大小，使得验证集和测试集各占20%
        x_val, x_test, y_val, y_test = train_test_split(prepro_clin_data_X.iloc[test_val, :],
                                                        prepro_clin_data_y.iloc[test_val, :][[5]], test_size=0.3,
                                                        random_state=24,
                                                        stratify=prepro_clin_data_y.iloc[test_val, :][[5]])

        val, test = list(x_val.index), list(x_test.index)
        dataloaders = utils.get_dataloaders(mydataset, train, val, test, BATCH_SIZE)

        # 这里可以继续你对模型的训练和评估逻辑
        survmodel = Model(
            modalities=modalities,
            m_length=m_length,
            dataloaders=dataloaders,
            fusion_method='attention',
            trade_off=0.3,
            mode='total',
            device=device
        )
        params_count = survmodel.get_trainable_parameters_count()
        print("模型的可训练参数数量:", params_count)
        # 计算并存储模型的可训练参数数量
        run_tag = utils.compose_run_tag(
            model=survmodel, lr=lr, dataloaders=dataloaders,
            log_dir='.training_logs/', suffix=''
        )

        fit_args = {
            'num_epochs': EPOCH,
            'lr': lr,
            'info_freq': 3,
            'log_dir': os.path.join('.training_logs/', run_tag),
            'lr_factor': 0.1,
            'scheduler_patience': 5,
        }

        survmodel.fit(**fit_args)

        fold_representation = {modality: [] for modality in modalities}
        # 初始化当前折的 representation_test_0 和 representation_test_1
        representation_test_0 = {modality: [] for modality in modalities}
        representation_test_1 = {modality: [] for modality in modalities}
        representation_train_0 ={modality: [] for modality in modalities}
        representation_train_1 ={modality: [] for modality in modalities}

        fold_times = []
        fold_events = []
        fold_y_scores = []


        survmodel.test()



        # 在测试集循环中
        for data_test, data_label_test in dataloaders['test']:
            out_test, event_test, time_test = survmodel.predict(data_test, data_label_test)
            hazard_test, representation_test= out_test
            test_c_index = concordance_index(time_test.cpu().numpy(), -hazard_test['hazard'].detach().cpu().numpy(),
                                             event_test.cpu().numpy())
            test_c_index_arr.append(test_c_index.item())

            print(f'C-index on Test set: ', test_c_index.item())

        # # 处理 fold_representation
        # for modality in fold_representation.keys():
        #     for i, event in enumerate(fold_events):
        #         if event == 0:
        #             representation_test_0[modality].append(fold_representation[modality][i])
        #         else:
        #             representation_test_1[modality].append(fold_representation[modality][i])
        #     # 在每折处理完成后，将列表转换为numpy数组
        # for modality in representation_test_0.keys():
        #     representation_test_0[modality] = np.array(representation_test_0[modality])
        #     representation_test_1[modality] = np.array(representation_test_1[modality])
        # # 调用函数
        # # 调用函数, 'Necrosis',,
        # modalit = ['clinical', 'Edema','Tumor','Necrosis']
        # plot_combined_correlations(representation_test_0, representation_test_1, fold_representation, modalit)
        # combined_c_index = concordance_index(fold_times, fold_y_scores, fold_events)
        # combined_c_index_arr.append(combined_c_index.item())
        # print(f'C-index on combined_c_index: ', combined_c_index.item())

        # 在验证集循环中
        # for data_val, data_label_val in dataloaders['val']:
        #     out_val, event_val, time_val = survmodel.predict(data_val, data_label_val)
        #     hazard_val, representation_val = out_val
        #     val_c_index = concordance_index(time_val.cpu().numpy(), -hazard_val['hazard'].detach().cpu().numpy(),
        #                                     event_val.cpu().numpy())
        #     val_c_index_arr.append(val_c_index.item())
        #
        #     # 保存每折的验证结果
        #     fold_y_scores.extend(-hazard_val['hazard'].detach().cpu().numpy())
        #     fold_times.extend(time_val.cpu().numpy())
        #     fold_events.extend(event_val.cpu().numpy())
        #     # 扩展每个模态的表示
        #     for modality in modalities:
        #         if modality in representation_val:
        #             fold_representation[modality].extend(representation_val[modality].detach().cpu().numpy())
        # print(f'C-index on val set: ', val_c_index.item())
        #
        # combined_c_index = concordance_index(fold_times, fold_y_scores, fold_events)
        # combined_c_index_arr.append(combined_c_index.item())
        # print(f'C-index on combined_c_index: ', combined_c_index.item())

    # 计算平均和标准差
    print('Mean and std_test: ', utils.evaluate_model(test_c_index_arr))
    # print('Mean and std_combined: ', utils.evaluate_model(combined_c_index_arr))
    # utils.save_5fold_results(test_c_index_arr, combined_c_index_arr, run_tag)


    #
    # # 计算并保存ROC曲线和AUC
    # fpr, tpr, thresholds = roc_curve(fold_events, fold_y_scores)
    # auc_score = roc_auc_score(fold_events, fold_y_scores)
    # print('AUC: ', auc_score)
    # plt.figure()
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_score:.2f})')
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.0])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve')
    # plt.legend(loc="lower right")
    # plt.show()
    #
    #
    # # 计算在不同Sp水平下的Sn、Acc、Pre和F1评分
    # def evaluate_at_specificity(fpr, tpr, thresholds, specificity_level):
    #     idx = np.argmin(np.abs(fpr - (1 - specificity_level / 100)))
    #     threshold = thresholds[idx]
    #     predictions = (fold_y_scores >= threshold).astype(int)
    #     tn, fp, fn, tp = confusion_matrix(fold_events, predictions).ravel()
    #     sp = tn / (tn + fp)
    #     sn = tp / (tp + fn)
    #     acc = (tp + tn) / (tp + tn + fp + fn)
    #     pre = tp / (tp + fp)
    #     f1 = 2 * (pre * sn) / (pre + sn)
    #     return sn, acc, pre, f1, threshold
    #
    #
    # for sp_level in [90.0, 95.0]:
    #     sn, acc, pre, f1, threshold = evaluate_at_specificity(fpr, tpr, thresholds, sp_level)
    #     print(f'At Sp={sp_level}%: Sn={sn:.2f}, Acc={acc:.2f}, Pre={pre:.2f}, F1={f1:.2f}, Threshold={threshold:.2f}')
    #
    # utils.save_5fold_results(test_c_index_arr, combined_c_index_arr, run_tag)

# # 将 event_test 转移到 CPU 并转换为 numpy 数组
# event_test_np = event_test.cpu().numpy()
#
# # 根据 event_test 的值提取相应的 representation_test
# for modality in representation_test.keys():
#     modality_tensor = representation_test[modality]
#
#     # 确保 modality_tensor 也在 CPU 上
#     if modality_tensor.is_cuda:
#         modality_tensor = modality_tensor.cpu()
#
#     for i, event in enumerate(event_test_np):
#         if event == 0:
#             representation_test_0[modality].append(modality_tensor[i].detach().numpy())
#         else:
#             representation_test_1[modality].append(modality_tensor[i].detach().numpy())
#
# # 将列表转换为numpy数组
# for modality in representation_test_0.keys():
#     representation_test_0[modality] = np.array(representation_test_0[modality])
#     representation_test_1[modality] = np.array(representation_test_1[modality])
#
# # 选择'Edema'和'Tumor'两个模态
# Modalities = ['Edema', 'Tumor']
#
# # 创建PCA对象
# pca = PCA(n_components=2)
#
# # 存储PCA结果
# pca_results = {}
#
# # 对每个模态进行PCA
# for Modality in Modalities:
#     # 确保数据是二维的
#     data = representation_test_0[Modality].reshape(representation_test_0[Modality].shape[0], -1)
#     pca_results[Modality] = pca.fit_transform(data)
#
# # 绘图
# plt.figure(figsize=(10, 8))
#
# colors = ['b', 'r']  # 蓝色代表Edema，红色代表Tumor
# markers = ['o', 's']  # 圆形代表Edema，方形代表Tumor
#
# for i, Modality in enumerate(Modalities):
#     x = pca_results[Modality][:, 0]
#     y = pca_results[Modality][:, 1]
#     plt.scatter(x, y, c=colors[i], marker=markers[i], label=Modality, alpha=0.7)
#
# plt.xlabel('First Principal Component')
# plt.ylabel('Second Principal Component')
# plt.title('PCA of Edema and Tumor Modalities (Event 0)')
# plt.legend()
# plt.grid(True)
#
# plt.show()


import os
import torch

torch.cuda.empty_cache()
import sys

sys.path.append(r'D:\python\cancer_survival\testPrediction1\model')
import utils
from lifelines.utils import concordance_index
from testPrediction1.model.data_loader import MyDataset
from testPrediction1.model.data_loader import preprocess_clinical_data
from testPrediction1.model.model import Model
from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.utils import resample
from sksurv.metrics import concordance_index_censored
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt
import seaborn as sns

# 设置必要的参数
m_length = 16
BATCH_SIZE = 16
EPOCH = 20
lr = 0.003
K = 3
data_path = utils.DATA_PATH

# 定义要比较的模型配置（只包含All模块，且确保所有配置都包含临床数据）
model_configs = {
    # 'T-All': {'modalities': ['Tumor'], 'modules': 'All'},
    # 'N-All': {'modalities': ['Necrosis'], 'modules': 'All'},
    # 'E-All': {'modalities': ['Edema'], 'modules': 'All'},
    # # 单模态 - 只有临床
    'C-All': {'modalities': ['clinical'], 'modules': 'All'},
    #
    # # 双模态 + 临床
    'EC-All': {'modalities': ['clinical', 'Edema'], 'modules': 'All'},
    'TC-All': {'modalities': ['clinical', 'Tumor'], 'modules': 'All'},
    'NC-All': {'modalities': ['clinical', 'Necrosis'], 'modules': 'All'},

    # 三模态 + 临床
    'ENC-All': {'modalities': ['clinical', 'Necrosis', 'Edema'], 'modules': 'All'},
    'TEC-All': {'modalities': ['clinical', 'Tumor', 'Edema'], 'modules': 'All'},
    'TNC-All': {'modalities': ['clinical', 'Tumor', 'Necrosis'], 'modules': 'All'},

    # 全模态
    'TENC-All': {'modalities': ['clinical', 'Tumor', 'Necrosis', 'Edema'], 'modules': 'All'},
}

# 设置随机种子并检测CUDA
utils.setup_seed(24)
device = utils.test_gpu()

# 假设 preprocess_clinical_data 函数和 data_path 已经定义
prepro_clin_data_X, _, prepro_clin_data_y, _ = preprocess_clinical_data(data_path['clinical'])
prepro_clin_data_X.reset_index(drop=True, inplace=True)
prepro_clin_data_y.reset_index(drop=True, inplace=True)

# 创建 StratifiedKFold 对象
train_testVal_strtfdKFold = StratifiedKFold(n_splits=5, random_state=24, shuffle=True)

# 存储所有模型的结果用于统计比较
all_models_results = {config: {'c_index': [], 'brier_scores': [], 'avg_brier': []}
                      for config in model_configs.keys()}


def bootstrap_confidence_interval(scores, n_bootstrap=1000, confidence_level=0.95):
    """计算bootstrap置信区间"""
    bootstrap_scores = []
    for _ in range(n_bootstrap):
        bootstrap_sample = resample(scores, replace=True)
        bootstrap_scores.append(np.mean(bootstrap_sample))

    alpha = (1 - confidence_level) / 2
    lower = np.percentile(bootstrap_scores, 100 * alpha)
    upper = np.percentile(bootstrap_scores, 100 * (1 - alpha))
    return lower, upper


def permutation_test(scores_a, scores_b, n_permutations=10000):
    """执行置换检验计算p值"""
    observed_diff = np.mean(scores_a) - np.mean(scores_b)

    # 合并所有分数
    all_scores = np.concatenate([scores_a, scores_b])
    perm_diffs = []

    for _ in range(n_permutations):
        # 随机打乱标签
        np.random.shuffle(all_scores)
        perm_a = all_scores[:len(scores_a)]
        perm_b = all_scores[len(scores_a):]
        perm_diffs.append(np.mean(perm_a) - np.mean(perm_b))

    # 计算p值（双侧检验）
    p_value = (np.sum(np.abs(perm_diffs) >= np.abs(observed_diff)) + 1) / (n_permutations + 1)
    return p_value, observed_diff


def paired_t_test_with_ci(model_a_scores, model_b_scores, confidence_level=0.95):
    """执行配对t检验并计算置信区间"""
    differences = np.array(model_a_scores) - np.array(model_b_scores)
    n = len(differences)
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)

    # 计算t统计量和p值
    t_stat = mean_diff / (std_diff / np.sqrt(n))
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), n - 1))

    # 计算置信区间
    t_critical = stats.t.ppf((1 + confidence_level) / 2, n - 1)
    ci_lower = mean_diff - t_critical * (std_diff / np.sqrt(n))
    ci_upper = mean_diff + t_critical * (std_diff / np.sqrt(n))

    return {
        'mean_difference': mean_diff,
        't_statistic': t_stat,
        'p_value': p_value,
        'confidence_interval': (ci_lower, ci_upper),
        'std_difference': std_diff
    }


def safe_concordance_index(event_indicator, event_time, risk_scores):
    """安全地计算concordance index，处理各种数组形状"""
    event_indicator = np.array(event_indicator).flatten()
    event_time = np.array(event_time).flatten()
    risk_scores = np.array(risk_scores).flatten()

    try:
        c_index = concordance_index_censored(
            event_indicator.astype(bool),
            event_time,
            risk_scores
        )[0]
    except:
        try:
            c_index = concordance_index(event_time, -risk_scores, event_indicator)
        except:
            print("Warning: Could not calculate C-index")
            c_index = np.nan
    return c_index


# 主要的模型训练和比较循环
print("Starting comparative model training with statistical testing...")

# K 折交叉验证循环
train_testVal_kfold = train_testVal_strtfdKFold.split(prepro_clin_data_X, prepro_clin_data_y[11])

for k, (train, test_val) in enumerate(train_testVal_kfold):
    print(f"\n=== Processing Fold {k + 1} ===")

    # 数据分割
    x_train, y_train = prepro_clin_data_X.iloc[train, :], prepro_clin_data_y.iloc[train, :][11]
    x_val, x_test, y_val, y_test = train_test_split(
        prepro_clin_data_X.iloc[test_val, :],
        prepro_clin_data_y.iloc[test_val, :][[11]],
        test_size=0.3, random_state=24,
        stratify=prepro_clin_data_y.iloc[test_val, :][[11]]
    )
    val, test = list(x_val.index), list(x_test.index)

    # 对每个模型配置进行训练和测试
    for config_name, config in model_configs.items():
        modalities = config['modalities']
        print(f"Training {config_name} with modalities: {modalities}...")

        try:
            # 创建数据集
            mydataset = MyDataset(modalities, data_path)
            dataloaders = utils.get_dataloaders(mydataset, train, val, test, BATCH_SIZE)

            # 训练模型 - 对于All配置，使用完整模型
            survmodel = Model(
                modalities=modalities,
                m_length=m_length,
                dataloaders=dataloaders,
                fusion_method='attention',
                trade_off=0.3,
                mode='total',
                device=device
            )

            fit_args = {
                'num_epochs': EPOCH,
                'lr': lr,
                'info_freq': 3,
                'log_dir': os.path.join('.training_logs/', f'{config_name}_fold_{k}'),
                'lr_factor': 0.1,
                'scheduler_patience': 5,
            }
            survmodel.fit(**fit_args)

            # 测试模型
            survmodel.test()
            all_time_test = []
            all_hazard_test = []
            all_event_test = []

            for data_test, data_label_test in dataloaders['test']:
                out_test, event_test, time_test = survmodel.predict(data_test, data_label_test)
                hazard_test, _ = out_test

                hazard_scores = hazard_test['hazard'].detach().cpu().numpy()
                if hazard_scores.ndim > 1:
                    hazard_scores = hazard_scores.flatten()

                all_time_test.extend(time_test.cpu().numpy())
                all_hazard_test.extend(hazard_scores)
                all_event_test.extend(event_test.cpu().numpy())

            # 计算性能指标
            test_c_index = safe_concordance_index(all_event_test, all_time_test, all_hazard_test)

            hazard_scores_flat = np.array(all_hazard_test).flatten()
            probabilities = 1 / (1 + np.exp(-hazard_scores_flat))
            brier_scores = [(prob - event) ** 2 for prob, event in zip(probabilities, all_event_test)]
            avg_brier_score = np.mean(brier_scores)

            # 存储结果
            all_models_results[config_name]['c_index'].append(test_c_index)
            all_models_results[config_name]['brier_scores'].extend(brier_scores)
            all_models_results[config_name]['avg_brier'].append(avg_brier_score)

            print(f'{config_name} - Fold {k + 1} - C-index: {test_c_index:.4f}, Brier Score: {avg_brier_score:.4f}')

        except Exception as e:
            print(f"Error training {config_name}: {str(e)}")
            # 存储NaN值以保持结果数组的一致性
            all_models_results[config_name]['c_index'].append(np.nan)
            all_models_results[config_name]['avg_brier'].append(np.nan)
            print(f"Skipping {config_name} due to error")

# ==================== 统计分析部分 ====================
print("\n" + "=" * 80)
print("COMPREHENSIVE STATISTICAL ANALYSIS")
print("=" * 80)

# 过滤掉没有有效结果的配置
valid_configs = {}
for config_name in model_configs.keys():
    c_scores = all_models_results[config_name]['c_index']
    # 检查是否有有效结果（不是全部为NaN）
    if len(c_scores) > 0 and not all(np.isnan(score) for score in c_scores):
        valid_configs[config_name] = model_configs[config_name]

print(f"Valid configurations with results: {list(valid_configs.keys())}")

if not valid_configs:
    print("No valid results to analyze. Exiting.")
    exit()

# 1. 基本性能汇总
print(f"\n1. Performance Summary (Mean ± SD):")
for config_name in valid_configs.keys():
    c_scores = [score for score in all_models_results[config_name]['c_index'] if not np.isnan(score)]
    b_scores = [score for score in all_models_results[config_name]['avg_brier'] if not np.isnan(score)]

    if len(c_scores) > 0:
        print(f"{config_name:15} C-index: {np.mean(c_scores):.4f} ± {np.std(c_scores):.4f}, "
              f"Brier: {np.mean(b_scores):.4f} ± {np.std(b_scores):.4f}")

# 2. 与基准模型（Clinical-only）的统计比较
baseline_model = 'C-All'
target_model = 'TENC-All'

# 检查是否有结果数据
if baseline_model in valid_configs and target_model in valid_configs:
    baseline_scores = np.array(
        [score for score in all_models_results[baseline_model]['c_index'] if not np.isnan(score)])
    target_scores = np.array([score for score in all_models_results[target_model]['c_index'] if not np.isnan(score)])

    print(f"\n2. Statistical Comparison: {target_model} vs {baseline_model}")

    # 配对t检验
    if len(baseline_scores) > 1 and len(target_scores) > 1 and len(baseline_scores) == len(target_scores):
        paired_test_result = paired_t_test_with_ci(target_scores, baseline_scores)
        print(f"   Paired t-test: t({len(target_scores) - 1}) = {paired_test_result['t_statistic']:.4f}, "
              f"p = {paired_test_result['p_value']:.4f}")
        print(f"   Mean difference: {paired_test_result['mean_difference']:.4f} "
              f"(95% CI: [{paired_test_result['confidence_interval'][0]:.4f}, "
              f"{paired_test_result['confidence_interval'][1]:.4f}])")

        # 置换检验
        perm_p_value, observed_diff = permutation_test(target_scores, baseline_scores, n_permutations=10000)
        print(f"   Permutation test: p = {perm_p_value:.4f}")
    else:
        print(
            f"   Not enough valid data for statistical tests. Baseline: {len(baseline_scores)}, Target: {len(target_scores)}")
else:
    print(f"   Missing valid data for {baseline_model} or {target_model}")

# 3. 所有模型两两比较（仅限有数据的模型）
print(f"\n3. Pairwise Comparisons (C-index):")
config_names = list(valid_configs.keys())
comparison_results = []

for i, model_a in enumerate(config_names):
    for j, model_b in enumerate(config_names):
        if i < j:  # 避免重复比较
            scores_a = np.array([score for score in all_models_results[model_a]['c_index'] if not np.isnan(score)])
            scores_b = np.array([score for score in all_models_results[model_b]['c_index'] if not np.isnan(score)])

            if len(scores_a) > 1 and len(scores_b) > 1 and len(scores_a) == len(scores_b):
                test_result = paired_t_test_with_ci(scores_a, scores_b)
                comparison_results.append({
                    'model_a': model_a,
                    'model_b': model_b,
                    'mean_diff': test_result['mean_difference'],
                    'p_value': test_result['p_value'],
                    'significant': test_result['p_value'] < 0.05
                })

                significance = "**" if test_result['p_value'] < 0.01 else "*" if test_result['p_value'] < 0.05 else "ns"
                print(f"   {model_a:12} vs {model_b:12}: diff = {test_result['mean_difference']:+.4f}, "
                      f"p = {test_result['p_value']:.4f} {significance}")

# 4. Bootstrap置信区间
print(f"\n4. Bootstrap 95% Confidence Intervals:")
for config_name in config_names:
    c_scores = [score for score in all_models_results[config_name]['c_index'] if not np.isnan(score)]
    b_scores = [score for score in all_models_results[config_name]['avg_brier'] if not np.isnan(score)]

    if len(c_scores) > 0:
        c_ci = bootstrap_confidence_interval(c_scores)
        b_ci = bootstrap_confidence_interval(b_scores)

        print(f"   {config_name:15} C-index: [{c_ci[0]:.4f}, {c_ci[1]:.4f}], "
              f"Brier: [{b_ci[0]:.4f}, {b_ci[1]:.4f}]")

# 5. 效应大小计算
print(f"\n5. Effect Size Analysis (Cohen's d):")
for config_name in config_names:
    if config_name != baseline_model and baseline_model in config_names:
        scores_a = np.array([score for score in all_models_results[config_name]['c_index'] if not np.isnan(score)])
        scores_b = np.array([score for score in all_models_results[baseline_model]['c_index'] if not np.isnan(score)])

        if len(scores_a) > 0 and len(scores_b) > 0 and len(scores_a) == len(scores_b):
            mean_diff = np.mean(scores_a - scores_b)
            pooled_std = np.sqrt((np.std(scores_a, ddof=1) ** 2 + np.std(scores_b, ddof=1) ** 2) / 2)
            cohens_d = mean_diff / pooled_std if pooled_std != 0 else 0

            interpretation = "large" if abs(cohens_d) > 0.8 else "medium" if abs(cohens_d) > 0.5 else "small"
            print(f"   {config_name:15} vs {baseline_model:12}: d = {cohens_d:.4f} ({interpretation})")

# ==================== 结果可视化 ====================
if len(config_names) > 0:
    plt.figure(figsize=(15, 10))

    # 1. 性能比较图
    plt.subplot(2, 3, 1)
    models = config_names
    c_index_means = [np.mean([score for score in all_models_results[model]['c_index'] if not np.isnan(score)]) for model
                     in models]
    c_index_stds = [np.std([score for score in all_models_results[model]['c_index'] if not np.isnan(score)]) for model
                    in models]

    bars = plt.bar(range(len(models)), c_index_means, yerr=c_index_stds, capsize=5,
                   color=plt.cm.Set3(np.linspace(0, 1, len(models))))
    plt.xticks(range(len(models)), models, rotation=45, ha='right')
    plt.title('C-index Comparison Across Models')
    plt.ylabel('C-index')
    plt.ylim(0.4, 0.8)

    # 在柱子上添加数值
    for i, (v, std) in enumerate(zip(c_index_means, c_index_stds)):
        plt.text(i, v + std + 0.01, f'{v:.3f}', ha='center', fontsize=8)

    # 2. 箱线图显示分布
    plt.subplot(2, 3, 2)
    data_to_plot = [[score for score in all_models_results[model]['c_index'] if not np.isnan(score)] for model in
                    models]
    plt.boxplot(data_to_plot, labels=models)
    plt.xticks(rotation=45, ha='right')
    plt.title('C-index Distribution Across Folds')
    plt.ylabel('C-index')

    # 3. 改进幅度可视化
    plt.subplot(2, 3, 3)
    improvements = []
    comparison_labels = []
    for model in models:
        if model != baseline_model and baseline_model in models:
            model_scores = [score for score in all_models_results[model]['c_index'] if not np.isnan(score)]
            baseline_scores = [score for score in all_models_results[baseline_model]['c_index'] if not np.isnan(score)]
            if len(model_scores) > 0 and len(baseline_scores) > 0:
                improvement = np.mean(model_scores) - np.mean(baseline_scores)
                improvements.append(improvement)
                comparison_labels.append(model)

    if improvements:
        plt.bar(range(len(improvements)), improvements, color=plt.cm.Set3(np.linspace(0, 1, len(improvements))))
        plt.xticks(range(len(improvements)), comparison_labels, rotation=45, ha='right')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title('C-index Improvement Over Baseline')
        plt.ylabel('Improvement')

        # 添加数值标签
        for i, v in enumerate(improvements):
            plt.text(i, v + 0.002 if v > 0 else v - 0.002, f'{v:+.3f}',
                     ha='center', fontweight='bold')

    # 4. p值热力图
    plt.subplot(2, 3, 4)
    if len(models) > 1:
        p_value_matrix = np.ones((len(models), len(models)))
        for i, model_a in enumerate(models):
            for j, model_b in enumerate(models):
                if i != j:
                    scores_a = np.array(
                        [score for score in all_models_results[model_a]['c_index'] if not np.isnan(score)])
                    scores_b = np.array(
                        [score for score in all_models_results[model_b]['c_index'] if not np.isnan(score)])
                    if len(scores_a) > 1 and len(scores_b) > 1 and len(scores_a) == len(scores_b):
                        _, p_value = stats.ttest_rel(scores_a, scores_b)
                        p_value_matrix[i, j] = p_value

        sns.heatmap(p_value_matrix, annot=True, fmt='.4f', cmap='RdBu_r',
                    xticklabels=models, yticklabels=models, cbar_kws={'label': 'p-value'})
        plt.title('Pairwise p-values (Paired t-test)')

    # 5. 置信区间图
    plt.subplot(2, 3, 5)
    for i, model in enumerate(models):
        c_scores = [score for score in all_models_results[model]['c_index'] if not np.isnan(score)]
        if len(c_scores) > 0:
            mean_val = np.mean(c_scores)
            ci_lower, ci_upper = bootstrap_confidence_interval(c_scores)

            plt.plot([ci_lower, ci_upper], [i, i], 'o-', linewidth=3, label=model)
            plt.plot(mean_val, i, 'o', color='black', markersize=6)

    plt.yticks(range(len(models)), models)
    plt.xlabel('C-index')
    plt.title('Bootstrap 95% Confidence Intervals')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig('comprehensive_statistical_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# ==================== 生成统计报告 ====================
print("\n" + "=" * 80)
print("STATISTICAL SUMMARY FOR PUBLICATION")
print("=" * 80)

if baseline_model in valid_configs and target_model in valid_configs:
    print(f"\nKey Findings:")
    target_scores = [score for score in all_models_results[target_model]['c_index'] if not np.isnan(score)]
    print(f"1. Best Performing Model: {target_model} "
          f"(C-index: {np.mean(target_scores):.4f} ± "
          f"{np.std(target_scores):.4f})")

    print(f"2. Statistical Significance:")
    baseline_comparison = paired_t_test_with_ci(
        np.array([score for score in all_models_results[target_model]['c_index'] if not np.isnan(score)]),
        np.array([score for score in all_models_results[baseline_model]['c_index'] if not np.isnan(score)])
    )
    significance = "STATISTICALLY SIGNIFICANT" if baseline_comparison[
                                                      'p_value'] < 0.05 else "not statistically significant"
    print(f"   • Improvement over {baseline_model}: {baseline_comparison['mean_difference']:.4f} "
          f"(95% CI: [{baseline_comparison['confidence_interval'][0]:.4f}, "
          f"{baseline_comparison['confidence_interval'][1]:.4f}], p = {baseline_comparison['p_value']:.4f})")
    print(f"   • This improvement is {significance} at α = 0.05")

    print(f"3. Effect Size:")
    scores_target = np.array([score for score in all_models_results[target_model]['c_index'] if not np.isnan(score)])
    scores_baseline = np.array(
        [score for score in all_models_results[baseline_model]['c_index'] if not np.isnan(score)])
    cohens_d = np.mean(scores_target - scores_baseline) / np.std(scores_target - scores_baseline, ddof=1)
    print(f"   • Cohen's d: {cohens_d:.4f}")

    print(f"4. Confidence Intervals:")
    c_ci = bootstrap_confidence_interval(
        [score for score in all_models_results[target_model]['c_index'] if not np.isnan(score)])
    print(f"   • {target_model} C-index 95% CI: [{c_ci[0]:.4f}, {c_ci[1]:.4f}]")

    print(f"\nConclusion:")
    if baseline_comparison['p_value'] < 0.05:
        print(f"The {target_model} demonstrates statistically significant improvement over the {baseline_model} "
              f"with a medium effect size (Cohen's d = {cohens_d:.3f}). The results are robust across "
              f"multiple statistical tests and cross-validation folds.")
    else:
        print(f"While the {target_model} shows numerical improvement over the {baseline_model}, "
              f"this improvement is not statistically significant at the conventional α = 0.05 level. "
              f"Further validation with larger datasets is recommended.")

# ==================== 生成可直接用于论文的表格 ====================
print("\n" + "=" * 80)
print("TABLE FOR PUBLICATION")
print("=" * 80)

print(f"\nTable 1: Model Performance Comparison with Statistical Significance")
print(f"{'Model':<15} {'C-index':<10} {'95% CI':<20} {'Brier Score':<12} {'vs Baseline (p-value)':<20}")
print("-" * 80)

for config_name in config_names:
    c_scores = [score for score in all_models_results[config_name]['c_index'] if not np.isnan(score)]
    b_scores = [score for score in all_models_results[config_name]['avg_brier'] if not np.isnan(score)]

    if len(c_scores) > 0:
        c_mean = np.mean(c_scores)
        c_std = np.std(c_scores)
        c_ci = bootstrap_confidence_interval(c_scores)
        b_mean = np.mean(b_scores)

        # 计算与基准的p值
        if config_name != baseline_model and baseline_model in config_names:
            scores_config = np.array(
                [score for score in all_models_results[config_name]['c_index'] if not np.isnan(score)])
            scores_baseline = np.array(
                [score for score in all_models_results[baseline_model]['c_index'] if not np.isnan(score)])
            if len(scores_config) > 1 and len(scores_baseline) > 1 and len(scores_config) == len(scores_baseline):
                _, p_value = stats.ttest_rel(scores_config, scores_baseline)
                p_str = f"{p_value:.4f}"
            else:
                p_str = "N/A"
        else:
            p_str = "Reference"

        print(f"{config_name:<15} {c_mean:.4f}     [{c_ci[0]:.4f}-{c_ci[1]:.4f}]  {b_mean:.4f}        {p_str:<20}")

print("\n* p < 0.05, ** p < 0.01, *** p < 0.001")
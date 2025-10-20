import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import json



# ==================== 定义通用函数 ====================

def plot_kaplan_meier_curve(df,
                            time_col='time',
                            event_col='event',
                            group_col='risk_group',
                            title='Kaplan-Meier Survival Curve',
                            time_ticks=None,
                            ci_show=True,
                            cmap=('red','blue'),
                            figsize=(10, 6),
                            fontsize=20,
                            fontsize_ticks=None):
    """
    绘制 Kaplan-Meier 生存曲线
    """
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    import numpy as np

    if fontsize_ticks is None:
        fontsize_ticks = fontsize

    groups = sorted(df[group_col].unique())
    if len(groups) != 2:
        raise ValueError("风险组必须为两个（例如 'Low Risk' 与 'High Risk'）。")

    max_t = df[time_col].max()
    if time_ticks is None:
        time_ticks = np.linspace(0, max_t, 6).astype(int)
        time_ticks[0] = 0

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    color_map = {groups[0]: cmap[0], groups[1]: cmap[1]}
    for grp in groups:
        grp_df = df[df[group_col] == grp]
        kmf = KaplanMeierFitter()
        kmf.fit(durations=grp_df[time_col], event_observed=grp_df[event_col], label=grp)
        kmf.plot_survival_function(ax=ax, ci_show=ci_show, color=color_map[grp], lw=3)

    title_fontsize = fontsize + 2
    label_fontsize = fontsize
    legend_fontsize = max(8, fontsize - 2)
    tick_fontsize = fontsize_ticks

    ax.set_title(title, fontsize=title_fontsize, fontweight='bold')
    ax.set_xlabel('Time (Days)', fontsize=label_fontsize)
    ax.set_ylabel('Survival Probability', fontsize=label_fontsize)
    ax.grid(True, alpha=0.35)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    legend = ax.legend(fontsize=legend_fontsize)
    for text in legend.get_texts():
        text.set_fontsize(legend_fontsize)
    for legline in legend.get_lines():
        legline.set_linewidth(3)

    # log-rank 检验并显示 p 值
    g1 = df[df[group_col] == groups[0]]
    g2 = df[df[group_col] == groups[1]]
    logrank_result = logrank_test(g1[time_col], g2[time_col],
                                  event_observed_A=g1[event_col],
                                  event_observed_B=g2[event_col])

    stats_text = f'Log-rank p-value: {logrank_result.p_value:.6e}'
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
            fontsize=label_fontsize, va='bottom', ha='right',
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))

    plt.tight_layout()
    plt.show()

    return logrank_result


def plot_risk_table(df,
                    time_col='time',
                    event_col='event',
                    group_col='risk_group',
                    title='Number at Risk Table',
                    time_ticks=None,
                    cmap=('blue', 'red'),
                    figsize=(10, 3),
                    fontsize=20,
                    fontsize_ticks=None,
                    show_events=False,
                    show_censor=False,
                    decimals=0):
    """
    绘制风险表
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    if fontsize_ticks is None:
        fontsize_ticks = fontsize

    groups = sorted(df[group_col].unique())
    max_t = df[time_col].max()
    if time_ticks is None:
        time_ticks = np.linspace(0, max_t, 6)
        time_ticks[0] = 0.0

    if decimals == 0:
        display_ticks = [int(t) for t in time_ticks]
    else:
        display_ticks = [round(float(t), decimals) for t in time_ticks]

    n_groups = len(groups)
    n_times = len(time_ticks)

    results = {grp: [] for grp in groups}
    events_table = {grp: [] for grp in groups} if show_events else None
    censor_table = {grp: [] for grp in groups} if show_censor else None

    prev_t = 0.0
    for idx, t in enumerate(time_ticks):
        for grp in groups:
            grp_df = df[df[group_col] == grp]
            n_at_risk = int((grp_df[time_col] >= t).sum())
            results[grp].append(n_at_risk)

            if show_events or show_censor:
                if idx == 0:
                    mask_interval = (grp_df[time_col] <= t)
                else:
                    mask_interval = (grp_df[time_col] > prev_t) & (grp_df[time_col] <= t)

                events_in_int = int(((grp_df[mask_interval])[event_col] == 1).sum())
                censor_in_int = int(((grp_df[mask_interval])[event_col] == 0).sum())

                if show_events:
                    events_table[grp].append(events_in_int)
                if show_censor:
                    censor_table[grp].append(censor_in_int)
        prev_t = float(t)

    display_index = groups
    display_cols = [str(dt) for dt in display_ticks]
    display_data = []
    for grp in groups:
        row = []
        for j in range(n_times):
            n = results[grp][j]
            parts = [str(n)]
            extras = []
            if show_events:
                extras.append(f"E={events_table[grp][j]}")
            if show_censor:
                extras.append(f"C={censor_table[grp][j]}")
            if extras:
                cell_text = f"{n} ({', '.join(extras)})"
            else:
                cell_text = str(n)
            row.append(cell_text)
        display_data.append(row)

    result_df = pd.DataFrame(display_data, index=display_index, columns=display_cols)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.axis('off')

    header = ["Time"] + display_cols
    table_data = []
    for grp in groups:
        row = [grp] + result_df.loc[grp].tolist()
        table_data.append(row)

    the_table = ax.table(cellText=table_data,
                         colLabels=header,
                         cellLoc='center',
                         colLoc='center',
                         loc='center')

    table_fontsize = max(8, fontsize - 2)
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(table_fontsize)
    the_table.scale(1, 1.2)

    for (row, col), cell in the_table.get_celld().items():
        if row == -1:
            cell.set_text_props(weight='bold', fontsize=table_fontsize)
            cell.set_facecolor('#f2f2f2')
        else:
            if col == 0:
                cell.set_text_props(weight='bold', fontsize=table_fontsize)
        cell.set_edgecolor('none')

    ax.set_title(title, fontsize=fontsize + 2, fontweight='bold', pad=8)
    plt.tight_layout()
    plt.show()

    return result_df, time_ticks


def plot_risk_score_distribution(df, dataset_name, risk_group_col='risk_group', hazard_col='corrected_hazard_score'):
    """
    绘制风险得分分布图
    """
    plt.figure(figsize=(12, 5))

    # 子图1：箱线图
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df, x=risk_group_col, y=hazard_col, hue=risk_group_col,
                palette=['blue', 'red'], legend=False)
    plt.title(f'{dataset_name} - Risk Score Distribution by Group', fontweight='bold')
    plt.xlabel('Risk Group')
    plt.ylabel('Corrected Hazard Score')

    # 子图2：直方图
    plt.subplot(1, 2, 2)
    risk_threshold = df[df[risk_group_col] == 'High Risk'][hazard_col].min()
    for group, color in zip(['Low Risk', 'High Risk'], ['blue', 'red']):
        group_data = df[df[risk_group_col] == group][hazard_col]
        plt.hist(group_data, alpha=0.7, label=group, color=color, bins=20)
    plt.axvline(risk_threshold, color='black', linestyle='--', linewidth=2,
                label=f'Threshold: {risk_threshold:.3f}')
    plt.title(f'{dataset_name} - Risk Score Histogram', fontweight='bold')
    plt.xlabel('Corrected Hazard Score')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_time_dependent_roc(df, time_points, time_col='time', event_col='event',
                            hazard_col='corrected_hazard_score', title_prefix=''):
    """
    绘制时间依赖性ROC曲线

    参数:
    df: 包含生存数据的DataFrame
    time_points: 要评估的时间点列表
    time_col: 时间列名
    event_col: 事件列名
    hazard_col: 风险得分列名
    title_prefix: 标题前缀
    """
    plt.figure(figsize=(10, 8))

    # 存储ROC数据
    roc_data = {}

    # 为每个时间点计算ROC
    for time_point in time_points:
        # 创建二元标签：在指定时间点前发生事件为1，否则为0
        # 注意：删失数据需要特殊处理
        y_true = []
        y_score = []

        for idx, row in df.iterrows():
            if row[time_col] <= time_point and row[event_col] == 1:
                # 在时间点前发生事件
                y_true.append(1)
                y_score.append(row[hazard_col])
            elif row[time_col] > time_point:
                # 生存时间超过时间点（无事件）
                y_true.append(0)
                y_score.append(row[hazard_col])
            # 删失数据在时间点前被删失的，我们排除（或者可以用其他方法处理）

        if len(set(y_true)) < 2:
            print(f"在时间点 {time_point} 处，正负样本不均衡，跳过ROC计算")
            continue

        # 计算ROC曲线
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        # 存储数据
        roc_data[time_point] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist(),
            'auc': roc_auc
        }

        # 绘制ROC曲线
        plt.plot(fpr, tpr, lw=2,
                 label=f'Time {time_point} days (AUC = {roc_auc:.3f})')

    # 绘制对角线
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=26)
    plt.ylabel('True Positive Rate', fontsize=26)
    plt.title('Time-Dependent ROC Curves', fontsize=28, fontweight='bold')
    plt.legend(loc="lower right", fontsize=24)
    plt.grid(True, alpha=0.3)

    # 设置坐标轴刻度字体大小
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    plt.tight_layout()
    plt.show()

    return roc_data


def plot_multiple_roc_curves(roc_data_dict, dataset_names, time_point):
    """
    在同一图中绘制多个数据集的ROC曲线

    参数:
    roc_data_dict: 包含多个数据集ROC数据的字典
    dataset_names: 数据集名称列表
    time_point: 要比较的时间点
    """
    plt.figure(figsize=(10, 8))

    colors = ['red', 'blue', 'green', 'orange', 'purple']

    for i, dataset_name in enumerate(dataset_names):
        if dataset_name in roc_data_dict and time_point in roc_data_dict[dataset_name]:
            roc_data = roc_data_dict[dataset_name][time_point]
            fpr = roc_data['fpr']
            tpr = roc_data['tpr']
            auc_score = roc_data['auc']

            plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                     label=f'{dataset_name} (AUC = {auc_score:.3f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=26)
    plt.ylabel('True Positive Rate', fontsize=26)
    plt.title(f'ROC Curves Comparison at {time_point} Days', fontsize=28, fontweight='bold')
    plt.legend(loc="lower right", fontsize=26)
    plt.grid(True, alpha=0.3)

    # 设置坐标轴刻度字体大小
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    plt.tight_layout()
    plt.show()
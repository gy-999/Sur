# visualize.p
import os
import matplotlib

matplotlib.use('Agg')  # 不显示图形，只保存
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
from skimage import measure



def plot_training_curves(train_losses, val_losses, dice_scores, model_name):
    """绘制训练和验证曲线并保存到result文件夹"""
    # 创建result文件夹
    os.makedirs('result', exist_ok=True)

    plt.figure(figsize=(15, 5))

    # 损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title(f'{model_name} - Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Dice分数曲线
    plt.subplot(1, 3, 2)
    plt.plot(dice_scores, label='Avg Dice Score', color='green')
    plt.title(f'{model_name} - Validation Dice Score')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Score')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)

    # 学习率曲线（可选）
    plt.subplot(1, 3, 3)
    # 这里可以添加学习率曲线
    plt.title(f'{model_name} - Learning Rate Schedule')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'result/training_curves_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形，不显示


def visualize_results(model, test_loader, device, num_samples=5, model_name="unet"):
    """可视化结果并保存到result文件夹，将分割掩膜叠加在原图上"""
    # 创建result文件夹
    os.makedirs('result', exist_ok=True)

    # 设置全局字体大小
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 20
    })

    # 定义医学图像颜色映射
    colors = {
        0: [0, 0, 0],  # 背景 - 黑色
        1: [255, 0, 0],  # 坏死 - 红色
        2: [0, 255, 0],  # 水肿 - 绿色
        3: [0, 0, 255]  # 增强肿瘤 - 蓝色
    }

    # 类名映射
    class_names = {
        0: "Background",
        1: "Necrosis",
        2: "Edema",
        3: "Enhancing Tumor"
    }

    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, masks, _) in enumerate(test_loader):
            inputs, masks = inputs.to(device), masks.to(device)

            if hasattr(model, 'return_attentions'):
                outputs, attentions = model(inputs, return_attentions=True)
            else:
                outputs = model(inputs)
                attentions = None

            for i in range(min(num_samples, inputs.size(0))):
                # 创建更大的图形以适应更多子图和更大的字体
                fig = plt.figure(figsize=(28, 14))

                # 获取数据
                input_img = inputs[i, 0].cpu().numpy()  # 取第一个模态
                true_mask = masks[i].cpu().numpy()
                pred_mask = torch.argmax(outputs[i], dim=0).cpu().numpy()

                # 确保掩膜是2D的
                if true_mask.ndim > 2:
                    true_mask = true_mask.squeeze()
                if pred_mask.ndim > 2:
                    pred_mask = pred_mask.squeeze()

                # 归一化输入图像
                input_normalized = (input_img - input_img.min()) / (input_img.max() - input_img.min() + 1e-8)

                # 1. 原始输入图像
                ax1 = plt.subplot(2, 5, 1)
                plt.imshow(input_img, cmap='gray')
                plt.title('Input MRI (T1)', fontsize=20, fontweight='bold', pad=20)
                plt.axis('off')

                # 2. 真实分割掩膜
                ax2 = plt.subplot(2, 5, 2)
                plt.imshow(true_mask, cmap='jet', vmin=0, vmax=3)
                plt.title('Ground Truth Mask', fontsize=20, fontweight='bold', pad=20)
                plt.axis('off')

                # 3. 预测分割掩膜
                ax3 = plt.subplot(2, 5, 3)
                plt.imshow(pred_mask, cmap='jet', vmin=0, vmax=3)
                plt.title('Predicted Mask', fontsize=20, fontweight='bold', pad=20)
                plt.axis('off')

                # 4. 真实分割叠加
                ax4 = plt.subplot(2, 5, 4)
                # 创建彩色分割掩膜
                true_colored = np.zeros((true_mask.shape[0], true_mask.shape[1], 3))
                for class_id, color in colors.items():
                    mask_area = true_mask == class_id
                    # 确保颜色是归一化的
                    normalized_color = [c / 255.0 for c in color]
                    # 为每个通道分别赋值
                    true_colored[mask_area, 0] = normalized_color[0]  # R通道
                    true_colored[mask_area, 1] = normalized_color[1]  # G通道
                    true_colored[mask_area, 2] = normalized_color[2]  # B通道

                # 创建轮廓
                for class_id in [1, 2, 3]:  # 只对肿瘤区域画轮廓
                    if np.any(true_mask == class_id):
                        contours = measure.find_contours(true_mask == class_id, 0.5)
                        for contour in contours:
                            plt.plot(contour[:, 1], contour[:, 0],
                                     color=[c / 255.0 for c in colors[class_id]],
                                     linewidth=3, linestyle='-')

                plt.imshow(input_normalized, cmap='gray')
                plt.imshow(true_colored, alpha=0.4)  # 半透明叠加
                plt.title('Ground Truth Overlay', fontsize=20, fontweight='bold', pad=20)
                plt.axis('off')

                # 5. 预测分割叠加
                ax5 = plt.subplot(2, 5, 5)
                # 创建彩色分割掩膜
                pred_colored = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3))
                for class_id, color in colors.items():
                    mask_area = pred_mask == class_id
                    # 确保颜色是归一化的
                    normalized_color = [c / 255.0 for c in color]
                    # 为每个通道分别赋值
                    pred_colored[mask_area, 0] = normalized_color[0]  # R通道
                    pred_colored[mask_area, 1] = normalized_color[1]  # G通道
                    pred_colored[mask_area, 2] = normalized_color[2]  # B通道

                # 创建轮廓
                for class_id in [1, 2, 3]:
                    if np.any(pred_mask == class_id):
                        contours = measure.find_contours(pred_mask == class_id, 0.5)
                        for contour in contours:
                            plt.plot(contour[:, 1], contour[:, 0],
                                     color=[c / 255.0 for c in colors[class_id]],
                                     linewidth=3, linestyle='-')

                plt.imshow(input_normalized, cmap='gray')
                plt.imshow(pred_colored, alpha=0.4)
                plt.title('Prediction Overlay', fontsize=20, fontweight='bold', pad=20)
                plt.axis('off')

                # 8-10. 注意力图（如果有）
                if attentions is not None:
                    for j, att in enumerate(attentions):
                        if j >= 3:  # 最多显示3个注意力图
                            break
                        ax = plt.subplot(2, 5, 8 + j)
                        att_map = att[i].squeeze().cpu().numpy()
                        # 确保注意力图是2D的
                        if att_map.ndim > 2:
                            att_map = att_map.squeeze()
                        att_map = cv2.resize(att_map, (input_img.shape[1], input_img.shape[0]),
                                             interpolation=cv2.INTER_CUBIC)
                        att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-8)

                        plt.imshow(input_normalized, cmap='gray')
                        plt.imshow(att_map, cmap='hot', alpha=0.6)
                        plt.title(f'Attention Map {j + 1}', fontsize=20, fontweight='bold', pad=20)
                        plt.axis('off')

                # 添加图例
                legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=[c / 255.0 for c in color],
                                                 label=class_names[class_id])
                                   for class_id, color in colors.items() if class_id > 0]

                # 创建图例，使用更大的字体
                legend = plt.figlegend(handles=legend_elements, loc='lower center',
                                       ncol=3, fontsize=18, framealpha=0.9,
                                       bbox_to_anchor=(0.5, 0.02))

                # 设置图例边框
                legend.get_frame().set_edgecolor('black')
                legend.get_frame().set_linewidth(2)

                plt.tight_layout()
                plt.subplots_adjust(bottom=0.12)  # 为图例留出更多空间
                plt.savefig(f'result/medical_visualization_{model_name}_batch{batch_idx}_sample{i}.png',
                            dpi=300, bbox_inches='tight')
                plt.close()

            break  # 只可视化第一个batch

def plot_model_comparison(all_results):
    """绘制模型性能对比图并保存到result文件夹"""
    # 创建result文件夹
    os.makedirs('result', exist_ok=True)

    models = [r['model'] for r in all_results]
    gflops = [r['gflops'] for r in all_results]
    avg_dice = [r['avg_dice'] for r in all_results]
    avg_jaccard = [r['avg_jaccard'] for r in all_results]
    test_loss = [r['test_loss'] for r in all_results]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # GFLOPs comparison
    bars1 = ax1.bar(models, gflops, color='skyblue', alpha=0.7)
    ax1.set_title('Model Computational Complexity (GFLOPs)')
    ax1.set_ylabel('GFLOPs')
    ax1.tick_params(axis='x', rotation=45)
    # 在柱状图上添加数值
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.2f}', ha='center', va='bottom')

    # Dice score comparison
    bars2 = ax2.bar(models, avg_dice, color='lightgreen', alpha=0.7)
    ax2.set_title('Average Dice Coefficient')
    ax2.set_ylabel('Dice Score')
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', rotation=45)
    # 在柱状图上添加数值
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.4f}', ha='center', va='bottom')

    # Jaccard score comparison
    bars3 = ax3.bar(models, avg_jaccard, color='lightcoral', alpha=0.7)
    ax3.set_title('Average Jaccard Coefficient')
    ax3.set_ylabel('Jaccard Score')
    ax3.set_ylim(0, 1)
    ax3.tick_params(axis='x', rotation=45)
    # 在柱状图上添加数值
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.4f}', ha='center', va='bottom')

    # Test loss comparison
    bars4 = ax4.bar(models, test_loss, color='gold', alpha=0.7)
    ax4.set_title('Test Loss Comparison')
    ax4.set_ylabel('Test Loss')
    ax4.tick_params(axis='x', rotation=45)
    # 在柱状图上添加数值
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('result/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形，不显示

    # 额外绘制一个综合性能对比图
    plot_comprehensive_comparison(all_results)


def plot_comprehensive_comparison(all_results):
    """绘制综合性能对比图"""
    models = [r['model'] for r in all_results]

    # 提取各类别的Dice分数
    necrosis_dice = [r['test_metrics']['dice'][0] for r in all_results]
    edema_dice = [r['test_metrics']['dice'][1] for r in all_results]
    enhancing_dice = [r['test_metrics']['dice'][2] for r in all_results]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, necrosis_dice, width, label='Necrosis', alpha=0.7)
    bars2 = ax.bar(x, edema_dice, width, label='Edema', alpha=0.7)
    bars3 = ax.bar(x + width, enhancing_dice, width, label='Enhancing', alpha=0.7)

    ax.set_xlabel('Models')
    ax.set_ylabel('Dice Scores')
    ax.set_title('Dice Scores by Tumor Sub-region')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45)
    ax.legend()
    ax.set_ylim(0, 1)

    # 添加数值标签
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('result/dice_by_region_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_efficiency_analysis(all_results):
    """绘制效率分析图"""
    models = [r['model'] for r in all_results]
    gflops = [r['gflops'] for r in all_results]
    avg_dice = [r['avg_dice'] for r in all_results]
    params = [r['params'] for r in all_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 计算效率 vs 性能
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    scatter1 = ax1.scatter(gflops, avg_dice, s=100, c=colors, alpha=0.7)
    ax1.set_xlabel('GFLOPs')
    ax1.set_ylabel('Average Dice Score')
    ax1.set_title('Computational Efficiency vs Performance')
    ax1.grid(True, alpha=0.3)

    # 参数量 vs 性能
    scatter2 = ax2.scatter(params, avg_dice, s=100, c=colors, alpha=0.7)
    ax2.set_xlabel('Number of Parameters')
    ax2.set_ylabel('Average Dice Score')
    ax2.set_title('Model Size vs Performance')
    ax2.grid(True, alpha=0.3)

    # 添加标签
    for i, model in enumerate(models):
        ax1.annotate(model, (gflops[i], avg_dice[i]), xytext=(5, 5),
                     textcoords='offset points', fontsize=8)
        ax2.annotate(model, (params[i], avg_dice[i]), xytext=(5, 5),
                     textcoords='offset points', fontsize=8)

    plt.tight_layout()
    plt.savefig('result/efficiency_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
# train.py
# train.py
import argparse
import torch
import os
import torch.optim as optim
from tqdm import tqdm
from seg.loss import *
from seg.utils import *
from seg.config import *
from seg.visualize import *
from seg.compute import *


def train_model(model_name, pretrained_path=None, test_only=False):
    # 假设这些函数已经定义
    train_loader, val_loader, test_loader = preprocess_data()
    model = get_model(model_name, in_channels=4, out_channels=N_CLASSES).to(DEVICE)

    # 如果是仅测试模式，直接加载模型进行测试
    if test_only:
        print(f"仅测试模式: {model_name}")
        if pretrained_path is None:
            # 如果没有指定预训练路径，尝试加载默认的最佳模型
            pretrained_path = f'best_{model_name}_model.pth'

        if os.path.exists(pretrained_path):
            print(f"加载模型: {pretrained_path}")
            checkpoint = load_model_weights(model, pretrained_path)

            # 如果有保存的参数信息，使用它们
            if 'total_params' in checkpoint:
                total_params = checkpoint['total_params']
            else:
                total_params = sum(p.numel() for p in model.parameters())

            if 'gflops' in checkpoint:
                gflops = checkpoint['gflops']
            else:
                gflops = compute_gflops(model)
        else:
            print(f"错误: 找不到模型文件 {pretrained_path}")
            return None

        print(f"模型: {model_name}")
        print(f"总参数量: {total_params:,}")
        print(f"GFLOPs: {gflops:.2f}")

        # 直接进行测试
        return test_model(model, test_loader, model_name, total_params, gflops)

    # 训练模式
    start_epoch = 0
    best_val_dice = 0.0
    train_losses = []
    val_losses = []
    val_dice_scores = []

    if pretrained_path is not None and os.path.exists(pretrained_path):
        print(f"加载预训练模型: {pretrained_path}")
        checkpoint = load_model_weights(model, pretrained_path)

        # 加载训练状态（如果存在）
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        if 'best_val_dice' in checkpoint:
            best_val_dice = checkpoint['best_val_dice']
        if 'train_losses' in checkpoint:
            train_losses = checkpoint['train_losses']
        if 'val_losses' in checkpoint:
            val_losses = checkpoint['val_losses']
        if 'val_dice_scores' in checkpoint:
            val_dice_scores = checkpoint['val_dice_scores']

        print(f"从 epoch {start_epoch} 继续训练, 最佳验证 Dice: {best_val_dice:.4f}")
    else:
        if pretrained_path is not None:
            print(f"警告: 预训练路径 {pretrained_path} 不存在，从头开始训练")
        else:
            print("从头开始训练")


    # 计算模型复杂度和参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    gflops = compute_gflops(model)

    print(f"模型: {model_name}")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"GFLOPs: {gflops:.2f}")

    criterion = CombinedLoss(n_classes=N_CLASSES, weight=CLASS_WEIGHTS.to(DEVICE), alpha=0.5)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 如果从检查点恢复，加载优化器状态
    if pretrained_path is not None and os.path.exists(pretrained_path) and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("优化器状态已恢复")

    n_iter_per_epoch = len(train_loader)
    lr_schedule = cosine_scheduler(
        base_value=LEARNING_RATE,
        final_value=1e-6,
        epochs=NUM_EPOCHS,
        niter_per_ep=n_iter_per_epoch,
        warmup_epochs=5,
        start_warmup_value=1e-5
    )

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{NUM_EPOCHS}')

        for i, (inputs, masks, _) in enumerate(progress_bar):
            current_lr = lr_schedule[epoch * n_iter_per_epoch + i]
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            inputs, masks = inputs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item(), lr=current_lr)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_metrics = {
            'dice': [0.0] * 3,
            'accuracy': [0.0] * 3,
            'jaccard': [0.0] * 3,
            'specificity': [0.0] * 3,
            'sensitivity': [0.0] * 3
        }

        with torch.no_grad():
            for inputs, masks, _ in val_loader:
                inputs, masks = inputs.to(DEVICE), masks.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                # 计算多种指标
                metrics = compute_metrics(outputs, masks, N_CLASSES)
                for key in val_metrics.keys():
                    for i in range(3):
                        val_metrics[key][i] += metrics[key][i]

        # 计算平均指标
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        for key in val_metrics.keys():
            val_metrics[key] = [score / len(val_loader) for score in val_metrics[key]]

        avg_dice = sum(val_metrics['dice']) / 3
        avg_jaccard = sum(val_metrics['jaccard']) / 3

        # 记录训练过程
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dice_scores.append(avg_dice)

        print(f'Epoch {epoch + 1}/{NUM_EPOCHS}')
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        print(f'Avg Dice: {avg_dice:.4f}')

        # 保存最佳模型
        if avg_dice > best_val_dice:
            best_val_dice = avg_dice
            save_path = f'best_{model_name}_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'dice': best_val_dice,
                'model_name': model_name,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_dice_scores': val_dice_scores,
                'total_params': total_params,
                'gflops': gflops
            }, save_path)
            print(f'Saved best model to {save_path}')

    print('Training complete')

    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, val_dice_scores, model_name)

    # 测试阶段 - 加载最佳模型进行测试
    print(f'\nEvaluating {model_name} on test set...')
    best_model_path = f'best_{model_name}_model.pth'

    # 使用测试函数进行测试
    return test_model(model, test_loader, model_name, total_params, gflops, best_model_path)


def test_model(model, test_loader, model_name, total_params, gflops, model_path=None):
    """测试模型的函数"""
    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"加载模型: {model_path}")
    else:
        print("使用当前模型进行测试")

    model.eval()

    criterion = CombinedLoss(n_classes=N_CLASSES, weight=CLASS_WEIGHTS.to(DEVICE), alpha=0.5)

    test_metrics = {
        'dice': [0.0] * 3,
        'accuracy': [0.0] * 3,
        'jaccard': [0.0] * 3,
        'specificity': [0.0] * 3,
        'sensitivity': [0.0] * 3
    }
    test_loss = 0.0

    with torch.no_grad():
        for inputs, masks, _ in test_loader:
            inputs, masks = inputs.to(DEVICE), masks.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            test_loss += loss.item()

            metrics = compute_metrics(outputs, masks, N_CLASSES)
            for key in test_metrics.keys():
                for i in range(3):
                    test_metrics[key][i] += metrics[key][i]

    test_loss /= len(test_loader)
    for key in test_metrics.keys():
        test_metrics[key] = [score / len(test_loader) for score in test_metrics[key]]

    avg_test_dice = sum(test_metrics['dice']) / 3
    avg_test_jaccard = sum(test_metrics['jaccard']) / 3

    print(f'Test Loss: {test_loss:.4f}')
    print(
        f'Test Dice - Necrosis: {test_metrics["dice"][0]:.4f}, Edema: {test_metrics["dice"][1]:.4f}, Tumor: {test_metrics["dice"][2]:.4f}')
    print(
        f'Test Jaccard - Necrosis: {test_metrics["jaccard"][0]:.4f}, Edema: {test_metrics["jaccard"][1]:.4f}, Tumor: {test_metrics["jaccard"][2]:.4f}')
    print(
        f'Test Accuracy - Necrosis: {test_metrics["accuracy"][0]:.4f}, Edema: {test_metrics["accuracy"][1]:.4f}, Tumor: {test_metrics["accuracy"][2]:.4f}')
    print(
        f'Test Sensitivity - Necrosis: {test_metrics["sensitivity"][0]:.4f}, Edema: {test_metrics["sensitivity"][1]:.4f}, Tumor: {test_metrics["sensitivity"][2]:.4f}')
    print(
        f'Test Specificity - Necrosis: {test_metrics["specificity"][0]:.4f}, Edema: {test_metrics["specificity"][1]:.4f}, Tumor: {test_metrics["specificity"][2]:.4f}')
    print(f'Avg Test Dice: {avg_test_dice:.4f} | Avg Test Jaccard: {avg_test_jaccard:.4f}')

    print(f'\nVisualizing results for {model_name}...')
    visualize_results(model, test_loader, DEVICE, num_samples=5, model_name=model_name)

    # 返回完整的测试结果
    return {
        "model": model_name,
        "params": total_params,
        "gflops": gflops,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "avg_dice": avg_test_dice,
        "avg_jaccard": avg_test_jaccard
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='脑肿瘤分割模型训练')
    parser.add_argument('--models', type=str, nargs='*',
                        default=[
    "EfficientUNet_Lite"],
                        choices=['EA_Unet', 'unet', 'segnet', 'attention_unet', 'DeepLabV3', 'nnunet', "TransUNet",
                                 "CE_Net"],
                        help='选择要训练的模型架构（默认为所有模型）')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='预训练模型路径，格式为 "模型名1:路径1,模型名2:路径2"')
    parser.add_argument('--test-only', action='store_true',
                        help='仅测试模式，不进行训练')
    parser.add_argument('--test-model-path', type=str, default=None,
                        help='测试模式下的模型路径，格式为 "模型名1:路径1,模型名2:路径2"')
    parser.add_argument('--auto-resume', action='store_true', default=True,
                        help='自动使用之前保存的最佳模型继续训练（默认开启）')
    parser.add_argument('--results-file', type=str, default='result/model_results.csv',
                        help='结果保存的CSV文件路径（默认: result/model_results.csv）')
    parser.add_argument('--save-details', action='store_true', default=True,
                        help='保存详细结果到JSON文件（默认开启）')
    args = parser.parse_args()

    # 创建结果文件夹
    os.makedirs('result', exist_ok=True)

    # 解析预训练模型路径
    pretrained_dict = {}
    if args.pretrained:
        for item in args.pretrained.split(','):
            if ':' in item:
                model_name, path = item.split(':', 1)
                pretrained_dict[model_name] = path
            else:
                print(f"警告: 忽略无效的预训练参数: {item}")

    # 解析测试模型路径
    test_model_dict = {}
    if args.test_model_path:
        for item in args.test_model_path.split(','):
            if ':' in item:
                model_name, path = item.split(':', 1)
                test_model_dict[model_name] = path
            else:
                print(f"警告: 忽略无效的测试模型参数: {item}")

    # 加载历史结果
    previous_results = load_previous_results(args.results_file)
    if previous_results is not None:
        print(f"已加载 {len(previous_results)} 条历史记录")
        print("历史最佳模型:")
        best_models = previous_results.loc[previous_results.groupby('model')['avg_dice'].idxmax()]
        print(best_models[['model', 'avg_dice', 'timestamp']].to_string(index=False))

    # 用于存储每个模型的训练结果
    all_results = []

    # 循环处理每个模型
    for model_name in args.models:
        print(f"\n{'=' * 60}")
        print(f"{'测试' if args.test_only else '训练'}模型: {model_name}")
        print(f"{'=' * 60}")

        # 获取该模型的预训练路径
        pretrained_path = pretrained_dict.get(model_name, None)

        # 如果启用了自动恢复且没有显式指定预训练路径，检查是否存在之前保存的最佳模型
        if args.auto_resume and not args.test_only and pretrained_path is None:
            best_model_path = f'best_{model_name}_model.pth'
            if os.path.exists(best_model_path):
                pretrained_path = best_model_path
                print(f"检测到之前保存的模型: {best_model_path}，将自动继续训练")

        # 如果是测试模式，使用测试模型路径
        if args.test_only:
            test_path = test_model_dict.get(model_name, None)
            # 如果没有指定测试路径，尝试使用默认的最佳模型
            if test_path is None:
                test_path = f'best_{model_name}_model.pth'
                if os.path.exists(test_path):
                    print(f"使用默认的最佳模型进行测试: {test_path}")
                else:
                    print(f"警告: 找不到默认的最佳模型 {test_path}")
                    continue
            results = train_model(model_name, pretrained_path=test_path, test_only=True)
        else:
            # 训练模式
            if pretrained_path:
                print(f"使用预训练模型: {pretrained_path}")
            results = train_model(model_name, pretrained_path=pretrained_path, test_only=False)

        if results is not None:
            # 只保存测试集的平均值
            simplified_results = {
                'model': results['model'],
                'params': results['params'],
                'gflops': results['gflops'],
                'test_loss': results['test_loss'],
                'avg_dice': results['avg_dice'],
                'avg_jaccard': results['avg_jaccard'],
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            # 添加各类别的平均Dice分数
            metrics = results['test_metrics']
            simplified_results['dice_necrosis'] = metrics['dice'][0]
            simplified_results['dice_edema'] = metrics['dice'][1]
            simplified_results['dice_enhancing'] = metrics['dice'][2]

            # 保存简化结果到CSV
            save_results_to_csv(simplified_results, args.results_file)

            # 保存详细结果到JSON（可选）
            if args.save_details:
                save_detailed_results(results)

            all_results.append(results)

    # 如果没有结果，退出
    if not all_results:
        print("没有有效的测试结果")
        exit(1)

    # 打印详细的训练结果总结
    print("\n" + "=" * 100)
    print(f"{'测试' if args.test_only else '训练'}结果总结:")
    print("=" * 100)

    for result in all_results:
        print(f"\n模型: {result['model']}")
        print(f"参数量: {result['params']:,}")
        print(f"GFLOPs: {result['gflops']:.2f}")
        print(f"测试损失: {result['test_loss']:.4f}")

        metrics = result['test_metrics']
        print("测试指标:")
        print(f"  Dice      - 坏死: {metrics['dice'][0]:.4f}, 水肿: {metrics['dice'][1]:.4f}, 肿瘤: {metrics['dice'][2]:.4f}")
        print(
            f"  Jaccard   - 坏死: {metrics['jaccard'][0]:.4f}, 水肿: {metrics['jaccard'][1]:.4f}, 肿瘤: {metrics['jaccard'][2]:.4f}")
        print(
            f"  Accuracy  - 坏死: {metrics['accuracy'][0]:.4f}, 水肿: {metrics['accuracy'][1]:.4f}, 肿瘤: {metrics['accuracy'][2]:.4f}")
        print(
            f"  Sensitivity- 坏死: {metrics['sensitivity'][0]:.4f}, 水肿: {metrics['sensitivity'][1]:.4f}, 肿瘤: {metrics['sensitivity'][2]:.4f}")
        print(
            f"  Specificity- 坏死: {metrics['specificity'][0]:.4f}, 水肿: {metrics['specificity'][1]:.4f}, 肿瘤: {metrics['specificity'][2]:.4f}")
        print(f"平均Dice: {result['avg_dice']:.4f} | 平均Jaccard: {result['avg_jaccard']:.4f}")

    # 创建对比表格
    print("\n" + "=" * 120)
    print("模型性能对比表:")
    print("=" * 120)
    print(f"{'模型':<15} {'参数量':<12} {'GFLOPs':<8} {'Avg Dice':<10} {'Avg Jaccard':<12} {'Test Loss':<10}")
    print("-" * 120)

    for result in all_results:
        print(f"{result['model']:<15} {result['params']:<12,} {result['gflops']:<8.2f} "
              f"{result['avg_dice']:<10.4f} {result['avg_jaccard']:<12.4f} {result['test_loss']:<10.4f}")

    # 绘制模型对比图
    plot_model_comparison(all_results)

    # 显示历史最佳记录对比
    if previous_results is not None:
        print("\n" + "=" * 120)
        print("与历史最佳记录对比:")
        print("=" * 120)

        current_best = {}
        for result in all_results:
            current_best[result['model']] = result['avg_dice']

        historical_best = previous_results.groupby('model')['avg_dice'].max()

        print(f"{'模型':<15} {'当前Dice':<10} {'历史最佳Dice':<12} {'变化':<8}")
        print("-" * 120)
        for model in current_best.keys():
            current = current_best[model]
            historical = historical_best.get(model, 0)
            change = current - historical
            change_str = f"+{change:.4f}" if change > 0 else f"{change:.4f}"
            print(f"{model:<15} {current:<10.4f} {historical:<12.4f} {change_str:<8}")

    print(f"\n所有结果已保存到: {args.results_file}")
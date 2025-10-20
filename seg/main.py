# train.py
import torch.optim as optim
from tqdm import tqdm
from seg.loss import *
from seg.utils import *
from seg.config import *
from seg.compute import *


def train_model(model_name, pretrained_path=None, test_only=False):
    # Assume these functions are already defined
    train_loader, val_loader, test_loader = preprocess_data()
    model = get_model(model_name, in_channels=4, out_channels=N_CLASSES).to(DEVICE)

    # If test-only mode, load model and run test directly
    if test_only:
        print(f"Test-only mode: {model_name}")
        if pretrained_path is None:
            # If no pretrained path specified, try the default best model
            pretrained_path = f'best_{model_name}_model.pth'

        if os.path.exists(pretrained_path):
            print(f"Loading model: {pretrained_path}")
            checkpoint = load_model_weights(model, pretrained_path)

            # Use saved parameter info if present
            if 'total_params' in checkpoint:
                total_params = checkpoint['total_params']
            else:
                total_params = sum(p.numel() for p in model.parameters())

            if 'gflops' in checkpoint:
                gflops = checkpoint['gflops']
            else:
                gflops = compute_gflops(model)
        else:
            print(f"Error: Model file not found: {pretrained_path}")
            return None

        print(f"Model: {model_name}")
        print(f"Total parameters: {total_params:,}")
        print(f"GFLOPs: {gflops:.2f}")

        # Run test
        return test_model(model, test_loader, model_name, total_params, gflops)

    # Training mode
    start_epoch = 0
    best_val_dice = 0.0
    train_losses = []
    val_losses = []
    val_dice_scores = []

    if pretrained_path is not None and os.path.exists(pretrained_path):
        print(f"Loading pretrained model: {pretrained_path}")
        checkpoint = load_model_weights(model, pretrained_path)

        # Load training state if present
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

        print(f"Resuming from epoch {start_epoch}, best validation Dice: {best_val_dice:.4f}")
    else:
        if pretrained_path is not None:
            print(f"Warning: Pretrained path {pretrained_path} does not exist, training from scratch")
        else:
            print("Training from scratch")

    # Compute model complexity and parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    gflops = compute_gflops(model)

    print(f"Model: {model_name}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"GFLOPs: {gflops:.2f}")

    criterion = CombinedLoss(n_classes=N_CLASSES, weight=CLASS_WEIGHTS.to(DEVICE), alpha=0.5)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # If resuming from checkpoint, load optimizer state
    if pretrained_path is not None and os.path.exists(pretrained_path) and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Optimizer state restored")

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

        # Validation phase
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

                # Compute various metrics
                metrics = compute_metrics(outputs, masks, N_CLASSES)
                for key in val_metrics.keys():
                    for j in range(3):
                        val_metrics[key][j] += metrics[key][j]

        # Compute average metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        for key in val_metrics.keys():
            val_metrics[key] = [score / len(val_loader) for score in val_metrics[key]]

        avg_dice = sum(val_metrics['dice']) / 3
        avg_jaccard = sum(val_metrics['jaccard']) / 3

        # Record training progress
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dice_scores.append(avg_dice)

        print(f'Epoch {epoch + 1}/{NUM_EPOCHS}')
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        print(f'Avg Dice: {avg_dice:.4f}')

        # Save best model
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

    # Test phase - load best model and evaluate on test set
    print(f'\nEvaluating {model_name} on test set...')
    best_model_path = f'best_{model_name}_model.pth'

    # Use test function to evaluate
    return test_model(model, test_loader, model_name, total_params, gflops, best_model_path)


def test_model(model, test_loader, model_name, total_params, gflops, model_path=None):
    """Test the model on the test set"""
    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model: {model_path}")
    else:
        print("Using the current model for testing")

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
                for j in range(3):
                    test_metrics[key][j] += metrics[key][j]

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
        f'Test Sensitivity - Necrosis: {test_metrics["sensitivity"][0]:.4f}, Edema: {test_metrics["sensitivity"][1]:.4f}, Tumor: {test_metrics["sensitivity'][2]:.4f}')
    print(
        f'Test Specificity - Necrosis: {test_metrics["specificity"][0]:.4f}, Edema: {test_metrics["specificity"][1]:.4f}, Tumor: {test_metrics["specificity"][2]:.4f}')
    print(f'Avg Test Dice: {avg_test_dice:.4f} | Avg Test Jaccard: {avg_test_jaccard:.4f}')

    print(f'\nVisualizing results for {model_name}...')

    # Return full test results
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
    parser = argparse.ArgumentParser(description='Brain tumor segmentation model training')
    parser.add_argument('--models', type=str, nargs='*',
                        default=[
                            "EfficientUNet_Lite"],
                        choices=['EA_Unet', 'unet', 'segnet', 'attention_unet', 'DeepLabV3', 'nnunet', "TransUNet",
                                 "CE_Net"],
                        help='Choose model architectures to train (default is EfficientUNet_Lite)')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Pretrained model paths in the format "model1:path1,model2:path2"')
    parser.add_argument('--test-only', action='store_true',
                        help='Test-only mode, do not train')
    parser.add_argument('--test-model-path', type=str, default=None,
                        help='Model paths for test mode in the format "model1:path1,model2:path2"')
    parser.add_argument('--auto-resume', action='store_true', default=True,
                        help='Automatically resume training using previously saved best model (enabled by default)')
    parser.add_argument('--results-file', type=str, default='result/model_results.csv',
                        help='CSV file path to save results (default: result/model_results.csv)')
    parser.add_argument('--save-details', action='store_true', default=True,
                        help='Save detailed results to JSON files (enabled by default)')
    args = parser.parse_args()

    # Create result folder
    os.makedirs('result', exist_ok=True)

    # Parse pretrained model paths
    pretrained_dict = {}
    if args.pretrained:
        for item in args.pretrained.split(','):
            if ':' in item:
                model_name, path = item.split(':', 1)
                pretrained_dict[model_name] = path
            else:
                print(f"Warning: Skipping invalid pretrained argument: {item}")

    # Parse test model paths
    test_model_dict = {}
    if args.test_model_path:
        for item in args.test_model_path.split(','):
            if ':' in item:
                model_name, path = item.split(':', 1)
                test_model_dict[model_name] = path
            else:
                print(f"Warning: Skipping invalid test model argument: {item}")

    # Load historical results
    previous_results = load_previous_results(args.results_file)
    if previous_results is not None:
        print(f"Loaded {len(previous_results)} historical records")
        print("Historical best models:")
        best_models = previous_results.loc[previous_results.groupby('model')['avg_dice'].idxmax()]
        print(best_models[['model', 'avg_dice', 'timestamp']].to_string(index=False))

    # Store results for each model
    all_results = []

    # Loop over each model
    for model_name in args.models:
        print(f"\n{'=' * 60}")
        print(f"{'Testing' if args.test_only else 'Training'} model: {model_name}")
        print(f"{'=' * 60}")

        # Get pretrained path for this model
        pretrained_path = pretrained_dict.get(model_name, None)

        # If auto-resume is enabled and not test-only and no explicit pretrained, check for previously saved best model
        if args.auto_resume and not args.test_only and pretrained_path is None:
            best_model_path = f'best_{model_name}_model.pth'
            if os.path.exists(best_model_path):
                pretrained_path = best_model_path
                print(f"Detected previously saved model: {best_model_path}, will resume training automatically")

        # If test-only, use test model path
        if args.test_only:
            test_path = test_model_dict.get(model_name, None)
            # If no test path specified, try default best model
            if test_path is None:
                test_path = f'best_{model_name}_model.pth'
                if os.path.exists(test_path):
                    print(f"Using default best model for testing: {test_path}")
                else:
                    print(f"Warning: Default best model not found: {test_path}")
                    continue
            results = train_model(model_name, pretrained_path=test_path, test_only=True)
        else:
            # Training mode
            if pretrained_path:
                print(f"Using pretrained model: {pretrained_path}")
            results = train_model(model_name, pretrained_path=pretrained_path, test_only=False)

        if results is not None:
            # Save only the test set averages
            simplified_results = {
                'model': results['model'],
                'params': results['params'],
                'gflops': results['gflops'],
                'test_loss': results['test_loss'],
                'avg_dice': results['avg_dice'],
                'avg_jaccard': results['avg_jaccard'],
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            # Add per-class average Dice scores
            metrics = results['test_metrics']
            simplified_results['dice_necrosis'] = metrics['dice'][0]
            simplified_results['dice_edema'] = metrics['dice'][1]
            simplified_results['dice_enhancing'] = metrics['dice'][2]

            # Save simplified results to CSV
            save_results_to_csv(simplified_results, args.results_file)

            # Save detailed results to JSON (optional)
            if args.save_details:
                save_detailed_results(results)

            all_results.append(results)

    # If no results, exit
    if not all_results:
        print("No valid test results")
        exit(1)

    # Print detailed training summary
    print("\n" + "=" * 100)
    print(f"{'Testing' if args.test_only else 'Training'} results summary:")
    print("=" * 100)

    for result in all_results:
        print(f"\nModel: {result['model']}")
        print(f"Parameters: {result['params']:,}")
        print(f"GFLOPs: {result['gflops']:.2f}")
        print(f"Test Loss: {result['test_loss']:.4f}")

        metrics = result['test_metrics']
        print("Test metrics:")
        print(f"  Dice      - Necrosis: {metrics['dice'][0]:.4f}, Edema: {metrics['dice'][1]:.4f}, Tumor: {metrics['dice'][2]:.4f}")
        print(f"  Jaccard   - Necrosis: {metrics['jaccard'][0]:.4f}, Edema: {metrics['jaccard'][1]:.4f}, Tumor: {metrics['jaccard'][2]:.4f}")
        print(f"  Accuracy  - Necrosis: {metrics['accuracy'][0]:.4f}, Edema: {metrics['accuracy'][1]:.4f}, Tumor: {metrics['accuracy'][2]:.4f}")
        print(f"  Sensitivity- Necrosis: {metrics['sensitivity'][0]:.4f}, Edema: {metrics['sensitivity'][1]:.4f}, Tumor: {metrics['sensitivity'][2]:.4f}")
        print(f"  Specificity- Necrosis: {metrics['specificity'][0]:.4f}, Edema: {metrics['specificity'][1]:.4f}, Tumor: {metrics['specificity'][2]:.4f}")
        print(f"Avg Dice: {result['avg_dice']:.4f} | Avg Jaccard: {result['avg_jaccard']:.4f}")

    # Create comparison table
    print("\n" + "=" * 120)
    print("Model performance comparison:")
    print("=" * 120)
    print(f"{'Model':<15} {'Params':<12} {'GFLOPs':<8} {'Avg Dice':<10} {'Avg Jaccard':<12} {'Test Loss':<10}")
    print("-" * 120)

    for result in all_results:
        print(f"{result['model']:<15} {result['params']:<12,} {result['gflops']:<8.2f} "
              f"{result['avg_dice']:<10.4f} {result['avg_jaccard']:<12.4f} {result['test_loss']:<10.4f}")

    if previous_results is not None:
        print("\n" + "=" * 120)
        print("Comparison with historical best records:")
        print("=" * 120)

        current_best = {}
        for result in all_results:
            current_best[result['model']] = result['avg_dice']

        historical_best = previous_results.groupby('model')['avg_dice'].max()

        print(f"{'Model':<15} {'Current Dice':<10} {'Historical Best Dice':<12} {'Change':<8}")
        print("-" * 120)
        for model in current_best.keys():
            current = current_best[model]
            historical = historical_best.get(model, 0)
            change = current - historical
            change_str = f"+{change:.4f}" if change > 0 else f"{change:.4f}"
            print(f"{model:<15} {current:<10.4f} {historical:<12.4f} {change_str:<8}")

    print(f"\nAll results saved to: {args.results_file}")
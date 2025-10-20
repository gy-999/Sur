import torch
import torch.nn.functional as F
from thop import profile
import numpy as np


def compute_metrics(output, target, n_classes, eps=1e-5):
    """
    Compute multiple evaluation metrics for multi-class segmentation.
    Returns: dice, accuracy, jaccard, specificity, sensitivity for each class
    """
    # Obtain predictions
    pred = torch.argmax(output, dim=1)  # (B, H, W)
    target = target.squeeze(-1)  # (B, H, W)

    # Convert to one-hot encoding
    pred_onehot = F.one_hot(pred, n_classes).permute(0, 3, 1, 2).float()  # (B, C, H, W)
    target_onehot = F.one_hot(target, n_classes).permute(0, 3, 1, 2).float()  # (B, C, H, W)

    # Initialize storage for per-class metrics
    dice_scores = []
    accuracy_scores = []
    jaccard_scores = []
    specificity_scores = []
    sensitivity_scores = []

    # Compute metrics for each class (exclude background class 0)
    for class_idx in range(1, n_classes):  # start from 1 to skip background
        # Predictions and ground truth for the current class
        pred_class = pred_onehot[:, class_idx, :, :]
        target_class = target_onehot[:, class_idx, :, :]

        # Compute TP, TN, FP, FN
        tp = torch.sum(pred_class * target_class, dim=(1, 2))
        tn = torch.sum((1 - pred_class) * (1 - target_class), dim=(1, 2))
        fp = torch.sum(pred_class * (1 - target_class), dim=(1, 2))
        fn = torch.sum((1 - pred_class) * target_class, dim=(1, 2))

        # Dice coefficient
        dice = (2. * tp + eps) / (2. * tp + fp + fn + eps)
        dice_scores.append(dice.mean().item())

        # Accuracy
        accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)
        accuracy_scores.append(accuracy.mean().item())

        # Jaccard (IoU)
        jaccard = (tp + eps) / (tp + fp + fn + eps)
        jaccard_scores.append(jaccard.mean().item())

        # Specificity
        specificity = (tn + eps) / (tn + fp + eps)
        specificity_scores.append(specificity.mean().item())

        # Sensitivity (Recall)
        sensitivity = (tp + eps) / (tp + fn + eps)
        sensitivity_scores.append(sensitivity.mean().item())

    return {
        'dice': dice_scores,
        'accuracy': accuracy_scores,
        'jaccard': jaccard_scores,
        'specificity': specificity_scores,
        'sensitivity': sensitivity_scores
    }


def compute_gflops(model, input_shape=(1, 4, 128, 128)):
    """
    Compute model GFLOPs.
    """
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_shape).to(device)

    try:
        # For models with multiple outputs (e.g., EfficientUNet)
        if hasattr(model, 'return_attentions'):
            # Create a wrapper that only returns the main output
            class ModelWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, x):
                    return self.model(x)[0] if isinstance(self.model(x), tuple) else self.model(x)

            wrapped_model = ModelWrapper(model)
            macs, params = profile(wrapped_model, inputs=(dummy_input,), verbose=False)
        else:
            macs, params = profile(model, inputs=(dummy_input,), verbose=False)

        gflops = macs / 1e9  # convert to GFLOPs
        return gflops
    except Exception as e:
        print(f"Error computing GFLOPs: {e}")
        return 0.0
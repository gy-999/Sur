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
from scipy import stats
import seaborn as sns
import pandas as pd
from sklearn.utils import resample
from sksurv.metrics import concordance_index_censored

# Set required parameters
m_length = 16
BATCH_SIZE = 16
EPOCH = 20
lr = 0.003
K = 3
data_path = utils.DATA_PATH

# Define model configurations to compare (only include "All" modules and ensure all include clinical)
model_configs = {
    # Single-modality - clinical only
    'C-All': {'modalities': ['clinical'], 'modules': 'All'},

    # Dual-modality + clinical
    'EC-All': {'modalities': ['clinical', 'Edema'], 'modules': 'All'},
    'TC-All': {'modalities': ['clinical', 'Tumor'], 'modules': 'All'},
    'NC-All': {'modalities': ['clinical', 'Necrosis'], 'modules': 'All'},

    # Three-modality + clinical
    'ENC-All': {'modalities': ['clinical', 'Necrosis', 'Edema'], 'modules': 'All'},
    'TEC-All': {'modalities': ['clinical', 'Tumor', 'Edema'], 'modules': 'All'},
    'TNC-All': {'modalities': ['clinical', 'Tumor', 'Necrosis'], 'modules': 'All'},

    # All modalities
    'TENC-All': {'modalities': ['clinical', 'Tumor', 'Necrosis', 'Edema'], 'modules': 'All'},
}

# Set random seed and check CUDA
utils.setup_seed(24)
device = utils.test_gpu()

# Assume preprocess_clinical_data and data_path are defined
prepro_clin_data_X, _, prepro_clin_data_y, _ = preprocess_clinical_data(data_path['clinical'])
prepro_clin_data_X.reset_index(drop=True, inplace=True)
prepro_clin_data_y.reset_index(drop=True, inplace=True)

# Create StratifiedKFold object
train_testVal_strtfdKFold = StratifiedKFold(n_splits=5, random_state=24, shuffle=True)

# Store results for all models for statistical comparison
all_models_results = {config: {'c_index': [], 'brier_scores': [], 'avg_brier': []}
                      for config in model_configs.keys()}


def bootstrap_confidence_interval(scores, n_bootstrap=1000, confidence_level=0.95):
    """Compute bootstrap confidence interval"""
    bootstrap_scores = []
    for _ in range(n_bootstrap):
        bootstrap_sample = resample(scores, replace=True)
        bootstrap_scores.append(np.mean(bootstrap_sample))

    alpha = (1 - confidence_level) / 2
    lower = np.percentile(bootstrap_scores, 100 * alpha)
    upper = np.percentile(bootstrap_scores, 100 * (1 - alpha))
    return lower, upper
def safe_concordance_index(event_indicator, event_time, risk_scores):
    """Safely compute concordance index, handling various array shapes"""
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


# Main training and comparison loop
print("Starting comparative model training with statistical testing...")

# K-fold cross-validation loop
train_testVal_kfold = train_testVal_strtfdKFold.split(prepro_clin_data_X, prepro_clin_data_y[11])

for k, (train, test_val) in enumerate(train_testVal_kfold):
    print(f"\n=== Processing Fold {k + 1} ===")

    # Data splitting
    x_train, y_train = prepro_clin_data_X.iloc[train, :], prepro_clin_data_y.iloc[train, :][11]
    x_val, x_test, y_val, y_test = train_test_split(
        prepro_clin_data_X.iloc[test_val, :],
        prepro_clin_data_y.iloc[test_val, :][[11]],
        test_size=0.3, random_state=24,
        stratify=prepro_clin_data_y.iloc[test_val, :][[11]]
    )
    val, test = list(x_val.index), list(x_test.index)

    # Train and test each model configuration
    for config_name, config in model_configs.items():
        modalities = config['modalities']
        print(f"Training {config_name} with modalities: {modalities}...")

        try:
            # Create dataset
            mydataset = MyDataset(modalities, data_path)
            dataloaders = utils.get_dataloaders(mydataset, train, val, test, BATCH_SIZE)

            # Train model - for "All" config use full model
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

            # Test model
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

            # Compute performance metrics
            test_c_index = safe_concordance_index(all_event_test, all_time_test, all_hazard_test)

            hazard_scores_flat = np.array(all_hazard_test).flatten()
            probabilities = 1 / (1 + np.exp(-hazard_scores_flat))
            brier_scores = [(prob - event) ** 2 for prob, event in zip(probabilities, all_event_test)]
            avg_brier_score = np.mean(brier_scores)

            # Store results
            all_models_results[config_name]['c_index'].append(test_c_index)
            all_models_results[config_name]['brier_scores'].extend(brier_scores)
            all_models_results[config_name]['avg_brier'].append(avg_brier_score)

            print(f'{config_name} - Fold {k + 1} - C-index: {test_c_index:.4f}, Brier Score: {avg_brier_score:.4f}')

        except Exception as e:
            print(f"Error training {config_name}: {str(e)}")
            # Store NaN to keep result arrays consistent
            all_models_results[config_name]['c_index'].append(np.nan)
            all_models_results[config_name]['avg_brier'].append(np.nan)
            print(f"Skipping {config_name} due to error")

# ==================== Statistical analysis ====================
print("\n" + "=" * 80)
print("COMPREHENSIVE STATISTICAL ANALYSIS")
print("=" * 80)

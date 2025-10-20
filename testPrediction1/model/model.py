# model.py
import os
import torch
from testPrediction1.model.net import FlexibleNet
from torch.optim import Adam
from testPrediction1.model.loss import Loss
from testPrediction1.model.model_coach import ModelCoach


class _BaseModelWithDataLoader:
    def __init__(self, modalities, m_length, dataloaders, model_type='multimodal',
                 fusion_method='attention', device=None, **model_kwargs):
        self.data_modalities = modalities
        self.m_length = m_length
        self.dataloaders = dataloaders
        self.device = device
        self.fusion_method = fusion_method
        self.model_type = model_type
        self.model_kwargs = model_kwargs

        self._instantiate_model()
        self.model_blocks = [name for name, _ in self.model.named_children()]

    def _instantiate_model(self, move2device=True):
        print(f'Instantiate {self.model_type} model...')

        # Choose the appropriate network according to the model type
        if self.model_type == 'multimodal':
            self.model = FlexibleNet(
                modalities=self.data_modalities,
                m_length=self.m_length,
                model_type='multimodal',  # explicitly specify multimodal
                fusion_method=self.fusion_method,
                device=self.device,
                **self.model_kwargs
            )
        elif self.model_type == 'deepsurv':
            # For single-modality models, ensure only clinical data is used
            clinical_modalities = ['clinical']
            self.model = FlexibleNet(
                modalities=clinical_modalities,
                m_length=self.m_length,
                model_type='deepsurv',
                fusion_method='none',  # no fusion needed for single modality
                device=self.device,
                **self.model_kwargs
            )
        elif self.model_type == 'coxtime':
            clinical_modalities = ['clinical']
            self.model = FlexibleNet(
                modalities=clinical_modalities,
                m_length=self.m_length,
                model_type='coxtime',
                fusion_method='none',
                device=self.device,
                **self.model_kwargs
            )
        elif self.model_type == 'nmtlr':
            clinical_modalities = ['clinical']
            self.model = FlexibleNet(
                modalities=clinical_modalities,
                m_length=self.m_length,
                model_type='nmtlr',
                fusion_method='none',
                device=self.device,
                **self.model_kwargs
            )
        elif self.model_type == 'deepcoxmixtures':
            clinical_modalities = ['clinical']
            self.model = FlexibleNet(
                modalities=clinical_modalities,
                m_length=self.m_length,
                model_type='deepcoxmixtures',
                fusion_method='none',
                device=self.device,
                **self.model_kwargs
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        if move2device:
            self.model = self.model.to(self.device)


class Model(_BaseModelWithDataLoader):
    def __init__(self, modalities, m_length, dataloaders, model_type='multimodal',
                 fusion_method='attention', trade_off=0.3, mode='total', device=None, **model_kwargs):
        # Adjust mode based on model type
        if model_type != 'multimodal' and mode == 'total':
            # For non-multimodal models, default to using only the Cox loss
            mode = 'only_cox'
            print(f"Notice: For {model_type} model, mode was automatically set to 'only_cox'")

        super().__init__(modalities, m_length, dataloaders, model_type, fusion_method, device, **model_kwargs)

        self.optimizer = Adam
        self.loss = Loss(trade_off=trade_off, mode=mode, model_type=model_type)
        self.model_type = model_type

    def fit(self, num_epochs, lr, info_freq, log_dir, lr_factor=0.1, scheduler_patience=5, weight_decay=1e-5):
        self._instantiate_model()
        optimizer = self.optimizer(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, mode='max', factor=lr_factor,
            patience=scheduler_patience, verbose=True, threshold=1e-3,
            threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8)

        model_coach = ModelCoach(
            model=self.model, modalities=self.data_modalities,
            dataloaders=self.dataloaders, optimizer=optimizer,
            criterion=self.loss, device=self.device, model_type=self.model_type
        )

        model_coach.train(num_epochs, scheduler, info_freq, log_dir)

        self.model = model_coach.model
        self.best_model_weights = model_coach.best_wts
        self.best_c_index = model_coach.best_perf
        self.current_c_index = model_coach.current_perf

    def save_weights(self, saved_epoch, prefix, weight_dir):
        print('Saving model weights to file:')
        if saved_epoch == 'current':
            epoch = list(self.current_concord.keys())[0]
            value = self.current_concord[epoch]
            file_name = os.path.join(
                weight_dir,
                f'{prefix}_{epoch}_c_index{value:.2f}.pth')
        else:
            file_name = os.path.join(
                weight_dir,
                f'{prefix}_{saved_epoch}_' + \
                f'c_index{self.best_concord_values[saved_epoch]:.2f}.pth')
            self.model.load_state_dict(self.best_model_weights[saved_epoch])

        torch.save(self.model.state_dict(), file_name)
        print(' ', file_name)

    def test(self):
        if hasattr(self, 'best_model_weights') and 'best_wts' in self.best_model_weights:
            self.model.load_state_dict(self.best_model_weights['best_wts'])
        self.model = self.model.to(self.device)

    def load_weights(self, path):
        print('Load model weights:')
        print(path)
        self.model.load_state_dict(torch.load(path))
        self.model = self.model.to(self.device)

    def predict(self, data, data_label):
        # For single-modality models, ensure only clinical data is passed
        if self.model_type != 'multimodal':
            # Keep only clinical data
            clinical_data = {}
            if 'clinical_categorical' in data:
                clinical_data['clinical_categorical'] = data['clinical_categorical']
            if 'clinical_continuous' in data:
                clinical_data['clinical_continuous'] = data['clinical_continuous']
            data = clinical_data

        for modality in data:
            data[modality] = data[modality].to(self.device)
        event = data_label['label'][:, 0].to(self.device)
        time = data_label['label'][:, 1].to(self.device)
        return self.model(data), event, time

    def get_trainable_parameters_count(self):
        count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return count


# Convenience classes for specific model types
class MultimodalModel(Model):
    """Multimodal model - uses all available modalities"""
    def __init__(self, modalities, m_length, dataloaders, fusion_method='attention',
                 trade_off=0.3, mode='total', device=None, **model_kwargs):
        super().__init__(modalities, m_length, dataloaders, 'multimodal',
                         fusion_method, trade_off, mode, device, **model_kwargs)


class DeepSurvModel(Model):
    """DeepSurv model - uses only clinical data"""
    def __init__(self, m_length, dataloaders, hidden_dims=[64, 32], dropout_rate=0.3,
                 trade_off=0.3, mode='only_cox', device=None, **model_kwargs):
        modalities = ['clinical']  # only clinical data
        super().__init__(modalities, m_length, dataloaders, 'deepsurv',
                         'none', trade_off, mode, device,
                         hidden_dims=hidden_dims, dropout_rate=dropout_rate, **model_kwargs)


class CoxTimeModel(Model):
    """CoxTime model - uses only clinical data"""
    def __init__(self, m_length, dataloaders, time_bins=10, hidden_dims=[64, 32],
                 dropout_rate=0.3, trade_off=0.3, mode='only_cox', device=None, **model_kwargs):
        modalities = ['clinical']  # only clinical data
        super().__init__(modalities, m_length, dataloaders, 'coxtime',
                         'none', trade_off, mode, device,
                         time_bins=time_bins, hidden_dims=hidden_dims,
                         dropout_rate=dropout_rate, **model_kwargs)


class NMTLRModel(Model):
    """N-MTLR model - uses only clinical data"""
    def __init__(self, m_length, dataloaders, time_bins=10, hidden_dims=[64, 32],
                 dropout_rate=0.3, trade_off=0.3, mode='only_cox', device=None, **model_kwargs):
        modalities = ['clinical']  # only clinical data
        super().__init__(modalities, m_length, dataloaders, 'nmtlr',
                         'none', trade_off, mode, device,
                         time_bins=time_bins, hidden_dims=hidden_dims,
                         dropout_rate=dropout_rate, **model_kwargs)


class DeepCoxMixturesModel(Model):
    """Deep Cox Mixtures model - uses only clinical data"""
    def __init__(self, m_length, dataloaders, n_components=3, hidden_dims=[64, 32],
                 dropout_rate=0.3, trade_off=0.3, mode='only_cox', device=None, **model_kwargs):
        modalities = ['clinical']  # only clinical data
        super().__init__(modalities, m_length, dataloaders, 'deepcoxmixtures',
                         'none', trade_off, mode, device,
                         n_components=n_components, hidden_dims=hidden_dims,
                         dropout_rate=dropout_rate, **model_kwargs)
""" model_coach.py """
import copy
import torch
from lifelines.utils import concordance_index
from torch.utils.tensorboard import SummaryWriter


class ModelCoach:
    def __init__(self, model, modalities, dataloaders, optimizer, criterion, device=None, model_type='multimodal'):
        self.model = model
        self.modalities = modalities
        self.dataloaders = dataloaders
        self.optimizer = optimizer
        self.criterion = criterion.to(device)
        self.device = device
        self.model_type = model_type

        # Initialize best-performance records
        self.best_perf = {'best_score': 0.0}
        self.best_wts = {'best_wts': None}
        self.current_perf = {'epoch a': 0}

    def _data2device(self, data):
        for modality in data:
            data[modality] = data[modality].to(self.device)
        return data

    def _compute_loss(self, representation, modalities, pred_hazard, event, time, **kwargs):
        """Compute loss; pass extra arguments depending on model type."""
        if self.model_type == 'nmtlr':
            # For N-MTLR, pass time bin information
            time_bins = getattr(self.model, 'time_bins', None)
            return self.criterion(
                representation=representation,
                modalities=modalities,
                pred_hazard=pred_hazard,
                event=event,
                time=time,
                time_bins=time_bins
            )
        elif self.model_type == 'deepcoxmixtures':
            # For Deep Cox Mixtures, pass weights and component hazards
            weights = kwargs.get('weights', None)
            component_hazards = kwargs.get('component_hazards', None)
            return self.criterion(
                representation=representation,
                modalities=modalities,
                pred_hazard=pred_hazard,
                event=event,
                time=time,
                weights=weights,
                component_hazards=component_hazards
            )
        else:
            # Standard call for other models
            return self.criterion(
                representation=representation,
                modalities=modalities,
                pred_hazard=pred_hazard,
                event=event,
                time=time
            )

    def _log_info(self, phase, logger, epoch, epoch_loss, epoch_c_index):
        info = {
            phase + '_loss': epoch_loss,
            phase + '_c_index': epoch_c_index
        }

        for tag, value in info.items():
            logger.add_scalar(tag, value, epoch)

    def _process_data_batch(self, data, data_label, phase):
        """
        Process a single batch (train or val).
        """
        data = self._data2device(data)
        event = data_label['label'][:, 0].to(self.device)
        time = data_label['label'][:, 1].to(self.device)

        with torch.set_grad_enabled(phase == 'train'):
            # Get different outputs depending on the model type
            if self.model_type == 'deepcoxmixtures':
                # Deep Cox Mixtures may return extra outputs
                model_output = self.model(data)
                if len(model_output) == 3:
                    hazard, representation, extra_outputs = model_output
                    weights = extra_outputs.get('weights', None)
                    component_hazards = extra_outputs.get('component_hazards', None)
                else:
                    hazard, representation = model_output
                    weights, component_hazards = None, None

                loss = self._compute_loss(
                    representation, self.modalities, hazard, event, time,
                    weights=weights, component_hazards=component_hazards
                )
            else:
                # Standard outputs for other models
                hazard, representation = self.model(data)

                # For NMTLR the loss call is handled inside _compute_loss; for other models the same
                loss = self._compute_loss(representation, self.modalities, hazard, event, time)

            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return loss, hazard['hazard'], time, event

    def _run_training_loop(self, num_epochs, scheduler, info_freq, log_dir):
        logger = SummaryWriter(log_dir)
        log_info = True

        if info_freq is not None:
            def print_header():
                sub_header = ' Epoch     Loss     Ctd     Loss     Ctd'
                print('-' * (len(sub_header) + 2))
                print('             Training        Validation')
                print('           ------------     ------------')
                print(sub_header)
                print('-' * (len(sub_header) + 2))

            print()
            print_header()

        for epoch in range(1, num_epochs + 1):
            if info_freq is None:
                print_info = False
            else:
                print_info = (epoch == 1) or (epoch % info_freq == 0)

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                running_loss = []

                if print_info or log_info:
                    running_sample_time = torch.FloatTensor().to(self.device)
                    running_sample_event = torch.LongTensor().to(self.device)
                    running_hazard = torch.FloatTensor().to(self.device)

                for data, data_label in self.dataloaders[phase]:
                    loss, hazard, time, event = self._process_data_batch(data, data_label, phase)

                    running_loss.append(loss.item())
                    running_sample_time = torch.cat((running_sample_time, time.data.float()))
                    running_sample_event = torch.cat((running_sample_event, event.long().data))
                    running_hazard = torch.cat((running_hazard, hazard.detach()))

                epoch_loss = torch.mean(torch.tensor(running_loss))

                epoch_c_index = concordance_index(
                    running_sample_time.cpu().numpy(),
                    -running_hazard.cpu().numpy(),
                    running_sample_event.cpu().numpy()
                )

                if print_info:
                    if phase == 'train':
                        message = f' {epoch}/{num_epochs}'
                    space = 10 if phase == 'train' else 27
                    message += ' ' * (space - len(message))
                    message += f'{epoch_loss:.4f}'
                    space = 19 if phase == 'train' else 36
                    message += ' ' * (space - len(message))
                    message += f'{epoch_c_index:.3f}'

                    if phase == 'val':
                        print(message)

                if log_info:
                    self._log_info(
                        phase=phase, logger=logger, epoch=epoch,
                        epoch_loss=epoch_loss, epoch_c_index=epoch_c_index
                    )

                if phase == 'val':
                    if scheduler:
                        scheduler.step(epoch_c_index)

                    # Record current performance
                    k = list(self.current_perf.keys())[0]
                    self.current_perf['epoch' + str(epoch)] = self.current_perf.pop(k)
                    self.current_perf['epoch' + str(epoch)] = epoch_c_index

                    # Record the best model so far
                    if epoch_c_index > self.best_perf['best_score']:
                        self.best_perf['best_score'] = epoch_c_index
                        self.best_wts['best_wts'] = copy.deepcopy(self.model.state_dict())

    def train(self, num_epochs, scheduler, info_freq, log_dir):
        self._run_training_loop(num_epochs, scheduler, info_freq, log_dir)
        print('>>>>> Best validation C-indices:')
        for k, v in self.best_perf.items():
            print(f'     {v} ({k})')


# Specialized coach classes for different model types
class MultimodalModelCoach(ModelCoach):
    """Coach for multimodal models"""
    def __init__(self, model, modalities, dataloaders, optimizer, criterion, device=None):
        super().__init__(model, modalities, dataloaders, optimizer, criterion, device, 'multimodal')


class DeepSurvModelCoach(ModelCoach):
    """Coach for DeepSurv models"""
    def __init__(self, model, modalities, dataloaders, optimizer, criterion, device=None):
        super().__init__(model, modalities, dataloaders, optimizer, criterion, device, 'deepsurv')


class CoxTimeModelCoach(ModelCoach):
    """Coach for CoxTime models"""
    def __init__(self, model, modalities, dataloaders, optimizer, criterion, device=None):
        super().__init__(model, modalities, dataloaders, optimizer, criterion, device, 'coxtime')


class NMTLRModelCoach(ModelCoach):
    """Coach for N-MTLR models"""
    def __init__(self, model, modalities, dataloaders, optimizer, criterion, device=None):
        super().__init__(model, modalities, dataloaders, optimizer, criterion, device, 'nmtlr')


class DeepCoxMixturesModelCoach(ModelCoach):
    """Coach for Deep Cox Mixtures models"""
    def __init__(self, model, modalities, dataloaders, optimizer, criterion, device=None):
        super().__init__(model, modalities, dataloaders, optimizer, criterion, device, 'deepcoxmixtures')



























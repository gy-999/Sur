""" Loss.py """
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import accumulate
from random import shuffle


class Loss(nn.Module):
    """Loss class supporting multiple survival analysis models."""

    def __init__(self, trade_off=0.3, mode='total', model_type='multimodal'):
        """
        Parameters
        ----------
        trade_off: float (Default:0.3)
            Balances the unsupervised loss and the Cox loss.

        mode: str (Default:'total')
            Determines which loss is used.

        model_type: str (Default:'multimodal')
            Type of model: 'multimodal', 'deepsurv', 'coxtime', 'nmtlr', 'deepcoxmixtures'
        """
        super(Loss, self).__init__()
        self.trade_off = trade_off
        self.mode = mode
        self.model_type = model_type

    def _negative_log_likelihood_loss(self, pred_hazard, event, time):
        """Negative log-likelihood loss for the Cox proportional hazards model."""
        risk = pred_hazard['hazard']
        _, idx = torch.sort(time, descending=True)
        event = event[idx]
        risk = risk[idx].squeeze()

        hazard_ratio = torch.exp(risk)
        log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0) + 1e-6)
        uncensored_likelihood = risk - log_risk
        censored_likelihood = uncensored_likelihood * event

        num_observed_events = torch.sum(event) + 1e-6
        neg_likelihood = -torch.sum(censored_likelihood) / num_observed_events

        return neg_likelihood

    def _random_match(self, batch_size):
        """
        Create two random index lists for contrastive pairing.
        Half of the batch is assigned to x1, the rest to x2. If sizes differ by one, duplicate one item.
        """
        idx = list(range(batch_size))
        split_size = int(batch_size * 0.5)
        shuffle(idx)
        x1, x2 = idx[:split_size], idx[split_size:]
        if len(x1) != len(x2):
            x1.append(x2[0])

        return x1, x2

    def _contrastive_loss1(self, x1_idx, x2_idx, representation, modalities, margin=0.2):
        """
        Contrastive loss for the single-modality case.
        """
        con_loss = 0
        # Use only modalities that actually exist in the representation dict
        available_modalities = [m for m in modalities if m in representation]
        if not available_modalities:
            return torch.tensor(0.0, device=next(iter(representation.values())).device)

        modality = available_modalities[0]
        for idx1, idx2 in zip(x1_idx, x2_idx):
            dis_x_y = torch.cosine_similarity(representation[modality][idx1],
                                              representation[modality][idx2], dim=0)
            con_loss += torch.pow(torch.clamp(margin + dis_x_y, min=0.0), 2)

        return con_loss / len(x1_idx)

    def _contrastive_loss2(self, x1_idx, x2_idx, representation, modalities, margin=0.2, alpha=0.5, beta=0.5):
        """
        Contrastive loss for the multi-modality case.
        Uses cross-modal similarities and in-modality similarities to construct a margin-based loss.
        """
        con_loss = 0
        # Use only modalities that actually exist in the representation dict
        available_modalities = [m for m in modalities if m in representation]
        if len(available_modalities) < 2:
            return self._contrastive_loss1(x1_idx, x2_idx, representation, available_modalities, margin)

        for idx1, idx2 in zip(x1_idx, x2_idx):
            dis_x_x = 0
            dis_y_y = 0
            for i in range(len(available_modalities) - 1):
                for j in range(i + 1, len(available_modalities)):
                    dis_x_x += F.cosine_similarity(representation[available_modalities[i]][idx1],
                                                   representation[available_modalities[j]][idx1], dim=0)
                    dis_y_y += F.cosine_similarity(representation[available_modalities[i]][idx2],
                                                   representation[available_modalities[j]][idx2], dim=0)

            dis_x_y = 0
            for modality in available_modalities:
                dis_x_y += F.cosine_similarity(representation[modality][idx1], representation[modality][idx2], dim=0)

            # Compute loss using a smooth approximation of max(0, ...)
            loss_term = margin + dis_x_y - alpha * dis_x_x - beta * dis_y_y
            con_loss += torch.log(1 + torch.exp(loss_term))

        return con_loss / len(x1_idx)

    def _unsupervised_similarity_loss(self, representation, modalities, t=1):
        """
        Compute unsupervised similarity loss using contrastive pairings.
        t controls how many random matchings are averaged.
        """
        k = 0
        similarity_loss = 0

        # Check which modalities actually exist in the representation dict
        available_modalities = [m for m in modalities if m in representation]

        if not available_modalities:
            return torch.tensor(0.0, device=next(iter(representation.values())).device)

        if len(available_modalities) > 1:
            while k < t:
                x1_idx, x2_idx = self._random_match(representation[available_modalities[0]].shape[0])
                similarity_loss += self._contrastive_loss2(x1_idx, x2_idx, representation, available_modalities)
                k += 1
        else:
            while k < t:
                x1_idx, x2_idx = self._random_match(representation[available_modalities[0]].shape[0])
                similarity_loss += self._contrastive_loss1(x1_idx, x2_idx, representation, available_modalities)
                k += 1

        return similarity_loss / t

    def _cross_entropy_loss(self, pred_hazard, event):
        """Compute negative log-likelihood (cross-entropy style) for discrete-score predictions."""
        return F.nll_loss(pred_hazard['score'], event)

    def forward(self, representation, modalities, pred_hazard, event, time):
        """
        When mode == 'total' use the proposed combined loss.
        When mode == 'only_cox' only use the Cox negative log-likelihood.
        """
        if self.mode == 'total':
            # Check for multimodal data to decide whether to include contrastive loss
            available_modalities = [m for m in modalities if m in representation]
            if len(available_modalities) > 1:
                # Multimodal data present: use full loss
                loss = (self._cross_entropy_loss(pred_hazard, event) +
                        self._negative_log_likelihood_loss(pred_hazard, event, time) +
                        self.trade_off * self._unsupervised_similarity_loss(representation, modalities))
            else:
                # Single-modality only: do not use contrastive loss
                loss = (self._cross_entropy_loss(pred_hazard, event) +
                        self._negative_log_likelihood_loss(pred_hazard, event, time))
        elif self.mode == 'only_cox':
            loss = self._negative_log_likelihood_loss(pred_hazard, event, time)

        return loss


# Create specialized loss classes for different model types
class MultimodalLoss(Loss):
    """Loss for multimodal models."""
    def __init__(self, trade_off=0.3, mode='total'):
        super().__init__(trade_off=trade_off, mode=mode, model_type='multimodal')


class DeepSurvLoss(Loss):
    """Loss for DeepSurv models."""
    def __init__(self):
        super().__init__(model_type='deepsurv')


class CoxTimeLoss(Loss):
    """Loss for CoxTime models."""
    def __init__(self):
        super().__init__(model_type='coxtime')


class NMTLRLoss(Loss):
    """Loss for N-MTLR models."""
    def __init__(self, time_bins=None):
        super().__init__(model_type='nmtlr')
        self.time_bins = time_bins

    def forward(self, representation, modalities, pred_hazard, event, time):
        # Forward kept compatible with parent; time_bins available on the instance if needed.
        return super().forward(representation, modalities, pred_hazard, event, time,
                              time_bins=self.time_bins)


class DeepCoxMixturesLoss(Loss):
    """Loss for Deep Cox Mixtures models."""
    def __init__(self):
        super().__init__(model_type='deepcoxmixtures')

    def forward(self, representation, modalities, pred_hazard, event, time,
                weights=None, component_hazards=None):
        return super().forward(representation, modalities, pred_hazard, event, time,
                              weights=weights, component_hazards=component_hazards)













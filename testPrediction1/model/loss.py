""" Loss.py"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import accumulate
from random import shuffle

""" Loss.py"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import accumulate
from random import shuffle


class Loss(nn.Module):
    """损失函数类，支持多种生存分析模型"""

    def __init__(self, trade_off=0.3, mode='total', model_type='multimodal'):
        """
        Parameters
        ----------
        trade_off: float (Default:0.3)
            To balance the unsupervised loss and cox loss.

        mode: str (Default:'total')
            To determine which loss is used.

        model_type: str (Default:'multimodal')
            Type of model: 'multimodal', 'deepsurv', 'coxtime', 'nmtlr', 'deepcoxmixtures'
        """
        super(Loss, self).__init__()
        self.trade_off = trade_off
        self.mode = mode
        self.model_type = model_type

    def _negative_log_likelihood_loss(self, pred_hazard, event, time):
        """Cox比例风险模型的负对数似然损失"""
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
        idx = list(range(batch_size))
        split_size = int(batch_size * 0.5)
        shuffle(idx)
        x1, x2 = idx[:split_size], idx[split_size:]
        if len(x1) != len(x2):
            x1.append(x2[0])

        return x1, x2

    def _contrastive_loss1(self, x1_idx, x2_idx, representation, modalities, margin=0.2):
        """
        Only one modality
        """
        con_loss = 0
        # 只使用实际存在的模态
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
        con_loss = 0
        # 只使用实际存在的模态
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

            # Calculate loss using a smooth approximation of max(0, ...)
            loss_term = margin + dis_x_y - alpha * dis_x_x - beta * dis_y_y
            con_loss += torch.log(1 + torch.exp(loss_term))

        return con_loss / len(x1_idx)

    def _unsupervised_similarity_loss(self, representation, modalities, t=1):
        k = 0
        similarity_loss = 0

        # 检查representation中实际存在的模态
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
        return F.nll_loss(pred_hazard['score'], event)

    def forward(self, representation, modalities, pred_hazard, event, time):
        """
        When mode = 'total' we use the proposed loss function,
        mode = 'only_cox' we remove the unsupervised loss.
        """
        if self.mode == 'total':
            # 检查是否有多模态数据来计算对比损失
            available_modalities = [m for m in modalities if m in representation]
            if len(available_modalities) > 1:
                # 有多模态数据，使用完整损失
                loss = (self._cross_entropy_loss(pred_hazard, event) +
                        self._negative_log_likelihood_loss(pred_hazard, event, time) +
                        self.trade_off * self._unsupervised_similarity_loss(representation, modalities))
            else:
                # 只有单模态数据，不使用对比损失
                loss = (self._cross_entropy_loss(pred_hazard, event) +
                        self._negative_log_likelihood_loss(pred_hazard, event, time))
        elif self.mode == 'only_cox':
            loss = self._negative_log_likelihood_loss(pred_hazard, event, time)

        return loss

# 为不同类型的模型创建专门的损失类
class MultimodalLoss(Loss):
    """多模态模型的损失函数"""
    def __init__(self, trade_off=0.3, mode='total'):
        super().__init__(trade_off=trade_off, mode=mode, model_type='multimodal')

class DeepSurvLoss(Loss):
    """DeepSurv模型的损失函数"""
    def __init__(self):
        super().__init__(model_type='deepsurv')

class CoxTimeLoss(Loss):
    """CoxTime模型的损失函数"""
    def __init__(self):
        super().__init__(model_type='coxtime')

class NMTLRLoss(Loss):
    """N-MTLR模型的损失函数"""
    def __init__(self, time_bins=None):
        super().__init__(model_type='nmtlr')
        self.time_bins = time_bins

    def forward(self, representation, modalities, pred_hazard, event, time):
        return super().forward(representation, modalities, pred_hazard, event, time,
                              time_bins=self.time_bins)

class DeepCoxMixturesLoss(Loss):
    """Deep Cox Mixtures模型的损失函数"""
    def __init__(self):
        super().__init__(model_type='deepcoxmixtures')

    def forward(self, representation, modalities, pred_hazard, event, time,
                weights=None, component_hazards=None):
        return super().forward(representation, modalities, pred_hazard, event, time,
                              weights=weights, component_hazards=component_hazards)
















"""sub_model: Attention-based multimodal fusion"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    def __init__(self, m_length, modalities, device=None, dropout_rate=0.2):
        super(CrossModalAttention, self).__init__()
        self.m_length = m_length
        self.data_modalities = modalities
        self.device = device
        self.dropout = nn.Dropout(dropout_rate)
        self.attention_weights = nn.ParameterDict({
            modality: nn.Parameter(torch.Tensor(m_length, m_length).to(self.device))
            for modality in self.data_modalities
        })
        self.reset_parameters()

    def reset_parameters(self):
        for modality in self.data_modalities:
            nn.init.xavier_uniform_(self.attention_weights[modality])

    def forward(self, multimodal_input):
        cross_attended_features = []
        for modality in self.data_modalities:
            modality_features = torch.zeros_like(multimodal_input[modality])
            for other_modality in self.data_modalities:
                if modality != other_modality:
                    Q = torch.matmul(multimodal_input[modality], self.attention_weights[modality])
                    K = torch.matmul(multimodal_input[other_modality], self.attention_weights[other_modality].T)
                    V = multimodal_input[other_modality]
                    attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.m_length, dtype=torch.float32))
                    attention_probs = F.softmax(attention_scores, dim=-1)
                    modality_features += torch.matmul(self.dropout(attention_probs), V)
            cross_attended_features.append(modality_features)

        return torch.stack(cross_attended_features).mean(dim=0)


class Attention(nn.Module):
	def __init__(self, m_length, modalities, device=None, dropout_rate=0.2):
		super(Attention, self).__init__()
		self.m_length = m_length
		self.data_modalities = modalities
		self.device = device


		self.pipeline = nn.ModuleDict()
		for modality in self.data_modalities:
			self.pipeline[modality] = nn.Linear(self.m_length, self.m_length, bias=False).to(self.device)


		self.cross_modality_extractor = CrossModalAttention(m_length, modalities, device, dropout_rate)


		self.gate_layer = nn.Linear(2 * m_length, m_length)


		self.dropout = nn.Dropout(dropout_rate)

	def forward(self, multimodal_input):

		attention_weight = []
		multimodal_features = []
		for modality in self.data_modalities:
			attention_weight.append(torch.tanh(self.pipeline[modality](multimodal_input[modality])))
			multimodal_features.append(multimodal_input[modality])

		attention_matrix = F.softmax(torch.stack(attention_weight), dim=0)
		f_intra = torch.sum(torch.stack(multimodal_features) * attention_matrix, dim=0)


		f_cross = self.cross_modality_extractor(multimodal_input)


		gate_input = torch.cat([f_intra, f_cross], dim=-1)
		gate = torch.sigmoid(self.gate_layer(gate_input))
		f_fused = gate * f_intra + (1 - gate) * f_cross


		# fused = self.projection(f_fused)
		fused = self.dropout(f_fused)


		stacked_features = torch.stack(multimodal_features)
		fused = self._scale_for_missing_modalities(stacked_features, fused)

		return fused

	def _scale_for_missing_modalities(self, x, out):
		# 同原代码
		batch_dim = x.shape[1]
		for i in range(batch_dim):
			patient = x[:, i, :]
			zero_dims = 0
			for modality in patient:
				if modality.sum().data == 0:
					zero_dims += 1

			if zero_dims > 0:
				scaler = zero_dims + 1
				out[i, :] = scaler * out[i, :]
		return out



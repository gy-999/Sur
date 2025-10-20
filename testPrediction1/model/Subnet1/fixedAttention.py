

import torch
import torch.nn as nn
import torch.nn.functional as F

class FixedAttention(nn.Module):
	def __init__(self, m_length, modalities, device=None):
		super(FixedAttention, self).__init__()
		self.m_length = m_length
		self.data_modalities = modalities
		self.device = device


	def forward(self, multimodal_input):

		attention_weight = tuple()
		multimodal_features = tuple()
		for modality in self.data_modalities:
			attention_weight += (torch.ones(multimodal_input[modality].shape[0], self.m_length).to(self.device),)
			multimodal_features += (multimodal_input[modality],)


		attention_matrix = F.softmax(torch.stack(attention_weight), dim=0)
		fused_vec = torch.sum(torch.stack(multimodal_features) * attention_matrix, dim=0)

		return fused_vec


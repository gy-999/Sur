# net.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from testPrediction1.model.Subnet1.clinical_enbedding import *
# from testPrediction1.model.Subnet1.ed_featurenet import Ed_FeatureNet
# from testPrediction1.model.Subnet1.Necrosisi_feature_net import Necrosis_FeatureNet
from testPrediction1.model.Subnet1.Tumor_feature_net import *
from testPrediction1.model.Subnet1.Attention import Attention
from testPrediction1.model.Subnet1.fixedAttention import FixedAttention


class MultimodalNet(nn.Module):
    """在单模态情况下完全使用DeepSurv结构的多模态网络"""

    def __init__(self, modalities, m_length, fusion_method='attention', device=None, nhead=3,
                 num_categories_list=[2, 3, 4, 4, 4, 9, 3, 2], dropout_rate=0.3,
                 clinical_hidden_dims=[64, 32]):
        super(MultimodalNet, self).__init__()
        self.data_modalities = modalities
        self.m_length = m_length
        self.device = device

        # 修复：检查 num_categories_list 是否为整数，如果是则转换为列表
        if isinstance(num_categories_list, int):
            print(f"MultimodalNet 警告: num_categories_list 应该是列表，但收到整数 {num_categories_list}，将转换为单元素列表")
            num_categories_list = [num_categories_list]

        self.num_categories_list = num_categories_list

        # 关键：严格区分单模态和多模态
        self.is_multimodal = len(modalities) > 1

        print(f"MultimodalNet - 模态: {modalities}, 是否多模态: {self.is_multimodal}")
        print(f"MultimodalNet - num_categories_list: {num_categories_list}, 长度: {len(num_categories_list)}")

        # 计算临床数据输入维度 - 现在安全使用 len()
        clinical_input_dim = len(num_categories_list) + 2
        print(f"MultimodalNet - 临床输入维度: {clinical_input_dim}")

        # 使用DeepSurv结构作为临床特征提取器
        if 'clinical' in self.data_modalities:
            self.clinical_net = DeepSurvClinicalNet(
                input_dim=clinical_input_dim,
                hidden_dims=clinical_hidden_dims,
                output_dim=m_length,
                dropout_rate=dropout_rate,
                device=self.device
            )

        # 图像模态网络
        self.image_networks = nn.ModuleDict()
        # 新增的三个胶质瘤模态
        if 'Tumor' in self.data_modalities:
            self.image_networks['Tumor'] = TumorCoreNet(m_length=self.m_length, device=self.device)
        if 'Edema' in self.data_modalities:
            self.image_networks['Edema'] = EnhancingTumorNet(m_length=self.m_length, device=self.device)
        if 'Necrosis' in self.data_modalities:
            self.image_networks['Necrosis'] = WholeTumorNet(m_length=self.m_length, device=self.device)

        # 只有在真正多模态时才使用融合
        if self.is_multimodal:
            if fusion_method == 'attention':
                self.fusion = Attention(m_length=self.m_length, modalities=self.data_modalities, device=self.device)
            else:
                self.fusion = FixedAttention(m_length=self.m_length, modalities=self.data_modalities,
                                             device=self.device)

            fusion_output_dim = m_length
        else:
            fusion_output_dim = m_length

        # 输出层
        self.dropout = nn.Dropout(dropout_rate)
        self.hazard_layer1 = nn.Linear(fusion_output_dim, 1)
        self.label_layer1 = nn.Linear(fusion_output_dim, 2)

    def forward(self, x):
        representation = {}

        # 临床数据
        if 'clinical' in self.data_modalities:
            representation['clinical'] = self.clinical_net(
                x['clinical_categorical'],
                x['clinical_continuous']
            )

        # 图像数据
        for modality in self.image_networks:
            if modality in x:
                representation[modality] = self.image_networks[modality](x[modality])
            else:
                print(f"警告: 输入数据中缺少模态 {modality}")

        # 特征融合
        if self.is_multimodal:
            x_final = self.fusion(representation)
        else:
            # 修改：支持任意单模态，而不仅仅是clinical
            if len(representation) == 0:
                raise ValueError("没有可用的模态特征")

            # 获取第一个可用的模态特征
            available_modalities = list(representation.keys())
            if len(available_modalities) > 1:
                print(f"警告: 单模态模式下找到多个模态特征: {available_modalities}, 使用第一个: {available_modalities[0]}")

            x_final = representation[available_modalities[0]]

        x_final = self.dropout(x_final)
        hazard = self.hazard_layer1(x_final)
        score = F.log_softmax(self.label_layer1(x_final), dim=1)

        return {'hazard': hazard, 'score': score}, representation

class DeepSurvNet(nn.Module):
    """恢复原来的DeepSurv结构"""

    def __init__(self, input_dim, hidden_dims=[64, 32], dropout_rate=0.3, device=None):
        super(DeepSurvNet, self).__init__()
        self.device = device
        self.input_dim = input_dim

        print(f"DeepSurvNet - 输入维度: {input_dim}, 隐藏层: {hidden_dims}")

        # 恢复原来的深层网络结构
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # 输出风险分数
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

        # 为了与您的代码兼容
        self.hazard_layer1 = nn.Identity()
        self.label_layer1 = nn.Linear(1, 2)

        if device:
            self.to(device)

    def forward(self, x):
        # 处理临床数据
        clinical_cat = x['clinical_categorical'].float()
        clinical_cont = x['clinical_continuous'].float()

        # 直接拼接临床数据，不进行特殊处理
        clinical_data = torch.cat([clinical_cat, clinical_cont], dim=1)

        # 检查输入维度
        if clinical_data.shape[1] != self.input_dim:
            print(f"警告: 输入维度不匹配。期望: {self.input_dim}, 实际: {clinical_data.shape[1]}")
            # 自动调整维度
            if clinical_data.shape[1] < self.input_dim:
                padding = torch.zeros(clinical_data.shape[0], self.input_dim - clinical_data.shape[1],
                                      device=clinical_data.device)
                clinical_data = torch.cat([clinical_data, padding], dim=1)
            else:
                clinical_data = clinical_data[:, :self.input_dim]

        # 通过深度网络
        hazard = self.network(clinical_data)

        # 为了与您的代码兼容
        score = F.log_softmax(self.label_layer1(hazard), dim=1)

        return {'hazard': hazard, 'score': score}, {'clinical': hazard}

class CoxTimeNet(nn.Module):
    """CoxTime 模型 - 修正维度问题"""
    def __init__(self, input_dim, time_bins=10, hidden_dims=[64, 32], dropout_rate=0.3, device=None):
        super(CoxTimeNet, self).__init__()
        self.device = device
        self.time_bins = time_bins
        self.input_dim = input_dim

        # 构建网络层
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # 输出层 - 每个时间bin的风险
        layers.append(nn.Linear(prev_dim, time_bins))
        self.network = nn.Sequential(*layers)

        # 为了与您的代码兼容
        self.hazard_layer1 = nn.Linear(time_bins, 1)
        self.label_layer1 = nn.Linear(time_bins, 2)

        if device:
            self.to(device)

    def forward(self, x):
        # 处理临床数据
        clinical_cat = x['clinical_categorical'].float()
        clinical_cont = x['clinical_continuous'].float()

        # 拼接临床数据
        clinical_data = torch.cat([clinical_cat, clinical_cont], dim=1)

        # 检查输入维度
        if clinical_data.shape[1] != self.input_dim:
            print(f"警告: 输入维度不匹配。期望: {self.input_dim}, 实际: {clinical_data.shape[1]}")
            if clinical_data.shape[1] < self.input_dim:
                padding = torch.zeros(clinical_data.shape[0], self.input_dim - clinical_data.shape[1],
                                    device=clinical_data.device)
                clinical_data = torch.cat([clinical_data, padding], dim=1)
            else:
                clinical_data = clinical_data[:, :self.input_dim]

        # 通过深度网络 - 输出每个时间bin的风险
        time_hazards = self.network(clinical_data)

        # 聚合时间风险为单一风险分数
        hazard = self.hazard_layer1(time_hazards)

        # 为了与您的代码兼容
        score = F.log_softmax(self.label_layer1(time_hazards), dim=1)

        return {'hazard': hazard, 'score': score}, {'clinical': time_hazards}


class NMTLRNet(nn.Module):
    """N-MTLR 模型 - 修正维度问题"""
    def __init__(self, input_dim, time_bins=10, hidden_dims=[64, 32], dropout_rate=0.3, device=None):
        super(NMTLRNet, self).__init__()
        self.device = device
        self.time_bins = time_bins
        self.input_dim = input_dim

        # 构建网络层
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # 输出层 - 每个时间bin的生存概率
        layers.append(nn.Linear(prev_dim, time_bins))
        self.network = nn.Sequential(*layers)

        # 为了与您的代码兼容
        self.hazard_layer1 = nn.Linear(time_bins, 1)
        self.label_layer1 = nn.Linear(time_bins, 2)

        if device:
            self.to(device)

    def forward(self, x):
        # 处理临床数据
        clinical_cat = x['clinical_categorical'].float()
        clinical_cont = x['clinical_continuous'].float()

        # 拼接临床数据
        clinical_data = torch.cat([clinical_cat, clinical_cont], dim=1)

        # 检查输入维度
        if clinical_data.shape[1] != self.input_dim:
            print(f"警告: 输入维度不匹配。期望: {self.input_dim}, 实际: {clinical_data.shape[1]}")
            if clinical_data.shape[1] < self.input_dim:
                padding = torch.zeros(clinical_data.shape[0], self.input_dim - clinical_data.shape[1],
                                    device=clinical_data.device)
                clinical_data = torch.cat([clinical_data, padding], dim=1)
            else:
                clinical_data = clinical_data[:, :self.input_dim]

        # 通过深度网络 - 输出每个时间bin的logits
        logits = self.network(clinical_data)

        # 转换为生存概率
        survival_probs = torch.sigmoid(logits)

        # 计算风险分数 (生存概率越低，风险越高)
        hazard = -torch.log(survival_probs + 1e-8).sum(dim=1, keepdim=True)

        # 为了与您的代码兼容
        score = F.log_softmax(self.label_layer1(logits), dim=1)

        return {'hazard': hazard, 'score': score}, {'clinical': logits}


class DeepCoxMixturesNet(nn.Module):
    """Deep Cox Mixtures 模型 - 修正维度问题"""
    def __init__(self, input_dim, n_components=3, hidden_dims=[64, 32], dropout_rate=0.3, device=None):
        super(DeepCoxMixturesNet, self).__init__()
        self.device = device
        self.n_components = n_components
        self.input_dim = input_dim

        # 共享特征提取层
        shared_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            shared_layers.append(nn.Linear(prev_dim, hidden_dim))
            shared_layers.append(nn.ReLU())
            shared_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        self.shared_network = nn.Sequential(*shared_layers)

        # 组件权重网络
        self.weights_network = nn.Sequential(
            nn.Linear(prev_dim, n_components),
            nn.Softmax(dim=1)
        )

        # 每个组件的基线风险网络
        self.baseline_hazards = nn.ModuleList([
            nn.Linear(prev_dim, 1) for _ in range(n_components)
        ])

        # 为了与您的代码兼容
        self.hazard_layer1 = nn.Identity()
        self.label_layer1 = nn.Linear(1, 2)

        if device:
            self.to(device)

    def forward(self, x):
        # 处理临床数据
        clinical_cat = x['clinical_categorical'].float()
        clinical_cont = x['clinical_continuous'].float()

        # 拼接临床数据
        clinical_data = torch.cat([clinical_cat, clinical_cont], dim=1)

        # 检查输入维度
        if clinical_data.shape[1] != self.input_dim:
            print(f"警告: 输入维度不匹配。期望: {self.input_dim}, 实际: {clinical_data.shape[1]}")
            if clinical_data.shape[1] < self.input_dim:
                padding = torch.zeros(clinical_data.shape[0], self.input_dim - clinical_data.shape[1],
                                    device=clinical_data.device)
                clinical_data = torch.cat([clinical_data, padding], dim=1)
            else:
                clinical_data = clinical_data[:, :self.input_dim]

        # 通过共享网络
        shared_features = self.shared_network(clinical_data)

        # 计算组件权重
        weights = self.weights_network(shared_features)

        # 计算每个组件的风险
        component_hazards = []
        for i in range(self.n_components):
            hazard = self.baseline_hazards[i](shared_features)
            component_hazards.append(hazard)

        # 加权平均风险
        hazards_stack = torch.stack(component_hazards, dim=2)  # [batch, 1, n_components]
        weights_expanded = weights.unsqueeze(1)  # [batch, 1, n_components]
        hazard = torch.sum(hazards_stack * weights_expanded, dim=2)  # [batch, 1]

        # 为了与您的代码兼容
        hazard_adapted = self.hazard_layer1(hazard)
        score = F.log_softmax(self.label_layer1(hazard), dim=1)

        return {'hazard': hazard_adapted, 'score': score}, {'clinical': shared_features}


# 修改 FlexibleNet 类
class FlexibleNet(nn.Module):
    """灵活的神经网络，支持多模态和单模态（临床数据）"""

    def __init__(self, modalities, m_length, model_type='multimodal',
                 fusion_method='attention', device=None, nhead=3,
                 dropout_rate=0.3, **kwargs):
        super(FlexibleNet, self).__init__()
        self.data_modalities = modalities
        self.m_length = m_length
        self.device = device
        self.model_type = model_type
        self.fusion_method = fusion_method
        self.num_categories_list = [2, 3, 4, 4, 4, 9, 3, 2]

        if model_type == 'multimodal':
            # 使用原始的多模态网络
            self.net = MultimodalNet(modalities, m_length, fusion_method, device, nhead,
                           self.num_categories_list, dropout_rate)
        elif model_type == 'deepsurv':
            # 计算临床数据的输入维度
            clinical_dim = len(self.num_categories_list) + 2  # 分类变量 + 连续变量
            print(f"DeepSurv 输入维度: {clinical_dim}")
            self.net = DeepSurvNet(clinical_dim, **kwargs, device=device)
        elif model_type == 'coxtime':
            clinical_dim = len(self.num_categories_list) + 2
            print(f"CoxTime 输入维度: {clinical_dim}")
            self.net = CoxTimeNet(clinical_dim, **kwargs, device=device)
        elif model_type == 'nmtlr':
            clinical_dim = len(self.num_categories_list) + 2
            print(f"N-MTLR 输入维度: {clinical_dim}")
            self.net = NMTLRNet(clinical_dim, **kwargs, device=device)
        elif model_type == 'deepcoxmixtures':
            clinical_dim = len(self.num_categories_list) + 2
            print(f"DeepCoxMixtures 输入维度: {clinical_dim}")
            self.net = DeepCoxMixturesNet(clinical_dim, **kwargs, device=device)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

    def forward(self, x):
        return self.net(x)


# 为了向后兼容，将 Net 指向 FlexibleNet
Net = FlexibleNet
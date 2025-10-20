# net.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from testPrediction1.model.Subnet1.clinical_enbedding import *
from testPrediction1.model.Subnet1.Tumor_feature_net import *
from testPrediction1.model.Subnet1.Attention import Attention
from testPrediction1.model.Subnet1.fixedAttention import FixedAttention


class MultimodalNet(nn.Module):
    def __init__(self, modalities, m_length, fusion_method='attention', device=None, nhead=3,
                 num_categories_list=[2, 3, 4, 4, 4, 9, 3, 2], dropout_rate=0.3,
                 clinical_hidden_dims=[64, 32]):
        super(MultimodalNet, self).__init__()
        self.data_modalities = modalities
        self.m_length = m_length
        self.device = device

        # Fix: if num_categories_list is an int, convert to a list
        if isinstance(num_categories_list, int):
            print(f"MultimodalNet WARNING: num_categories_list should be a list but got int {num_categories_list}; converting to single-element list")
            num_categories_list = [num_categories_list]

        self.num_categories_list = num_categories_list

        # Important: strictly distinguish single-modality vs multi-modality
        self.is_multimodal = len(modalities) > 1

        print(f"MultimodalNet - modalities: {modalities}, is_multimodal: {self.is_multimodal}")
        print(f"MultimodalNet - num_categories_list: {num_categories_list}, length: {len(num_categories_list)}")

        # Compute clinical input dimension (safe use of len())
        clinical_input_dim = len(num_categories_list) + 2
        print(f"MultimodalNet - clinical input dim: {clinical_input_dim}")

        # Use a DeepSurv style structure as clinical feature extractor
        if 'clinical' in self.data_modalities:
            self.clinical_net = DeepSurvClinicalNet(
                input_dim=clinical_input_dim,
                hidden_dims=clinical_hidden_dims,
                output_dim=m_length,
                dropout_rate=dropout_rate,
                device=self.device
            )

        # Image modality networks
        self.image_networks = nn.ModuleDict()
        # Three glioma-related modalities
        if 'Tumor' in self.data_modalities:
            self.image_networks['Tumor'] = TumorCoreNet(m_length=self.m_length, device=self.device)
        if 'Edema' in self.data_modalities:
            self.image_networks['Edema'] = EnhancingTumorNet(m_length=self.m_length, device=self.device)
        if 'Necrosis' in self.data_modalities:
            self.image_networks['Necrosis'] = WholeTumorNet(m_length=self.m_length, device=self.device)

        # Use fusion only when truly multimodal
        if self.is_multimodal:
            if fusion_method == 'attention':
                self.fusion = Attention(m_length=self.m_length, modalities=self.data_modalities, device=self.device)
            else:
                self.fusion = FixedAttention(m_length=self.m_length, modalities=self.data_modalities,
                                             device=self.device)
            fusion_output_dim = m_length
        else:
            fusion_output_dim = m_length

        # Output layers
        self.dropout = nn.Dropout(dropout_rate)
        self.hazard_layer1 = nn.Linear(fusion_output_dim, 1)
        self.label_layer1 = nn.Linear(fusion_output_dim, 2)

    def forward(self, x):
        representation = {}

        # Clinical data
        if 'clinical' in self.data_modalities:
            representation['clinical'] = self.clinical_net(
                x['clinical_categorical'],
                x['clinical_continuous']
            )

        # Image data
        for modality in self.image_networks:
            if modality in x:
                representation[modality] = self.image_networks[modality](x[modality])
            else:
                print(f"WARNING: modality {modality} missing from input data")

        # Feature fusion
        if self.is_multimodal:
            x_final = self.fusion(representation)
        else:
            # Support any single modality, not only clinical
            if len(representation) == 0:
                raise ValueError("No available modality features")

            # Pick the first available modality feature
            available_modalities = list(representation.keys())
            if len(available_modalities) > 1:
                print(f"WARNING: Found multiple modality features in single-modality mode: {available_modalities}; using first: {available_modalities[0]}")

            x_final = representation[available_modalities[0]]

        x_final = self.dropout(x_final)
        hazard = self.hazard_layer1(x_final)
        score = F.log_softmax(self.label_layer1(x_final), dim=1)

        return {'hazard': hazard, 'score': score}, representation


class DeepSurvNet(nn.Module):
    """Restore original DeepSurv structure"""
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout_rate=0.3, device=None):
        super(DeepSurvNet, self).__init__()
        self.device = device
        self.input_dim = input_dim

        print(f"DeepSurvNet - input dim: {input_dim}, hidden_dims: {hidden_dims}")

        # Rebuild deep network
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # Output risk score
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

        # For compatibility with your code
        self.hazard_layer1 = nn.Identity()
        self.label_layer1 = nn.Linear(1, 2)

        if device:
            self.to(device)

    def forward(self, x):
        # Process clinical inputs
        clinical_cat = x['clinical_categorical'].float()
        clinical_cont = x['clinical_continuous'].float()

        # Concatenate clinical data
        clinical_data = torch.cat([clinical_cat, clinical_cont], dim=1)

        # Check input dimension
        if clinical_data.shape[1] != self.input_dim:
            print(f"WARNING: input dimension mismatch. expected: {self.input_dim}, got: {clinical_data.shape[1]}")
            # Auto-adjust dimension
            if clinical_data.shape[1] < self.input_dim:
                padding = torch.zeros(clinical_data.shape[0], self.input_dim - clinical_data.shape[1],
                                      device=clinical_data.device)
                clinical_data = torch.cat([clinical_data, padding], dim=1)
            else:
                clinical_data = clinical_data[:, :self.input_dim]

        # Forward through deep network
        hazard = self.network(clinical_data)

        # For compatibility
        score = F.log_softmax(self.label_layer1(hazard), dim=1)

        return {'hazard': hazard, 'score': score}, {'clinical': hazard}


class CoxTimeNet(nn.Module):
    """CoxTime model — fix dimension issues"""
    def __init__(self, input_dim, time_bins=10, hidden_dims=[64, 32], dropout_rate=0.3, device=None):
        super(CoxTimeNet, self).__init__()
        self.device = device
        self.time_bins = time_bins
        self.input_dim = input_dim

        # Build layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # Output per-time-bin hazards
        layers.append(nn.Linear(prev_dim, time_bins))
        self.network = nn.Sequential(*layers)

        # For compatibility
        self.hazard_layer1 = nn.Linear(time_bins, 1)
        self.label_layer1 = nn.Linear(time_bins, 2)

        if device:
            self.to(device)

    def forward(self, x):
        clinical_cat = x['clinical_categorical'].float()
        clinical_cont = x['clinical_continuous'].float()

        clinical_data = torch.cat([clinical_cat, clinical_cont], dim=1)

        if clinical_data.shape[1] != self.input_dim:
            print(f"WARNING: input dimension mismatch. expected: {self.input_dim}, got: {clinical_data.shape[1]}")
            if clinical_data.shape[1] < self.input_dim:
                padding = torch.zeros(clinical_data.shape[0], self.input_dim - clinical_data.shape[1],
                                    device=clinical_data.device)
                clinical_data = torch.cat([clinical_data, padding], dim=1)
            else:
                clinical_data = clinical_data[:, :self.input_dim]

        # Per-time-bin hazards
        time_hazards = self.network(clinical_data)

        # Aggregate time hazards to single risk score
        hazard = self.hazard_layer1(time_hazards)

        # For compatibility
        score = F.log_softmax(self.label_layer1(time_hazards), dim=1)

        return {'hazard': hazard, 'score': score}, {'clinical': time_hazards}


class NMTLRNet(nn.Module):
    """N-MTLR model — fix dimension issues"""
    def __init__(self, input_dim, time_bins=10, hidden_dims=[64, 32], dropout_rate=0.3, device=None):
        super(NMTLRNet, self).__init__()
        self.device = device
        self.time_bins = time_bins
        self.input_dim = input_dim

        # Build layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # Output logits for each time bin
        layers.append(nn.Linear(prev_dim, time_bins))
        self.network = nn.Sequential(*layers)

        # For compatibility
        self.hazard_layer1 = nn.Linear(time_bins, 1)
        self.label_layer1 = nn.Linear(time_bins, 2)

        if device:
            self.to(device)

    def forward(self, x):
        clinical_cat = x['clinical_categorical'].float()
        clinical_cont = x['clinical_continuous'].float()

        clinical_data = torch.cat([clinical_cat, clinical_cont], dim=1)

        if clinical_data.shape[1] != self.input_dim:
            print(f"WARNING: input dimension mismatch. expected: {self.input_dim}, got: {clinical_data.shape[1]}")
            if clinical_data.shape[1] < self.input_dim:
                padding = torch.zeros(clinical_data.shape[0], self.input_dim - clinical_data.shape[1],
                                    device=clinical_data.device)
                clinical_data = torch.cat([clinical_data, padding], dim=1)
            else:
                clinical_data = clinical_data[:, :self.input_dim]

        logits = self.network(clinical_data)

        # Convert to survival probabilities
        survival_probs = torch.sigmoid(logits)

        # Risk score: lower survival prob => higher risk
        hazard = -torch.log(survival_probs + 1e-8).sum(dim=1, keepdim=True)

        score = F.log_softmax(self.label_layer1(logits), dim=1)

        return {'hazard': hazard, 'score': score}, {'clinical': logits}


class DeepCoxMixturesNet(nn.Module):
    """Deep Cox Mixtures model — fix dimension issues"""
    def __init__(self, input_dim, n_components=3, hidden_dims=[64, 32], dropout_rate=0.3, device=None):
        super(DeepCoxMixturesNet, self).__init__()
        self.device = device
        self.n_components = n_components
        self.input_dim = input_dim

        # Shared feature extractor
        shared_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            shared_layers.append(nn.Linear(prev_dim, hidden_dim))
            shared_layers.append(nn.ReLU())
            shared_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        self.shared_network = nn.Sequential(*shared_layers)

        # Component weights network
        self.weights_network = nn.Sequential(
            nn.Linear(prev_dim, n_components),
            nn.Softmax(dim=1)
        )

        # Baseline hazard for each component
        self.baseline_hazards = nn.ModuleList([
            nn.Linear(prev_dim, 1) for _ in range(n_components)
        ])

        # For compatibility
        self.hazard_layer1 = nn.Identity()
        self.label_layer1 = nn.Linear(1, 2)

        if device:
            self.to(device)

    def forward(self, x):
        clinical_cat = x['clinical_categorical'].float()
        clinical_cont = x['clinical_continuous'].float()

        clinical_data = torch.cat([clinical_cat, clinical_cont], dim=1)

        if clinical_data.shape[1] != self.input_dim:
            print(f"WARNING: input dimension mismatch. expected: {self.input_dim}, got: {clinical_data.shape[1]}")
            if clinical_data.shape[1] < self.input_dim:
                padding = torch.zeros(clinical_data.shape[0], self.input_dim - clinical_data.shape[1],
                                    device=clinical_data.device)
                clinical_data = torch.cat([clinical_data, padding], dim=1)
            else:
                clinical_data = clinical_data[:, :self.input_dim]

        shared_features = self.shared_network(clinical_data)

        weights = self.weights_network(shared_features)

        component_hazards = []
        for i in range(self.n_components):
            hazard = self.baseline_hazards[i](shared_features)
            component_hazards.append(hazard)

        # Weighted average hazard
        hazards_stack = torch.stack(component_hazards, dim=2)  # [batch, 1, n_components]
        weights_expanded = weights.unsqueeze(1)  # [batch, 1, n_components]
        hazard = torch.sum(hazards_stack * weights_expanded, dim=2)  # [batch, 1]

        hazard_adapted = self.hazard_layer1(hazard)
        score = F.log_softmax(self.label_layer1(hazard), dim=1)

        return {'hazard': hazard_adapted, 'score': score}, {'clinical': shared_features}


class FlexibleNet(nn.Module):
    """Flexible network supporting multimodal and single-modality (clinical) usage"""
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
            # Use original multimodal network
            self.net = MultimodalNet(modalities, m_length, fusion_method, device, nhead,
                           self.num_categories_list, dropout_rate)
        elif model_type == 'deepsurv':
            # Compute clinical input dim
            clinical_dim = len(self.num_categories_list) + 2  # categorical + continuous
            print(f"DeepSurv input dim: {clinical_dim}")
            self.net = DeepSurvNet(clinical_dim, **kwargs, device=device)
        elif model_type == 'coxtime':
            clinical_dim = len(self.num_categories_list) + 2
            print(f"CoxTime input dim: {clinical_dim}")
            self.net = CoxTimeNet(clinical_dim, **kwargs, device=device)
        elif model_type == 'nmtlr':
            clinical_dim = len(self.num_categories_list) + 2
            print(f"N-MTLR input dim: {clinical_dim}")
            self.net = NMTLRNet(clinical_dim, **kwargs, device=device)
        elif model_type == 'deepcoxmixtures':
            clinical_dim = len(self.num_categories_list) + 2
            print(f"DeepCoxMixtures input dim: {clinical_dim}")
            self.net = DeepCoxMixturesNet(clinical_dim, **kwargs, device=device)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def forward(self, x):
        return self.net(x)


# Backwards-compatible alias
Net = FlexibleNet
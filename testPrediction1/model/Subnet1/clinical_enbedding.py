import torch
import torch.nn as nn

class DeepSurvClinicalNetFM(nn.Module):
    """
    DeepSurvClinicalNetFM: Uses a DeepSurv MLP backbone combined with a Factorization Machines (FM) module.
    Usage notes:
    - x_categorical: tensor, shape (B, D_cat)    # assumed already encoded (one-hot or numeric/embedded)
    - x_continuous: tensor, shape (B, D_cont)    # continuous features
    - FM concatenates discrete and continuous to form a raw feature vector of length D = D_cat + D_cont
    - fm_k: FM latent dimensionality (default 8)
    - Final output dimension is output_dim (aligned with original m_length)
    Design choices:
    - Compute FM second-order interactions and project them to output_dim, then add to DeepSurv MLP output (residual fusion)
    - This preserves MLP expressiveness while explicitly incorporating second-order interactions
    """

    def __init__(self, input_dim, hidden_dims=[64, 32], output_dim=16,
                 dropout_rate=0.3, device=None, fm_k=8, fm_use_bias=True):
        """
        input_dim: total clinical input feature dimension = D_cat + D_cont
        fm_k: FM latent vector dimension
        """
        super(DeepSurvClinicalNetFM, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fm_k = fm_k

        # DeepSurv MLP backbone (keeps original structure)
        layers = []
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))  # project to output_dim
        self.network = nn.Sequential(*layers)

        # FM parameters: linear term (optional) and latent matrix V
        # First-order linear term: w (shape [D]) and bias b (scalar)
        self.fm_w = nn.Parameter(torch.zeros(input_dim)) if fm_use_bias else None
        if fm_use_bias:
            self.fm_b = nn.Parameter(torch.tensor(0.0))
        else:
            self.fm_b = None

        # Initialize with small values
        self.V = nn.Parameter(torch.randn(input_dim, fm_k) * 0.01)

        # Project FM low-order interactions to output_dim so they can be fused with MLP output
        # I represent the FM second-order as a vector (dim fm_k), then map it to output_dim
        self.fm_to_output = nn.Linear(fm_k, output_dim)

        # Print configuration info (helpful for debugging)
        print(f"DeepSurvClinicalNet initialized: input_dim={input_dim}, hidden_dims={hidden_dims}, "
              f"output_dim={output_dim}, fm_k={fm_k}, fm_use_bias={fm_use_bias}")

    def fm_second_order_vector(self, x):
        """
        Compute FM second-order interactions and return a vector (B, fm_k):
        - x: (B, D)
        - V: (D, k)
        Formula (vector form):
        1/2 * ( (x @ V) ** 2 - (x**2) @ (V**2) )
        Here we do NOT sum across k; we return the k-dimensional vector, which is then mapped to output_dim.
        """
        xV = torch.matmul(x, self.V)
        x_squared = x * x
        V_squared = self.V * self.V
        xV_squared_term = torch.matmul(x_squared, V_squared)

        # second order vector: 0.5 * (xV^2 - xV_squared_term)
        second_order = 0.5 * (xV * xV - xV_squared_term)
        return second_order

    def forward(self, x_categorical, x_continuous):
        # Concatenate categorical and continuous features
        # Ensure float type
        x_cat = x_categorical.float()
        x_cont = x_continuous.float()
        x = torch.cat([x_cat, x_cont], dim=1)
        B, D = x.shape

        if D != self.input_dim:
            # Friendly error if dimensions do not match so user can check
            raise ValueError(f"Input dimension mismatch: x shape {x.shape}, expected input_dim={self.input_dim}")

        # DeepSurv backbone output
        mlp_out = self.network(x)

        # FM part
        # 1) First-order linear term (optional)
        if self.fm_w is not None:
            linear_term = torch.matmul(x, self.fm_w)
            linear_term = linear_term.unsqueeze(1)
        else:
            linear_term = None

        # 2) Second-order vector
        second_order_vec = self.fm_second_order_vector(x)

        # 3) Project second-order vector to output_dim
        fm_proj = self.fm_to_output(second_order_vec)

        # 4) If desired, project linear_term and add; typically linear_term is scalar per sample
        if linear_term is not None:
            # Project linear_term to output_dim and add (using a simple expansion for brevity)
            linear_proj = linear_term.expand(-1, self.output_dim)  # (B, output_dim)
            fm_total = fm_proj + linear_proj
        else:
            fm_total = fm_proj

        # 5) Fusion strategy: add FM projection to MLP output (residual fusion)
        features = mlp_out + fm_total  # (B, output_dim)

        # Optionally return more info for debugging
        return features


class DeepSurvClinicalNet(nn.Module):
    """Use DeepSurv architecture directly as a clinical feature extractor."""

    def __init__(self, input_dim, hidden_dims=[64, 32], output_dim=16, dropout_rate=0.3, device=None):
        super(DeepSurvClinicalNet, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim

        print(f"DeepSurvClinicalNet - input_dim: {input_dim}, output_dim: {output_dim}")

        # Replicate the DeepSurv network structure exactly
        layers = []
        prev_dim = input_dim

        # Build the same layers as DeepSurv
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
            print(f"DeepSurv layer {i + 1}: {prev_dim if i == 0 else hidden_dims[i - 1]} -> {hidden_dim}")

        # Project to the target output dimension (matching original ClinicalEmbeddingNet output_dim)
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

        print(f"DeepSurvClinicalNet final output: {prev_dim} -> {output_dim}")

    def forward(self, x_categorical, x_continuous):
        # Concatenate categorical and continuous variables
        x = torch.cat([x_categorical.float(), x_continuous.float()], dim=1)

        # Pass through the DeepSurv network
        features = self.network(x)

        return features
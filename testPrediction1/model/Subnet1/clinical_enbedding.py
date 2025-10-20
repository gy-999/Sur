

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepSurvClinicalNetFM(nn.Module):
    """
    DeepSurvClinicalNet: 使用 DeepSurv MLP 作为主干，同时加入 Factorization Machines (FM) 模块
    用法说明:
    - x_categorical: tensor, shape (B, D_cat)    # 假设已经是 one-hot 或已编码为数值/整数后经过 embedding/one-hot
    - x_continuous: tensor, shape (B, D_cont)    # 连续特征
    - FM 会把离散与连续拼接为一个长度 D = D_cat + D_cont 的原始特征向量
    - fm_k: FM 隐向量维度（默认为 8）
    - 最终输出维度为 output_dim（与原先 m_length 对齐）
    设计选择:
    - 计算 FM 的二阶交互项并投影到 output_dim，然后与 DeepSurv MLP 输出相加（残差式融合）
    - 这样既保留 MLP 表达能力，又显式加入二阶交互信息
    """

    def __init__(self, input_dim, hidden_dims=[64, 32], output_dim=16,
                 dropout_rate=0.3, device=None, fm_k=8, fm_use_bias=True):
        """
        input_dim: 整个临床输入的特征维度 = D_cat + D_cont
        fm_k: FM 隐向量维度
        """
        super(DeepSurvClinicalNetFM, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fm_k = fm_k

        # DeepSurv MLP 主干（保留原结构）
        layers = []
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))  # 投影到 output_dim
        self.network = nn.Sequential(*layers)

        # FM 参数：线性项（可选）和隐向量矩阵 V
        # FM 一阶线性项：w (shape [D]) 和偏置 b（标量）
        self.fm_w = nn.Parameter(torch.zeros(input_dim)) if fm_use_bias else None
        if fm_use_bias:
            self.fm_b = nn.Parameter(torch.tensor(0.0))
        else:
            self.fm_b = None

        # FM 隐向量矩阵 V: shape (D, fm_k)
        # 使用较小的初始值
        self.V = nn.Parameter(torch.randn(input_dim, fm_k) * 0.01)

        # 将 FM 低阶交互投影到 output_dim 以便与 MLP 输出融合
        # 我采用 FM 二阶表示为向量（维度 fm_k），然后线性映射到 output_dim
        self.fm_to_output = nn.Linear(fm_k, output_dim)

        # 记录一些信息（便于 debug）
        print(f"DeepSurvClinicalNet 初始化: input_dim={input_dim}, hidden_dims={hidden_dims}, "
              f"output_dim={output_dim}, fm_k={fm_k}, fm_use_bias={fm_use_bias}")

    def fm_second_order_vector(self, x):
        """
        计算 FM 的二阶交互，返回一个向量 (B, fm_k)：
        - x: (B, D)
        - V: (D, k)
        公式 (向量形式):
        1/2 * ( (x @ V) ** 2 - (x**2) @ (V**2) )
        这里不在最后对 k 维求和，而是返回 k 维向量，之后由线性层映射到 output_dim。
        """
        # x: (B, D)
        # V: (D, k)
        # xV = x @ V  -> (B, k)
        xV = torch.matmul(x, self.V)  # (B, k)
        x_squared = x * x             # (B, D)
        V_squared = self.V * self.V   # (D, k)
        xV_squared_term = torch.matmul(x_squared, V_squared)  # (B, k)

        # second order vector: 0.5 * (xV^2 - xV_squared_term)
        second_order = 0.5 * (xV * xV - xV_squared_term)  # (B, k)
        return second_order

    def forward(self, x_categorical, x_continuous):
        # 将分类和连续拼接成一个向量
        # 确保 float 类型
        x_cat = x_categorical.float()
        x_cont = x_continuous.float()
        x = torch.cat([x_cat, x_cont], dim=1)  # (B, D)
        B, D = x.shape

        if D != self.input_dim:
            # 友好提示：如果维度不匹配，抛出异常以便用户检查
            raise ValueError(f"输入维度不匹配: x shape {x.shape}, 期望 input_dim={self.input_dim}")

        # DeepSurv 主干输出
        mlp_out = self.network(x)  # (B, output_dim)

        # FM 部分
        # 1) 一阶线性项（可选）
        if self.fm_w is not None:
            linear_term = torch.matmul(x, self.fm_w)  # (B,)
            linear_term = linear_term.unsqueeze(1)    # (B,1)
        else:
            linear_term = None

        # 2) 二阶向量 (B, fm_k)
        second_order_vec = self.fm_second_order_vector(x)  # (B, fm_k)

        # 3) 将二阶向量投影到 output_dim
        fm_proj = self.fm_to_output(second_order_vec)  # (B, output_dim)

        # 4) 如果需要，也可以把 linear_term 投影并加入，但通常 linear_term 是标量
        if linear_term is not None:
            # 投影 linear_term 到 output_dim 并加入（使用一个线性 map）
            # 为简洁起见，把 linear term 扩展并与 fm_proj 相加
            linear_proj = linear_term.expand(-1, self.output_dim)  # (B, output_dim)
            fm_total = fm_proj + linear_proj
        else:
            fm_total = fm_proj

        # 5) 融合策略：将 FM 投影与 MLP 输出相加（残差融合）
        features = mlp_out + fm_total  # (B, output_dim)

        # optionally 可以返回更多信息用于调试
        return features
#
class DeepSurvClinicalNet(nn.Module):
    """直接使用DeepSurv结构作为临床特征提取器"""

    def __init__(self, input_dim, hidden_dims=[64, 32], output_dim=16, dropout_rate=0.3, device=None):
        super(DeepSurvClinicalNet, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim

        print(f"DeepSurvClinicalNet - 输入维度: {input_dim}, 输出维度: {output_dim}")

        # 完全复制DeepSurv的网络结构
        layers = []
        prev_dim = input_dim

        # 构建与DeepSurv完全相同的网络层
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
            print(f"DeepSurv层 {i + 1}: {prev_dim if i == 0 else hidden_dims[i - 1]} -> {hidden_dim}")

        # 投影到目标输出维度（与原来ClinicalEmbeddingNet的输出维度一致）
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

        print(f"DeepSurvClinicalNet 最终输出: {prev_dim} -> {output_dim}")

    def forward(self, x_categorical, x_continuous):
        # 拼接分类和连续变量
        x = torch.cat([x_categorical.float(), x_continuous.float()], dim=1)

        # 通过DeepSurv网络
        features = self.network(x)


        return features
#



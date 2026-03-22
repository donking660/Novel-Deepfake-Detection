"""
CLS-Guided Mixture of Experts (MoE)
CLS 引导的混合专家层

借鉴 DeMo 的 ATM (Attention Token Mixture) 设计, 核心改进:
  - DeMo 原版: query = cat(7 vectors) → 7 个向量自己决定路由权重
  - 本模块:    query = cat(CLS_token, 7 vectors) → CLS 全局信息引导路由决策

CLS token 编码了 CLIP 对整张图的全局语义理解, 用它来引导
"哪个区域组合的专家最应该被激活" 的路由决策, 比自路由更具先验信息.

数据流:
  CLS token (B, D) + 7 个特征向量 (各 B, D)
  → CLS-Guided CrossAttention → 门控权重 (B, num_heads, 1, 7)
  → Multi-Head Expert Network → 加权专家输出
  → 拼接 + 展平 → (B, 7*D)

Multi-Head 结构 (与 DeMo 一致):
  - 将 D 维特征分为 num_heads 份, 每份 head_dim
  - 每个 head 有 7 个 Expert (独立 MLP)
  - 门控权重由 CrossAttention 产生, 对 7 个 Expert 输出做加权
"""

import torch
import torch.nn as nn


class QuickGELU(nn.Module):
    """快速 GELU 激活函数 (CLIP 风格)"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class Expert(nn.Module):
    """
    单个专家: Linear → QuickGELU → BatchNorm

    Args:
        input_dim: 输入/输出维度 (= head_dim)
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            QuickGELU(),
            nn.BatchNorm1d(input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class ExpertHead(nn.Module):
    """
    单个 Head 的 7 个 Expert 集合

    对 7 个输入 chunk 分别过对应的 Expert, 然后用门控权重加权.

    Args:
        input_dim: 每个 chunk 的维度 (= head_dim)
        num_experts: Expert 数量 (= 7)
    """
    def __init__(self, input_dim: int, num_experts: int = 7):
        super().__init__()
        self.expert_list = nn.ModuleList([
            Expert(input_dim) for _ in range(num_experts)
        ])

    def forward(
        self,
        x_chunks: list,
        gate_weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_chunks: 长度为 7 的列表, 每个元素 (B, head_dim)
            gate_weight: (B, 1, 7) — 当前 head 的门控权重

        Returns:
            expert_outputs: (B, 7, head_dim) — 加权后的专家输出
        """
        # 每个 expert 处理对应的 chunk: [(B, head_dim)] * 7
        outputs = [expert(chunk) for expert, chunk in zip(self.expert_list, x_chunks)]
        # 堆叠: (B, 7, head_dim)
        outputs = torch.stack(outputs, dim=1)
        # 门控加权: gate_weight (B, 1, 7) → squeeze → (B, 7) → unsqueeze → (B, 7, 1)
        gate = gate_weight.squeeze(1).unsqueeze(2)  # (B, 7, 1)
        return outputs * gate  # (B, 7, head_dim)


class CLSGuidedCrossAttention(nn.Module):
    """
    CLS 引导的 CrossAttention 门控网络

    与 DeMo 的 CrossAttention 的区别:
      - DeMo: linear_re 输入为 cat(7 vectors) = (B, 7*D) → (B, D)
      - 本模块: linear_re 输入为 cat(CLS, 7 vectors) = (B, 8*D) → (B, D)
        CLS token 的全局信息参与 query 构造, 引导路由决策

    计算流程:
      1. query = linear_re(cat(CLS, 7 vectors)) → (B, D)
      2. key = 7 vectors stacked → (B, 7, D)
      3. Multi-head attention: q * k^T / sqrt(head_dim) → softmax → gates

    Args:
        dim: 特征维度
        num_heads: 注意力头数
    """
    def __init__(self, dim: int, num_heads: int = 16):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # CLS-Guided: 输入为 8*dim (CLS + 7 vectors 拼接)
        self.linear_re = nn.Sequential(
            nn.Linear(8 * dim, dim),
            QuickGELU(),
            nn.BatchNorm1d(dim),
        )
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        cls_token: torch.Tensor,
        x_cat: torch.Tensor,
        key_stack: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            cls_token:  (B, D) — CLS 全局特征 (引导信号)
            x_cat:      (B, 7*D) — 7 个向量的拼接
            key_stack:  (B, 7, D) — 7 个向量的堆叠

        Returns:
            gates: (B, num_heads, 1, 7) — 每个 head 对 7 个 expert 的注意力权重
        """
        B, N, C = key_stack.shape  # N=7, C=D

        # CLS-Guided query: cat(CLS, 7 vectors) → (B, 8*D) → linear → (B, D)
        query_input = torch.cat([cls_token, x_cat], dim=-1)  # (B, 8*D)
        query = self.linear_re(query_input)  # (B, D)

        # Q/K 投影 + 多头重塑
        q = self.q_proj(query)  # (B, D)
        q = q.reshape(B, 1, self.num_heads, C // self.num_heads)  # (B, 1, H, head_dim)
        q = q.permute(0, 2, 1, 3)  # (B, H, 1, head_dim)

        k = self.k_proj(key_stack)  # (B, 7, D)
        k = k.reshape(B, N, self.num_heads, C // self.num_heads)  # (B, 7, H, head_dim)
        k = k.permute(0, 2, 1, 3)  # (B, H, 7, head_dim)

        # Scaled dot-product attention → softmax → gates
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, 1, 7)
        gates = attn.softmax(dim=-1)  # (B, H, 1, 7)

        return gates


class CLSGuidedGatingNetwork(nn.Module):
    """
    CLS 引导的门控网络 (封装 CLSGuidedCrossAttention)

    Args:
        input_dim: 特征维度
        num_heads: 注意力头数
    """
    def __init__(self, input_dim: int, num_heads: int = 16):
        super().__init__()
        self.gate = CLSGuidedCrossAttention(input_dim, num_heads)

    def forward(
        self,
        cls_token: torch.Tensor,
        x_cat: torch.Tensor,
        key_stack: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns:
            gates: (B, num_heads, 1, 7)
        """
        return self.gate(cls_token, x_cat, key_stack)


class CLSGuidedMoE(nn.Module):
    """
    CLS 引导的混合专家层 (Mixture of Experts)

    整体架构 (与 DeMo MoM 一致, 增加 CLS 引导):
      1. 将 7 个向量按 head 维度切分
      2. CLS-Guided CrossAttention 生成门控权重
      3. 每个 head 的 7 个 Expert 处理对应 chunk
      4. 门控加权 → 拼接所有 head → 展平

    Args:
        feat_dim: 特征维度 (默认 1024)
        num_experts: 每个 head 的专家数量 (默认 7)
        num_heads: Multi-head 数量 (默认 16, head_dim=64)
    """
    def __init__(
        self,
        feat_dim: int = 1024,
        num_experts: int = 7,
        num_heads: int = 16,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.head_dim = feat_dim // num_heads
        self.num_experts = num_experts

        assert feat_dim % num_heads == 0, \
            f"feat_dim ({feat_dim}) must be divisible by num_heads ({num_heads})"

        # 门控网络 (CLS 引导)
        self.gating_network = CLSGuidedGatingNetwork(feat_dim, num_heads)

        # 每个 head 有一组 ExpertHead (7 个 Expert)
        self.expert_heads = nn.ModuleList([
            ExpertHead(self.head_dim, num_experts) for _ in range(num_heads)
        ])

    def forward(
        self,
        cls_token: torch.Tensor,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
        x5: torch.Tensor,
        x6: torch.Tensor,
        x7: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            cls_token: (B, D) — CLS 全局特征 (路由引导信号)
            x1 ~ x7:  各 (B, D) — HDM 产生的 7 个特征向量

        Returns:
            moe_feat: (B, 7*D) — 融合后的特征 (训练/推理均返回)
        """
        # 确保 batch 维度
        if x1.dim() == 1:
            cls_token = cls_token.unsqueeze(0)
            x1 = x1.unsqueeze(0)
            x2 = x2.unsqueeze(0)
            x3 = x3.unsqueeze(0)
            x4 = x4.unsqueeze(0)
            x5 = x5.unsqueeze(0)
            x6 = x6.unsqueeze(0)
            x7 = x7.unsqueeze(0)

        # 1. 按 head 切分每个向量
        x1_chunks = torch.chunk(x1, self.num_heads, dim=-1)
        x2_chunks = torch.chunk(x2, self.num_heads, dim=-1)
        x3_chunks = torch.chunk(x3, self.num_heads, dim=-1)
        x4_chunks = torch.chunk(x4, self.num_heads, dim=-1)
        x5_chunks = torch.chunk(x5, self.num_heads, dim=-1)
        x6_chunks = torch.chunk(x6, self.num_heads, dim=-1)
        x7_chunks = torch.chunk(x7, self.num_heads, dim=-1)

        # 每个 head 的 7 个 chunk
        head_inputs = [
            [x1_chunks[i], x2_chunks[i], x3_chunks[i], x4_chunks[i],
             x5_chunks[i], x6_chunks[i], x7_chunks[i]]
            for i in range(self.num_heads)
        ]

        # 2. CLS-Guided 门控
        x_cat = torch.cat([x1, x2, x3, x4, x5, x6, x7], dim=-1)   # (B, 7*D)
        key_stack = torch.stack([x1, x2, x3, x4, x5, x6, x7], dim=1)  # (B, 7, D)
        gate_heads = self.gating_network(cls_token, x_cat, key_stack)   # (B, H, 1, 7)

        # 3. 每个 head 过 ExpertHead + 门控加权
        expert_outputs = []
        for i, expert_head in enumerate(self.expert_heads):
            # gate_heads[:, i] → (B, 1, 7) — 第 i 个 head 的门控权重
            output = expert_head(head_inputs[i], gate_heads[:, i])  # (B, 7, head_dim)
            expert_outputs.append(output)

        # 4. 拼接所有 head: (B, 7, head_dim) * H → cat → (B, 7, D) → flatten → (B, 7*D)
        outputs = torch.cat(expert_outputs, dim=-1)  # (B, 7, D)
        outputs = outputs.flatten(start_dim=1, end_dim=-1)  # (B, 7*D)

        return outputs

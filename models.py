"""
模型定义。
包含主模型（双注意力）和消融实验用的三种变体模型。

参考:
  Zheng & Lu, "Investigating Critical Frequency Bands and Channels for
  EEG-based Emotion Recognition with Deep Neural Networks", IEEE TAMD, 2015.

本实现的修改:
  - 使用轻量级 SE-Net 风格注意力，而非原论文的 DNN 方法
  - 注意力 scale 因子初始化为 0（gradual attention）
  - 逐被试标准化 + MixUp 数据增强用于跨被试泛化
  - 修正 band-major 数据布局下的 view/transpose 错误
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EegAttentionModel(nn.Module):
    """
    脑电双注意力模型 —— 轻量设计，避免跨被试过拟合。

    架构:
      - 输入 (310,) → 两种视图: (62,5) 通道视角 + (5,62) 频段视角
      - 通道注意力: pool over bands → MLP(62→15→62) → sigmoid → 62 维权重
      - 频段注意力: pool over channels → MLP(5→10→5) → sigmoid → 5 维权重
      - 加权: feature * (1.0 + α·ch_attn + β·fr_attn), α,β 初始化为 0
      - 分类器: 310 → 64 → 32 → 3 (Dropout=0.6)
    """

    def __init__(self, n_channels=62, n_bands=5, n_classes=3, dropout=0.6):
        super().__init__()
        self.n_channels = n_channels
        self.n_bands = n_bands

        # ---- 通道注意力（轻量） ----
        self.ch_attn_fc1 = nn.Linear(n_channels, n_channels // 4)     # 62 → 15
        self.ch_attn_fc2 = nn.Linear(n_channels // 4, n_channels)    # 15 → 62

        # ---- 频段注意力（轻量） ----
        self.freq_attn_fc1 = nn.Linear(n_bands, n_bands * 2)        # 5 → 10
        self.freq_attn_fc2 = nn.Linear(n_bands * 2, n_bands)        # 10 → 5

        # 注意力缩放因子（初始化为 0，让模型从 baseline 开始学）
        self.ch_scale = nn.Parameter(torch.zeros(1))
        self.freq_scale = nn.Parameter(torch.zeros(1))

        # ---- 分类器（轻量，强 Dropout） ----
        self.classifier = nn.Sequential(
            nn.Linear(n_channels * n_bands, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )

        # 存储注意力权重（用于后续分析）
        self.channel_weights = None
        self.freq_weights = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, return_attention=False):
        """
        Args:
            x: (batch, 310) 原始 EEG 特征
            return_attention: 是否同时返回注意力权重

        Returns:
            logits: (batch, 3)
            若 return_attention: 同时返回 (channel_attn, freq_attn)
        """
        batch = x.size(0)

        # 构建两种视图（数据为 band-major: δ₀…δ₆₂, θ₀…θ₆₂, α₀…）
        x_fr_view = x.view(batch, self.n_bands, self.n_channels)   # (batch, 5, 62)
        x_ch_view = x_fr_view.transpose(1, 2).contiguous()          # (batch, 62, 5)

        # ---- 通道注意力 ----
        ch_pooled = x_ch_view.mean(dim=-1)                           # (batch, 62)
        ch_attn = torch.sigmoid(
            self.ch_attn_fc2(F.relu(self.ch_attn_fc1(ch_pooled)))
        )
        self.channel_weights = ch_attn.detach()

        # ---- 频段注意力 ----
        fr_pooled = x_fr_view.mean(dim=-1)                           # (batch, 5)
        fr_attn = torch.sigmoid(
            self.freq_attn_fc2(F.relu(self.freq_attn_fc1(fr_pooled)))
        )
        self.freq_weights = fr_attn.detach()

        # ---- 应用注意力（残差 + 缩放） ----
        ch_w = ch_attn.unsqueeze(-1)                                 # (batch, 62, 1)
        fr_w = fr_attn.unsqueeze(1)                                  # (batch, 1, 5)
        weighted = x_ch_view * (1.0 + self.ch_scale * ch_w + self.freq_scale * fr_w)

        # Flatten + 分类
        flat = weighted.reshape(batch, -1)                           # (batch, 310)
        logits = self.classifier(flat)

        if return_attention:
            return logits, ch_attn.detach(), fr_attn.detach()
        return logits


# ======================================================================
# 消融实验用模型
# ======================================================================

class BaselineMLP(nn.Module):
    """无 Attention 的纯 MLP 基线。分类器架构与主模型一致。"""

    def __init__(self, input_dim=310, n_classes=3, dropout=0.6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.BatchNorm1d(64),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.BatchNorm1d(32),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )
        self.channel_weights = None
        self.freq_weights = None

    def forward(self, x, return_attention=False):
        logits = self.net(x)
        if return_attention:
            b = x.size(0)
            return logits, torch.ones(b, 62, device=x.device), \
                   torch.ones(b, 5, device=x.device)
        return logits


class ChannelOnlyModel(nn.Module):
    """仅通道注意力（去掉频段注意力）。"""

    def __init__(self, n_channels=62, n_bands=5, n_classes=3, dropout=0.6):
        super().__init__()
        self.n_channels = n_channels
        self.n_bands = n_bands
        self.ch_fc1 = nn.Linear(n_channels, n_channels // 4)
        self.ch_fc2 = nn.Linear(n_channels // 4, n_channels)
        self.ch_scale = nn.Parameter(torch.zeros(1))
        self.classifier = nn.Sequential(
            nn.Linear(n_channels * n_bands, 64), nn.BatchNorm1d(64),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.BatchNorm1d(32),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )
        self.channel_weights = None
        self.freq_weights = None

    def forward(self, x, return_attention=False):
        b = x.size(0)
        x_fr = x.view(b, 5, 62)
        x_ch = x_fr.transpose(1, 2).contiguous()
        ch_pooled = x_ch.mean(dim=-1)
        ch_attn = torch.sigmoid(self.ch_fc2(F.relu(self.ch_fc1(ch_pooled))))
        self.channel_weights = ch_attn.detach()
        weighted = x_ch * (1.0 + self.ch_scale * ch_attn.unsqueeze(-1))
        logits = self.classifier(weighted.reshape(b, -1))
        if return_attention:
            return logits, ch_attn.detach(), torch.ones(b, 5, device=x.device)
        return logits


class FrequencyOnlyModel(nn.Module):
    """仅频段注意力（去掉通道注意力）。"""

    def __init__(self, n_channels=62, n_bands=5, n_classes=3, dropout=0.6):
        super().__init__()
        self.n_channels = n_channels
        self.n_bands = n_bands
        self.fr_fc1 = nn.Linear(n_bands, n_bands * 2)
        self.fr_fc2 = nn.Linear(n_bands * 2, n_bands)
        self.fr_scale = nn.Parameter(torch.zeros(1))
        self.classifier = nn.Sequential(
            nn.Linear(n_channels * n_bands, 64), nn.BatchNorm1d(64),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.BatchNorm1d(32),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )
        self.channel_weights = None
        self.freq_weights = None

    def forward(self, x, return_attention=False):
        b = x.size(0)
        x_fr = x.view(b, 5, 62)
        x_ch = x_fr.transpose(1, 2).contiguous()
        fr_pooled = x_fr.mean(dim=-1)
        fr_attn = torch.sigmoid(self.fr_fc2(F.relu(self.fr_fc1(fr_pooled))))
        self.freq_weights = fr_attn.detach()
        weighted = x_ch * (1.0 + self.fr_scale * fr_attn.unsqueeze(1))
        logits = self.classifier(weighted.reshape(b, -1))
        if return_attention:
            return logits, torch.ones(b, 62, device=x.device), fr_attn.detach()
        return logits

"""
实验配置文件。
包含所有超参数、路径和数据相关常量。
"""

CONFIG = {
    # ---- 数据路径 ----
    'data_dir': '../dataset/EEG/',
    'channel_locs_file': '../dataset/channel_62_pos.locs',

    # ---- 数据维度 ----
    'n_channels': 62,       # EEG 电极数
    'n_bands': 5,           # 频段数 (δ, θ, α, β, γ)
    'n_classes': 3,         # 情绪类别数 (Positive, Negative, Neutral)

    # ---- 标签名称 ----
    'band_names': ['delta (δ)', 'theta (θ)', 'alpha (α)', 'beta (β)', 'gamma (γ)'],
    'class_names': ['Positive', 'Negative', 'Neutral'],

    # ---- 训练超参数 ----
    'batch_size': 128,
    'learning_rate': 5e-4,     # AdamW 学习率
    'weight_decay': 5e-3,      # AdamW 权重衰减
    'n_epochs': 100,           # 主实验训练轮数
    'n_epochs_ablation': 80,   # 消融实验训练轮数
    'dropout': 0.6,            # Dropout 比率
    'mixup_alpha': 0.3,        # MixUp 数据增强强度

    # ---- 设备与复现 ----
    'device': 'cuda' if __import__('torch').cuda.is_available() else 'cpu',
    'seed': 42,
}

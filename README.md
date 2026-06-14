# 基于注意力机制的关键脑电电极与频段探索

基于 SEED 数据集，通过双注意力机制探索脑电情绪识别中的关键电极和频率成分。

## 项目结构

```
seed/
├── config.py          # 超参数和路径配置
├── data_loader.py     # 数据读取和预处理
├── models.py          # 模型定义（主模型 + 消融变体）
├── trainer.py         # 训练、评估、LOSO 交叉验证、消融实验
├── visualize.py       # 可视化（脑地形图、频段柱状图、混淆矩阵）
├── main.py            # 主实验入口
├── requirements.txt   # 依赖库
└── README.md          # 本文件
```

## 运行环境

- Python 3.8+
- 建议使用 GPU（CUDA），CPU 也可运行但较慢

## 依赖安装

```bash
pip install -r requirements.txt
```

## 数据准备

将 SEED 数据集放置在以下路径：

```
../dataset/
├── EEG/                      # 脑电特征 .npz 文件（命名: {subject}_{session}.npz）
├── channel_62_pos.locs       # 62 导联电极位置文件
└── subject-id-gender-seed.txt
```

## 运行命令

```bash
cd seed
python main.py
```

## 输出

运行完成后，`./figures/` 目录下生成：

| 文件 | 内容 |
|------|------|
| `loso_overview.png` | 逐被试准确率柱状图 + 总体混淆矩阵 |
| `average_attention.png` | 平均通道注意力脑地形图 + 频段注意力柱状图 |
| `per_subject_attention.png` | 12 个被试各自的通道注意力脑地形图 |
| `per_subject_freq_attention.png` | 12 个被试各自的频段注意力柱状图 |
| `loso_results.npz` | 所有数值结果（可用 `numpy.load` 加载） |

## 方法说明

### 模型架构

- **通道注意力**：对 62 个电极学习空间重要性权重
- **频段注意力**：对 5 个频段 (δ, θ, α, β, γ) 学习频谱重要性权重
- **双注意力融合**：两种注意力通过可学习 scale 因子组合，残差连接保留原始特征
- **分类器**：310 → 64 → 32 → 3（强 Dropout 正则化）

### 预处理

- 逐被试 z-score 标准化（消除个体差异，保留情绪相关模式）
- MixUp 数据增强（α=0.3）

### 评估方式

- 留一被试交叉验证（Leave-One-Subject-Out, LOSO）：12 折
- 消融实验：对比纯 MLP / 仅通道注意力 / 仅频段注意力 / 双注意力

## 参考文献

[1] Wei-Long Zheng, and Bao-Liang Lu, "Investigating Critical Frequency Bands
    and Channels for EEG-based Emotion Recognition with Deep Neural Networks,"
    IEEE Transactions on Autonomous Mental Development (IEEE TAMD),
    7(3): 162-175, 2015.

## 代码来源说明

本项目的模型架构（SE-Net 风格双注意力）和训练流程（LOSO + MixUp + 逐被试标准化）为本组独立实现。

参考的开源组件：
- PyTorch 框架 (BSD License)
- scikit-learn 的 StandardScaler 和 confusion_matrix (BSD License)
- matplotlib 可视化库 (PSF License)

## 修改说明

与原始 SEED 论文 [1] 的主要区别：
1. 使用注意力机制（而非原论文的 DNN）进行关键电极/频段探索
2. 采用 LOSO 跨被试评估（而非原论文的 subject-dependent 划分）
3. 引入逐被试标准化 + MixUp 增强解决跨被试过拟合
4. 修正了 band-major 数据布局下 view/transpose 的 reshape 错误

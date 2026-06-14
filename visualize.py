"""
可视化模块。
生成四张分析图：
  1. LOSO 概览（逐被试准确率 + 总体混淆矩阵）
  2. 平均注意力（脑地形图 + 频段柱状图）
  3. 逐被试通道注意力脑地形图
  4. 逐被试频段注意力柱状图
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
from matplotlib.colors import Normalize


# ======================================================================
# 坐标转换与子图绘制
# ======================================================================

def angle_to_cartesian(angle_deg, radius):
    """
    将电极极坐标转换为笛卡尔坐标。
    通道文件中角度定义: 0° 在顶部（12点钟方向），顺时针递增。
    """
    rad = np.radians(90 - angle_deg)
    return radius * np.cos(rad), radius * np.sin(rad)


def plot_brain_topography(channel_weights, channel_locs,
                          ax=None, title="Channel Attention", cmap='YlOrRd'):
    """
    在 2D 头皮拓扑图上绘制 62 个电极的注意力权重。

    Args:
        channel_weights: (62,) 每个电极的注意力权重
        channel_locs: list[dict] 电极位置信息 (angle, radius, name)
        ax: matplotlib Axes
        title: 图标题
        cmap: 颜色映射，高注意力 → 深色
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))

    xs, ys = zip(*[angle_to_cartesian(l['angle'], l['radius'])
                   for l in channel_locs])
    xs, ys = np.array(xs), np.array(ys)

    norm = Normalize(vmin=channel_weights.min(), vmax=channel_weights.max())
    sc = ax.scatter(xs, ys, c=channel_weights, cmap=cmap, norm=norm,
                    s=140, edgecolors='black', linewidth=0.5, zorder=5)

    # 头部轮廓
    ax.add_patch(Circle((0, 0), 0.55, fill=False, color='black', linewidth=2))
    ax.add_patch(Wedge((0, -0.55), 0.1, 75, 105, color='black', linewidth=1.5))
    ax.add_patch(Circle((-0.55, 0), 0.04, fill=False, color='black', linewidth=1.5))
    ax.add_patch(Circle((0.55, 0), 0.04, fill=False, color='black', linewidth=1.5))

    # 标注 Top-8 电极名称
    for idx in np.argsort(channel_weights)[-8:]:
        ax.annotate(channel_locs[idx]['name'], (xs[idx], ys[idx]),
                    textcoords="offset points", xytext=(4, 4),
                    fontsize=6.5, color='navy', fontweight='bold')

    plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02, label='Attention')
    ax.set(xlim=(-0.65, 0.65), ylim=(-0.65, 0.65))
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold')
    return ax


def plot_freq_attention(freq_weights, band_names, ax=None):
    """
    绘制 5 个频段注意力权重的柱状图。

    Args:
        freq_weights: (5,) 各频段注意力权重
        band_names: ['delta (δ)', 'theta (θ)', ...]
        ax: matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    bars = ax.bar(range(5), freq_weights, color=colors,
                  edgecolor='black', linewidth=1.2)
    for bar, w in zip(bars, freq_weights):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{w:.4f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    ax.set_xticks(range(5))
    ax.set_xticklabels(band_names, fontsize=11)
    ax.set_ylabel('Attention Weight', fontsize=12)
    ax.set_title('Frequency Band Attention', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(freq_weights) * 1.25 + 0.02)
    ax.grid(axis='y', alpha=0.3)
    return ax


# ======================================================================
# 综合图表生成
# ======================================================================

def save_all_figures(results, channel_locs, config, out_dir='./figures'):
    """
    生成并保存全部四张可视化图表。

    Args:
        results: run_loso() 返回的结果 dict
        channel_locs: 电极位置列表
        config: 配置字典
        out_dir: 输出目录

    Returns:
        avg_ch: (62,) 所有被试平均的通道注意力
        avg_fr: (5,)  所有被试平均的频段注意力
    """
    os.makedirs(out_dir, exist_ok=True)
    subject_ids = sorted(results['channel_attns'].keys())
    n = len(subject_ids)

    # ====== 图 1: LOSO 概览 ======
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 左: 逐被试准确率柱状图
    bar_colors = ['#2ca02c' if a >= results['avg_acc'] else '#d62728'
                  for a in results['accs']]
    bars = ax1.bar(range(n), results['accs'], color=bar_colors,
                   edgecolor='black')
    ax1.axhline(results['avg_acc'], color='blue', ls='--', lw=2,
                label=f'Mean: {results["avg_acc"]:.3f}')
    ax1.axhline(1 / 3, color='gray', ls=':', lw=1.5, label='Chance: 0.333')
    for b, a in zip(bars, results['accs']):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                 f'{a:.3f}', ha='center', fontsize=8)
    ax1.set_xticks(range(n))
    ax1.set_xticklabels([f'S{s}' for s in subject_ids], fontsize=10)
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1.05)
    ax1.set_title(f'LOSO Per-Subject Accuracy\n'
                  f'(Mean ± Std: {results["avg_acc"]:.3f} '
                  f'± {results["std_acc"]:.3f})',
                  fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3)

    # 右: 总体混淆矩阵
    cm = results['overall_cm']
    im = ax2.imshow(cm, cmap='Blues')
    for i in range(3):
        for j in range(3):
            ax2.text(j, i, str(cm[i, j]), ha='center', va='center',
                     fontsize=14, fontweight='bold',
                     color='white' if cm[i, j] > cm.max() / 2 else 'black')
    ax2.set_xticks([0, 1, 2])
    ax2.set_yticks([0, 1, 2])
    ax2.set_xticklabels(config['class_names'])
    ax2.set_yticklabels(config['class_names'])
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    ax2.set_title('Overall Confusion Matrix', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax2)

    fig1.tight_layout()
    fig1.savefig(f'{out_dir}/loso_overview.png', dpi=150, bbox_inches='tight')
    print(f"  [√] {out_dir}/loso_overview.png")

    # ====== 图 2: 平均注意力 ======
    avg_ch = np.stack(list(results['channel_attns'].values())).mean(axis=0)
    avg_fr = np.stack(list(results['freq_attns'].values())).mean(axis=0)

    fig2 = plt.figure(figsize=(18, 7))
    ax_ch = fig2.add_subplot(1, 2, 1)
    plot_brain_topography(avg_ch, channel_locs, ax=ax_ch,
                          title="Average Channel Attention (all LOSO folds)")
    ax_fr = fig2.add_subplot(1, 2, 2)
    plot_freq_attention(avg_fr, config['band_names'], ax=ax_fr)

    fig2.tight_layout()
    fig2.savefig(f'{out_dir}/average_attention.png', dpi=150, bbox_inches='tight')
    print(f"  [√] {out_dir}/average_attention.png")

    # ====== 图 3: 逐被试通道注意力脑地形图 ======
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig3, axes = plt.subplots(nrows, ncols, figsize=(20, 5 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for i, sid in enumerate(subject_ids):
        plot_brain_topography(results['channel_attns'][sid], channel_locs,
                              ax=axes[i],
                              title=f"Subject {sid} (acc={results['accs'][i]:.3f})")
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    fig3.suptitle("Per-Subject Channel Attention Topography",
                  fontsize=16, fontweight='bold', y=1.01)
    fig3.tight_layout()
    fig3.savefig(f'{out_dir}/per_subject_attention.png',
                 dpi=150, bbox_inches='tight')
    print(f"  [√] {out_dir}/per_subject_attention.png")

    # ====== 图 4: 逐被试频段注意力 ======
    ncols2 = min(4, n)
    nrows2 = (n + ncols2 - 1) // ncols2
    fig4, axes4 = plt.subplots(nrows2, ncols2,
                                figsize=(4 * ncols2, 3.5 * nrows2))
    axes4 = np.atleast_1d(axes4).flatten()
    for i, sid in enumerate(subject_ids):
        plot_freq_attention(results['freq_attns'][sid],
                            config['band_names'], ax=axes4[i])
        axes4[i].set_title(f"Subject {sid} (acc={results['accs'][i]:.3f})",
                           fontsize=11)
    for j in range(n, len(axes4)):
        axes4[j].set_visible(False)
    fig4.suptitle("Per-Subject Frequency Band Attention",
                  fontsize=15, fontweight='bold')
    fig4.tight_layout()
    fig4.savefig(f'{out_dir}/per_subject_freq_attention.png',
                 dpi=150, bbox_inches='tight')
    print(f"  [√] {out_dir}/per_subject_freq_attention.png")

    plt.close('all')
    return avg_ch, avg_fr

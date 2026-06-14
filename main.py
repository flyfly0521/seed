"""
主实验脚本。
基于注意力机制的关键脑电电极与频段探索。

用法:
    python main.py

输出:
    ./figures/ 目录下的可视化结果和数值结果
"""

import sys
import numpy as np
from config import CONFIG
from data_loader import load_channel_locs, load_all_eeg_data
from models import (EegAttentionModel, BaselineMLP,
                    ChannelOnlyModel, FrequencyOnlyModel)
from trainer import set_seed, run_loso, run_ablation
from visualize import save_all_figures


def main():
    print("=" * 55)
    print("基于注意力机制的关键脑电电极与频段探索")
    print("SEED 数据集 — 留一被试交叉验证 (LOSO)")
    print("=" * 55)
    print(f"设备: {CONFIG['device']}")

    set_seed(CONFIG['seed'])

    # ================================================================
    # Step 1: 加载数据
    # ================================================================
    print("\n[Step 1] 加载数据 ...")
    channel_locs = load_channel_locs(CONFIG['channel_locs_file'])
    print(f"  加载了 {len(channel_locs)} 个电极位置")
    subject_data, subject_ids = load_all_eeg_data(CONFIG['data_dir'])

    # ================================================================
    # Step 2: LOSO 主实验（双注意力模型）
    # ================================================================
    print("\n[Step 2] 运行双注意力模型 LOSO ...")
    main_results = run_loso(
        subject_data, subject_ids,
        lambda: EegAttentionModel(dropout=CONFIG['dropout']),
        CONFIG,
    )

    # ================================================================
    # Step 3: 可视化
    # ================================================================
    print("\n[Step 3] 生成可视化 ...")
    avg_ch, avg_fr = save_all_figures(main_results, channel_locs, CONFIG)

    # ================================================================
    # Step 4: 关键发现
    # ================================================================
    print("\n[Step 4] 关键电极与频段发现")
    top_idx = np.argsort(avg_ch)[::-1][:10]
    print("  Top-10 关键电极（平均注意力权重）:")
    for rank, idx in enumerate(top_idx, 1):
        loc = channel_locs[idx]
        print(f"    {rank:2d}. {loc['name']:>5s}  "
              f"(idx={idx:2d}, attn={avg_ch[idx]:.4f})")

    print("\n  各频段平均注意力权重:")
    for name, w in zip(CONFIG['band_names'], avg_fr):
        print(f"    {name:>15s}: {w:.4f}")

    # ================================================================
    # Step 5: 消融实验
    # ================================================================
    print("\n[Step 5] 消融实验 ...")
    ablation_factories = {
        'Baseline (Pure MLP)':
            lambda: BaselineMLP(dropout=CONFIG['dropout']),
        'Channel Attention Only':
            lambda: ChannelOnlyModel(dropout=CONFIG['dropout']),
        'Frequency Attention Only':
            lambda: FrequencyOnlyModel(dropout=CONFIG['dropout']),
        'Dual Attention':
            lambda: EegAttentionModel(dropout=CONFIG['dropout']),
    }
    ablation_results = run_ablation(subject_data, subject_ids,
                                    CONFIG, ablation_factories)

    # ================================================================
    # Step 6: 保存数值结果
    # ================================================================
    print("\n[Step 6] 保存数值结果 ...")
    np.savez('./figures/loso_results.npz',
             accs=main_results['accs'],
             avg_acc=main_results['avg_acc'],
             std_acc=main_results['std_acc'],
             avg_channel_attn=avg_ch,
             avg_freq_attn=avg_fr,
             channel_names=[l['name'] for l in channel_locs],
             band_names=CONFIG['band_names'],
    )
    print("  [√] figures/loso_results.npz")

    print(f"\n{'='*55}")
    print("全部实验完成！结果保存在 ./figures/")
    print(f"{'='*55}")


if __name__ == '__main__':
    main()

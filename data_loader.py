"""
数据读取与预处理。
从 SEED 数据集的 .npz 文件中加载 EEG 特征，按被试重新组织。
"""

import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from config import CONFIG


def load_channel_locs(filepath):
    """
    加载 62 导联电极位置信息。

    文件格式（每行）: 编号  角度  半径  电极名称

    Args:
        filepath: channel_62_pos.locs 文件路径

    Returns:
        list[dict]: 每个元素包含 idx, angle, radius, name
    """
    locs = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 4:
                locs.append({
                    'idx': int(parts[0]),
                    'angle': float(parts[1]),
                    'radius': float(parts[2]),
                    'name': parts[3],
                })
    locs.sort(key=lambda x: x['idx'])
    return locs


def load_all_eeg_data(data_dir):
    """
    加载所有被试的 EEG 数据，按被试 ID 重新组织。

    原始数据结构:
      每个 .npz 文件对应一个被试的一次会话 (subject_session.npz):
        - train_data (pickle dict): delta/theta/alpha/beta/gamma, 每个 (n_train, 62)
        - test_data  (pickle dict): 同上, 每个 (n_test, 62)
        - train_label: (n_train,)  标签 {0, 1, 2}
        - test_label:  (n_test,)   标签 {0, 1, 2}

    数据布局 (band-major):
      将 5 个频段 × 62 个通道拼接为 310 维特征:
        索引   0 ~  61: delta 频段的 62 个通道值
        索引  62 ~ 123: theta 频段的 62 个通道值
        索引 124 ~ 185: alpha 频段的 62 个通道值
        索引 186 ~ 247: beta  频段的 62 个通道值
        索引 248 ~ 309: gamma 频段的 62 个通道值

    Args:
        data_dir: EEG .npz 文件所在目录

    Returns:
        subject_data: dict {subject_id: {'X': (n_samples, 310), 'y': (n_samples,)}}
        subject_ids: 排序后的被试 ID 列表
    """
    subject_data = {}
    files = sorted(os.listdir(data_dir))

    for fname in files:
        if not fname.endswith('.npz'):
            continue
        basename = fname.replace('.npz', '')
        subject_id = int(basename.split('_')[0])

        npz = np.load(os.path.join(data_dir, fname), allow_pickle=True)
        train_dict = pickle.loads(npz['train_data'])
        test_dict = pickle.loads(npz['test_data'])
        train_label = npz['train_label']
        test_label = npz['test_label']

        n_train, n_test = len(train_label), len(test_label)
        n_total = n_train + n_test
        n_ch = CONFIG['n_channels']
        n_bands = CONFIG['n_bands']

        # 拼接为 band-major 布局的 310 维特征
        X_session = np.zeros((n_total, n_ch * n_bands), dtype=np.float32)
        y_session = np.concatenate([train_label, test_label]).astype(np.int64)

        band_keys = [b.split()[0] for b in CONFIG['band_names']]
        for band_idx, band_key in enumerate(band_keys):
            all_band = np.concatenate([train_dict[band_key], test_dict[band_key]], axis=0)
            X_session[:, band_idx * n_ch:(band_idx + 1) * n_ch] = all_band

        # 按被试聚合
        if subject_id not in subject_data:
            subject_data[subject_id] = {'X': [], 'y': []}
        subject_data[subject_id]['X'].append(X_session)
        subject_data[subject_id]['y'].append(y_session)

    # 合并同一被试的多个会话，并逐被试 z-score 标准化
    for sid in subject_data:
        subject_data[sid]['X'] = np.concatenate(subject_data[sid]['X'], axis=0)
        subject_data[sid]['y'] = np.concatenate(subject_data[sid]['y'], axis=0)
        # 逐被试独立标准化：消除个体基线差异，保留情绪相关的相对模式
        subject_data[sid]['X'] = StandardScaler().fit_transform(
            subject_data[sid]['X']).astype(np.float32)

    subject_ids = sorted(subject_data.keys())
    print(f"共加载 {len(subject_ids)} 名被试数据，被试ID = {subject_ids}")
    for sid in subject_ids:
        dist = dict(zip(*np.unique(subject_data[sid]['y'], return_counts=True)))
        print(f"  被试 {sid}: {subject_data[sid]['X'].shape[0]} 样本, "
              f"标签分布 = {dist}")
    return subject_data, subject_ids

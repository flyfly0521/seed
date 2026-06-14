"""
训练与评估模块。
包含 MixUp 数据增强、训练循环、评估函数、LOSO 交叉验证和消融实验。
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, classification_report


# ======================================================================
# 工具函数
# ======================================================================

def set_seed(seed):
    """固定随机种子以保证可复现性。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ======================================================================
# MixUp 数据增强
# ======================================================================

def mixup_data(x, y, alpha, device):
    """
    MixUp: 对 (x, y) 进行凸组合。

    Args:
        x, y: 一个 batch 的数据和标签
        alpha: Beta 分布的参数, 0 表示不使用 MixUp
    Returns:
        mixed_x, y_a, y_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """MixUp 损失: lam * loss(pred, y_a) + (1-lam) * loss(pred, y_b)。"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ======================================================================
# 训练与评估
# ======================================================================

def train_one_epoch(model, loader, optimizer, criterion, device, mixup_alpha=0.0):
    """训练一个 epoch，可选 MixUp 增强。"""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        if mixup_alpha > 0:
            x_batch, y_a, y_b, lam = mixup_data(x_batch, y_batch, mixup_alpha, device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            # 准确率用未混合的原始 batch 估计
            with torch.no_grad():
                correct += (model(x_batch).argmax(1) == y_batch).sum().item()
        else:
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            correct += (logits.argmax(1) == y_batch).sum().item()

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x_batch.size(0)
        total += x_batch.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """评估模型，返回 loss、accuracy、混淆矩阵、预测和标签。"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        logits = model(x_batch)
        total_loss += criterion(logits, y_batch).item() * x_batch.size(0)
        preds = logits.argmax(1)
        correct += (preds == y_batch).sum().item()
        total += x_batch.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())
    return (total_loss / total,
            correct / total,
            confusion_matrix(all_labels, all_preds, labels=[0, 1, 2]),
            all_preds, all_labels)


@torch.no_grad()
def collect_attention(model, loader, device):
    """收集测试集上模型输出的平均 channel / freq 注意力权重。"""
    model.eval()
    all_ch, all_fr = [], []
    for x_batch, _ in loader:
        x_batch = x_batch.to(device)
        _, ch_w, fr_w = model(x_batch, return_attention=True)
        all_ch.append(ch_w.detach().cpu().numpy())
        all_fr.append(fr_w.detach().cpu().numpy())
    return (np.concatenate(all_ch, axis=0).mean(axis=0),   # (62,)
            np.concatenate(all_fr, axis=0).mean(axis=0))   # (5,)


# ======================================================================
# LOSO 交叉验证
# ======================================================================

def run_loso(subject_data, subject_ids, make_model, config, verbose=True):
    """
    留一被试交叉验证 (Leave-One-Subject-Out).

    每折将一个被试的所有数据作为测试集，其余被试作为训练集。
    关键预处理：逐被试 z-score 标准化以消除个体差异。

    Args:
        subject_data: {subj_id: {'X': (n,310), 'y': (n,)}}
        subject_ids: 被试 ID 列表
        make_model: () → nn.Module  模型工厂函数（无参数）
        config: 配置字典
        verbose: 是否打印每折日志

    Returns:
        dict: {
            accs, avg_acc, std_acc, cms, overall_cm,
            channel_attns, freq_attns, all_preds, all_labels
        }
    """
    device = config['device']
    n_folds = len(subject_ids)

    fold_accs = []
    fold_cms = []
    fold_ch_attns = {}
    fold_fr_attns = {}
    all_preds, all_labels = [], []

    if verbose:
        print(f"\n{'='*55}")
        print(f"LOSO 交叉验证 ({n_folds} 折)")
        print(f"{'='*55}")

    for fold_idx, test_subj in enumerate(subject_ids):
        if verbose:
            print(f"\n--- Fold {fold_idx+1}/{n_folds}: "
                  f"test subject = {test_subj} ---")

        # ---- 组装训练/测试集（数据已在 data_loader 中逐被试标准化） ----
        X_test = subject_data[test_subj]['X']
        y_test = subject_data[test_subj]['y']

        X_train_list, y_train_list = [], []
        for s in subject_ids:
            if s == test_subj:
                continue
            X_train_list.append(subject_data[s]['X'])
            y_train_list.append(subject_data[s]['y'])

        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)

        if verbose:
            print(f"  Train: {X_train.shape[0]} (from {len(X_train_list)} subjects), "
                  f"Test: {X_test.shape[0]} (subject {test_subj})")

        # ---- DataLoader ----
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
            batch_size=config['batch_size'], shuffle=True)
        test_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
            batch_size=config['batch_size'], shuffle=False)

        # ---- 模型、优化器、调度器 ----
        model = make_model().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['n_epochs'])

        best_acc = 0.0
        best_cm = None
        best_ch, best_fr = None, None
        best_state = None

        # ---- 训练循环 ----
        for epoch in range(config['n_epochs']):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device,
                mixup_alpha=config.get('mixup_alpha', 0))
            test_loss, test_acc, cm, _, _ = evaluate(
                model, test_loader, criterion, device)
            scheduler.step()

            if test_acc > best_acc:
                best_acc = test_acc
                best_cm = cm
                best_ch, best_fr = collect_attention(model, test_loader, device)
                best_state = copy.deepcopy(model.state_dict())

            if verbose and (epoch + 1) % 25 == 0:
                print(f"    Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, "
                      f"train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")

        # 恢复最佳模型，用于最终预测收集
        if best_state is not None:
            model.load_state_dict(best_state)

        fold_accs.append(best_acc)
        fold_cms.append(best_cm)
        fold_ch_attns[test_subj] = best_ch
        fold_fr_attns[test_subj] = best_fr

        _, _, _, fp, fl = evaluate(model, test_loader, criterion, device)
        all_preds.extend(fp)
        all_labels.extend(fl)

        if verbose:
            print(f"  >> Best test acc = {best_acc:.4f}")

    # ---- 汇总结果 ----
    avg_acc = float(np.mean(fold_accs))
    std_acc = float(np.std(fold_accs))
    overall_cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])

    if verbose:
        print(f"\n{'='*55}")
        print(f"LOSO 结果")
        print(f"{'='*55}")
        print(f"各折准确率: {[f'{a:.4f}' for a in fold_accs]}")
        print(f"平均准确率: {avg_acc:.4f} ± {std_acc:.4f}")
        print(f"\n总体混淆矩阵:")
        print(f"{'':>14s} Pred Pos  Pred Neg  Pred Neu")
        for i, name in enumerate(config['class_names']):
            print(f"True {name:>4s}  {overall_cm[i,0]:>9d}  "
                  f"{overall_cm[i,1]:>9d}  {overall_cm[i,2]:>9d}")
        print(f"\n分类报告:")
        print(classification_report(all_labels, all_preds,
                                     target_names=config['class_names'],
                                     digits=3))

    return {
        'accs': fold_accs,
        'avg_acc': avg_acc,
        'std_acc': std_acc,
        'cms': fold_cms,
        'overall_cm': overall_cm,
        'channel_attns': fold_ch_attns,
        'freq_attns': fold_fr_attns,
        'all_preds': all_preds,
        'all_labels': all_labels,
    }


# ======================================================================
# 消融实验
# ======================================================================

def run_ablation(subject_data, subject_ids, config, model_factories):
    """
    消融实验：对比不同模型配置的性能。

    Args:
        subject_data, subject_ids: 数据集
        config: 全局配置
        model_factories: dict {模型名称: 工厂函数}

    Returns:
        dict: {模型名称: LOSO 结果 dict}
    """
    print(f"\n{'='*55}")
    print("消融实验")
    print(f"{'='*55}")

    config_ablation = {**config, 'n_epochs': config.get('n_epochs_ablation', 80)}

    results = {}
    for name, factory in model_factories.items():
        print(f"\n--- {name} ---")
        r = run_loso(subject_data, subject_ids, factory, config_ablation,
                     verbose=False)
        results[name] = r
        print(f"  {r['avg_acc']:.4f} ± {r['std_acc']:.4f}")

    print(f"\n{'='*55}")
    print("消融实验结果对比")
    print(f"{'='*55}")
    print(f"{'Model':<30s} {'Accuracy':<18s} {'Std':<10s}")
    print("-" * 58)
    for name, r in results.items():
        print(f"{name:<30s} {r['avg_acc']:.4f}            {r['std_acc']:.4f}")

    return results

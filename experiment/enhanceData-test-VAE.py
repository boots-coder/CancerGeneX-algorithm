from copy import deepcopy

import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd


def get_device():
    """
    优先使用 MPS (Metal Performance Shaders)，若不可用则用CPU或CUDA。
    需要 macOS+Apple Silicon 并安装 PyTorch>=1.12 来支持 MPS。
    # """
    # if torch.backends.mps.is_available():
    #     return torch.device("mps")
    # elif torch.cuda.is_available():
    #     return torch.device("cuda")
    # else:
    return torch.device("cpu")


class ResidualBlock(nn.Module):
    """残差块"""

    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out


class CVAE(nn.Module):
    """条件变分自编码器"""

    def __init__(self, input_dim, n_classes, latent_dim=20, hidden_dims=[512, 256, 128]):
        super(CVAE, self).__init__()

        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.input_dim = input_dim

        # 编码器
        encoder_layers = []
        prev_dim = input_dim + n_classes  # 添加类别信息的输入维度

        for dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                ResidualBlock(dim)
            ])
            prev_dim = dim

        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # 解码器
        decoder_layers = []
        prev_dim = latent_dim + n_classes  # 潜在空间加类别信息

        for dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                ResidualBlock(dim)
            ])
            prev_dim = dim

        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x, c):
        """编码过程"""
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        if len(c.shape) == 1:
            c = c.unsqueeze(0)
        inputs = torch.cat([x, c], dim=1)
        h = self.encoder(inputs)
        return self.fc_mu(h), self.fc_var(h)

    def reparameterize(self, mu, log_var, temperature=1.0):
        """重参数化技巧"""
        std = torch.exp(0.5 * log_var) * temperature
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        """解码过程"""
        if len(z.shape) == 1:
            z = z.unsqueeze(0)
        if len(c.shape) == 1:
            c = c.unsqueeze(0)
        inputs = torch.cat([z, c], dim=1)
        return self.decoder(inputs)

    def forward(self, x, c, temperature=1.0):
        mu, log_var = self.encode(x, c)
        z = self.reparameterize(mu, log_var, temperature)
        return self.decode(z, c), mu, log_var

    def generate(self, n_samples, c, device, temperature=1.0):
        """生成样本"""
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim).to(device) * temperature
            if len(c.shape) == 1:
                c = c.unsqueeze(0)
            if c.shape[0] == 1:
                c = c.repeat(n_samples, 1)
            return self.decode(z, c)


def compute_contrastive_loss(features, labels, temperature=0.5):
    """
    逐行（per-sample）计算对比损失，避免批量内部同类样本数不一致导致的 reshape 错误。
    """
    device = features.device
    batch_size = features.size(0)
    # 归一化
    features = F.normalize(features, dim=1)

    total_loss = 0.0
    for i in range(batch_size):
        # (1) 相似度
        sim_i = torch.matmul(features[i].unsqueeze(0), features.T) / temperature

        # (2) 同类mask
        label_i = labels[i]
        mask_i = (labels == label_i)
        # 可选：不包含自己
        # mask_i[i] = False

        positives_i = sim_i[:, mask_i]    # shape [1, #pos]
        negatives_i = sim_i[:, ~mask_i]   # shape [1, #neg]

        logits_i = torch.cat([positives_i, negatives_i], dim=1)
        target_i = torch.zeros(1, dtype=torch.long, device=device)
        loss_i = F.cross_entropy(logits_i, target_i)
        total_loss += loss_i

    return total_loss / batch_size


def train_cvae(X_train, y_train, X_val, y_val, n_classes, epochs=300, batch_size=32,
               latent_dim=20, early_stopping_patience=30):
    """训练条件VAE"""
    device = get_device()

    # 动态调整批量大小
    batch_size = min(batch_size, len(X_train) // 2)
    batch_size = max(1, batch_size)

    input_dim = X_train.shape[1]
    cvae = CVAE(input_dim, n_classes, latent_dim).to(device)
    optimizer = torch.optim.AdamW(cvae.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)

    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)

    # 创建one-hot编码
    y_train_onehot = torch.zeros(len(y_train), n_classes)
    y_train_onehot.scatter_(1, y_train.unsqueeze(1), 1)

    y_val_onehot = torch.zeros(len(y_val), n_classes)
    y_val_onehot.scatter_(1, y_val.unsqueeze(1), 1)

    print(f"训练集大小: {len(X_train)}")
    print(f"验证集大小: {len(X_val)}")
    print(f"使用的批量大小: {batch_size}")

    train_dataset = TensorDataset(torch.FloatTensor(X_train), y_train_onehot)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    val_dataset = TensorDataset(torch.FloatTensor(X_val), y_val_onehot)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=False)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model = None

    temp_min = 0.1
    temp_max = 1.0
    temp_cycles = 3

    try:
        for epoch in range(epochs):
            cvae.train()
            train_loss = 0.0

            # 余弦式退火温度
            temp = temp_min + 0.5 * (temp_max - temp_min) * (
                1 + np.cos(
                    np.pi * ((epoch % (epochs // temp_cycles)) / (epochs // temp_cycles))
                )
            )

            for batch_idx, (data, labels_onehot) in enumerate(train_loader):
                data = data.to(device)
                labels_onehot = labels_onehot.to(device)

                optimizer.zero_grad()

                recon_batch, mu, log_var = cvae(data, labels_onehot, temperature=temp)
                recon_loss = F.mse_loss(recon_batch, data, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

                # labels_idx 用于对比损失
                labels_idx = torch.argmax(labels_onehot, dim=1)
                contrastive_loss = compute_contrastive_loss(mu, labels_idx)

                beta = min(1.0, epoch / 50)
                loss = recon_loss + beta * kl_loss + 0.1 * contrastive_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(cvae.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()

            scheduler.step()

            # 验证
            cvae.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_idx, (data, labels_onehot) in enumerate(val_loader):
                    data = data.to(device)
                    labels_onehot = labels_onehot.to(device)

                    recon_batch, mu, log_var = cvae(data, labels_onehot)
                    recon_loss = F.mse_loss(recon_batch, data, reduction='sum')
                    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                    val_loss += (recon_loss + beta * kl_loss).item()

            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = CVAE(input_dim, n_classes, latent_dim)
                best_model.load_state_dict(cvae.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

            if (epoch + 1) % 50 == 0:
                print(f'Epoch {epoch + 1}:')
                print(f'Training loss = {train_loss / len(train_loader.dataset):.4f}')
                print(f'Validation loss = {val_loss / len(val_loader.dataset):.4f}')
                print(f'Temperature = {temp:.4f}')
                print(f'Learning rate = {optimizer.param_groups[0]["lr"]:.6f}')

    except Exception as e:
        print(f"训练过程中出错: {str(e)}")
        raise

    return best_model, device


def generate_synthetic_data_for_iteration(cvae, X_train, y_train, device,
                                          multiplier=5, class_weights=None,
                                          iteration=0, n_classes=2):
    """为特定迭代动态生成合成数据"""
    cvae.eval()
    synthetic_samples = []
    synthetic_labels = []

    # 计算每个类别的特征统计信息
    class_statistics = {}
    for class_label in range(n_classes):
        class_mask = y_train == class_label
        if np.any(class_mask):
            class_samples = X_train[class_mask]
            class_statistics[class_label] = {
                'mean': np.mean(class_samples, axis=0),
                'std': np.std(class_samples, axis=0),
                'covar': np.cov(class_samples.T)
            }

    for class_label in range(n_classes):
        class_mask = y_train == class_label
        class_samples = X_train[class_mask]
        if len(class_samples) == 0:
            continue

        weight = class_weights.get(class_label, 1.0) if class_weights else 1.0
        n_synthetic = int(len(class_samples) * multiplier * weight)

        # 类别one-hot
        class_onehot = F.one_hot(torch.tensor([class_label]), n_classes).float().to(device)
        class_onehot = class_onehot.repeat(n_synthetic, 1)

        noise_scales = [
            max(0.1, 0.8 - 0.1 * iteration),
            1.0,
            min(2.0, 1.2 + 0.1 * iteration)
        ]

        for noise_scale in noise_scales:
            n_samples = n_synthetic // len(noise_scales)
            with torch.no_grad():
                z = torch.randn(n_samples, cvae.latent_dim).to(device) * noise_scale
                synth = cvae.decode(z, class_onehot[:n_samples]).cpu().numpy()

            if class_label in class_statistics:
                synth = post_process_samples(synth, class_statistics[class_label], iteration)

            synthetic_samples.append(synth)
            synthetic_labels.extend([class_label] * len(synth))

    if not synthetic_samples:
        return np.array([]), np.array([])

    X_synthetic = np.vstack(synthetic_samples)
    y_synthetic = np.array(synthetic_labels)
    return X_synthetic, y_synthetic


def post_process_samples(synthetic_samples, class_stats, iteration):
    """后处理合成样本以提高质量"""
    mahalanobis_dist = calculate_mahalanobis_distance(
        synthetic_samples,
        class_stats['mean'],
        class_stats['covar']
    )

    threshold = np.percentile(mahalanobis_dist, 98 - iteration)
    valid_samples = synthetic_samples[mahalanobis_dist < threshold]
    if len(valid_samples) == 0:
        return synthetic_samples

    valid_samples = maintain_feature_correlations(
        valid_samples,
        class_stats['covar'],
        strength=0.5 / (iteration + 1)
    )
    return valid_samples


def calculate_mahalanobis_distance(X, mean, covar):
    """计算马氏距离"""
    covar_inv = np.linalg.pinv(covar)
    diff = X - mean
    dist = np.sqrt(np.sum(np.dot(diff, covar_inv) * diff, axis=1))
    return dist


def maintain_feature_correlations(samples, covar, strength=0.5):
    """保持特征间的相关性"""
    corr = np.corrcoef(samples.T)
    target_corr = np.corrcoef(covar)
    adjusted_samples = samples.copy()
    for i in range(samples.shape[1]):
        for j in range(i + 1, samples.shape[1]):
            if abs(target_corr[i, j]) > 0.3:
                adjustment = (target_corr[i, j] - corr[i, j]) * strength
                adjusted_samples[:, j] += adjustment * samples[:, i]
    return adjusted_samples


def compute_class_weights(y):
    """计算类别权重"""
    class_counts = np.bincount(y)
    total_samples = len(y)
    class_weights = {}
    for i, count in enumerate(class_counts):
        class_weights[i] = total_samples / (len(class_counts) * count)
    return class_weights


def clone(clf):
    """创建分类器的深度复制"""
    # 如果分类器参数固定，可简单返回新实例
    return LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced'
    )


def train_with_dynamic_augmentation(X, y, train_size, n_classes):
    """
    使用动态数据增强进行训练和评估，按指定 train_size 划分数据集。
    - train_size: 训练集占整个数据的比例。
    """
    try:
        metrics = {'accuracy': [], 'f1': [], 'precision': [], 'recall': [], 'auc': []}

        print(f"\n开始训练，训练集比例: {train_size}")
        print(f"数据集大小: {X.shape}")
        print(f"类别数: {n_classes}")

        # 按 train_size 划分数据集
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=1-train_size, random_state=42, stratify=y
        )
        X_test, X_val, y_test, y_val = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )

        print(f"训练集大小: {X_train.shape[0]}, 验证集大小: {X_val.shape[0]}, 测试集大小: {X_test.shape[0]}")

        # 训练CVAE
        cvae, device = train_cvae(X_train, y_train, X_val, y_val, n_classes)

        class_weights = compute_class_weights(y_train)
        clf = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')

        best_val_score = 0.0
        best_clf = None
        patience = 5
        patience_counter = 0
        has_fit_once = False  # 标识是否成功 fit 过

        for iteration in range(10):
            # 动态生成增强数据
            X_synthetic, y_synthetic = generate_synthetic_data_for_iteration(
                cvae, X_train, y_train, device,
                multiplier=max(1, 3 * (1 - train_size)),
                class_weights=class_weights,
                iteration=iteration,
                n_classes=n_classes
            )

            # 如果生成数据量为 0，则跳过本轮
            if len(X_synthetic) == 0:
                print(f"第 {iteration + 1} 次动态生成数据失败，跳过...")
                continue

            # 合并增强数据
            X_train_aug = np.vstack([X_train, X_synthetic])
            y_train_aug = np.hstack([y_train, y_synthetic])

            # 训练分类器
            clf.fit(X_train_aug, y_train_aug)
            has_fit_once = True

            # 验证性能
            val_score = f1_score(y_val, clf.predict(X_val), average='weighted')
            if val_score > best_val_score:
                best_val_score = val_score
                best_clf = deepcopy(clf)
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("早停条件满足，结束动态增强...")
                break

        # 如果一次都没 fit，就用原始训练数据训练
        if not has_fit_once:
            print("动态增强失败，回退到原始训练数据...")
            clf.fit(X_train, y_train)
            best_clf = clf

        # 若有最佳模型，则使用
        if best_clf is not None:
            clf = best_clf

        # 测试集评估性能
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)

        metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['f1'].append(f1_score(y_test, y_pred, average='weighted'))
        metrics['precision'].append(precision_score(y_test, y_pred, average='weighted'))
        metrics['recall'].append(recall_score(y_test, y_pred, average='weighted'))
        if n_classes == 2:
            metrics['auc'].append(roc_auc_score(y_test, y_pred_proba[:, 1]))

    except Exception as e:
        print(f"动态增强训练过程出错: {str(e)}")
        raise

    return {k: (np.mean(v), np.std(v)) for k, v in metrics.items()}

def evaluate_with_dynamic_split(X, y, train_size, random_state=42):
    """
    使用动态分割的方式评估基线模型性能。
    - train_size: 训练集占整个数据的比例。
    """
    metrics = {'accuracy': [], 'f1': [], 'precision': [], 'recall': [], 'auc': []}

    # 划分训练集与剩余数据
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=1-train_size, random_state=random_state, stratify=y
    )

    # 剩余数据按0.5/0.5划分为测试集和验证集
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
    )

    clf = LogisticRegression(random_state=random_state, max_iter=1000, class_weight='balanced')

    # 训练模型
    clf.fit(X_train, y_train)

    # 验证集性能
    y_pred_val = clf.predict(X_val)
    y_pred_proba_val = clf.predict_proba(X_val)

    metrics['accuracy'].append(accuracy_score(y_val, y_pred_val))
    metrics['f1'].append(f1_score(y_val, y_pred_val, average='weighted'))
    metrics['precision'].append(precision_score(y_val, y_pred_val, average='weighted'))
    metrics['recall'].append(recall_score(y_val, y_pred_val, average='weighted'))

    if len(np.unique(y)) == 2:
        metrics['auc'].append(roc_auc_score(y_val, y_pred_proba_val[:, 1]))

    # 将结果格式化返回
    return {k: (np.mean(v), np.std(v)) for k, v in metrics.items()}

def evaluate_dataset(X, y, dataset_name, train_sizes=[0.4, 0.5, 0.6, 0.7, 0.8]):
    """评估单个数据集"""
    results = {
        'train_size': [],
        'baseline': {'mean': [], 'std': []},
        'vae': {'mean': [], 'std': []},
        'dataset_name': dataset_name,
        'detailed_metrics': []
    }

    scaler = StandardScaler()
    n_features = min(50, X.shape[1])
    if X.shape[1] > 1000:
        n_features = min(100, X.shape[1] // 10)

    selector = SelectKBest(score_func=f_classif, k=n_features)
    X_processed = scaler.fit_transform(X)
    X_processed = selector.fit_transform(X_processed, y)
    X_processed = X_processed.astype(np.float32)

    n_classes = len(np.unique(y))

    print(f"\n===== 处理数据集: {dataset_name} =====")
    print(f"数据集大小: {X.shape}")
    print(f"处理后特征数: {X_processed.shape[1]}")
    print(f"类别分布: {np.bincount(y)}")

    for ts in train_sizes:
        print(f"\n----- 评估训练集比例: {ts:.2f} -----")
        baseline_metrics = evaluate_with_dynamic_split(X_processed, y, ts)
        vae_metrics = train_with_dynamic_augmentation(X_processed, y, ts, n_classes)

        results['train_size'].append(ts)
        results['baseline']['mean'].append(baseline_metrics['accuracy'][0])
        results['baseline']['std'].append(baseline_metrics['accuracy'][1])
        results['vae']['mean'].append(vae_metrics['accuracy'][0])
        results['vae']['std'].append(vae_metrics['accuracy'][1])

        detailed_metrics = {
            'train_size': ts,
            'baseline': baseline_metrics,
            'vae': vae_metrics
        }
        results['detailed_metrics'].append(detailed_metrics)

        print("\n基线模型性能:")
        for metric, (mean, std) in baseline_metrics.items():
            print(f"{metric}: {mean:.4f} ± {std:.4f}")

        print("\nVAE增强模型性能:")
        for metric, (mean, std) in vae_metrics.items():
            print(f"{metric}: {mean:.4f} ± {std:.4f}")

    return results


def load_data_from_mat(file_path):
    """从.mat文件加载数据集"""
    try:
        mat_data = scipy.io.loadmat(file_path)
        X = mat_data['X']  # 特征矩阵
        Y = mat_data['Y'].flatten()  # 标签向量

        # 修正标签到 {0, 1}
        if -1 in Y:
            Y[Y == -1] = 0
        elif 2 in Y:
            Y[Y == 2] = 0

        print(f"成功加载数据集: {file_path}")
        print(f"特征矩阵形状: {X.shape}")
        print(f"标签分布: {np.bincount(Y)}")
        return X, Y

    except Exception as e:
        print(f"加载数据集时出错 {file_path}: {str(e)}")
        raise


def plot_all_datasets_comparison(all_results):
    """为所有数据集绘制详细的对比图"""
    n_datasets = len(all_results)
    n_rows = (n_datasets + 1) // 2
    fig = plt.figure(figsize=(15, 6 * n_rows))
    plt.style.use('default')
    colors = ['#2ecc71', '#e74c3c']

    for idx, results in enumerate(all_results):
        ax1 = plt.subplot(n_rows, 2, idx + 1)

        for model, color, label in zip(
            ['baseline', 'vae'],
            colors,
            ['Baseline', 'VAE Augmented']
        ):
            means = results[model]['mean']
            stds = results[model]['std']
            ax1.plot(results['train_size'], means, marker='o', color=color,
                     label=label, linewidth=2)
            ax1.fill_between(
                results['train_size'],
                np.array(means) - np.array(stds),
                np.array(means) + np.array(stds),
                color=color, alpha=0.2
            )
        ax1.set_xlabel('Training Set Ratio')
        ax1.set_ylabel('Accuracy')
        ax1.set_title(f'Dataset: {results["dataset_name"]}')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(loc='lower right')

        # 标注性能提升
        for i, (size, base, vae) in enumerate(zip(
            results['train_size'],
            results['baseline']['mean'],
            results['vae']['mean']
        )):
            improvement = ((vae - base) / base) * 100
            if improvement > 0:
                ax1.annotate(
                    f'+{improvement:.1f}%',
                    xy=(size, vae),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    va='bottom',
                    color='green'
                )

    plt.tight_layout()
    plt.show()
    # plot_summary_statistics(all_results)


# def plot_summary_statistics(all_results):
#     """绘制汇总统计图"""
#     improvements = []
#     dataset_names = []
#
#     for results in all_results:
#         base_mean = np.mean(results['baseline']['mean'])
#         vae_mean = np.mean(results['vae']['mean'])
#         improvement = ((vae_mean - base_mean) / base_mean) * 100
#         improvements.append(improvement)
#         dataset_names.append(results['dataset_name'])
#
#     plt.figure(figsize=(10, 6))
#     bars = plt.bar(dataset_names, improvements)
#     for i, v in enumerate(improvements):
#         bars[i].set_color('#2ecc71' if v > 0 else '#e74c3c')
#     plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
#     plt.title('VAE Improvement Over Baseline (%)')
#     plt.xticks(rotation=45, ha='right')
#     plt.ylabel('Improvement (%)')
#
#     for i, v in enumerate(improvements):
#         plt.text(i, v, f'{v:.1f}%',
#                  ha='center', va='bottom' if v > 0 else 'top')
#     plt.tight_layout()
#     plt.show()


def save_detailed_results(all_results, output_file='detailed_results.csv'):
    """保存详细结果到CSV文件"""
    rows = []
    for results in all_results:
        dataset_name = results['dataset_name']
        for detailed_metric in results['detailed_metrics']:
            train_size = detailed_metric['train_size']
            for metric, (mean, std) in detailed_metric['baseline'].items():
                rows.append({
                    'Dataset': dataset_name,
                    'Train_Size': train_size,
                    'Model': 'Baseline',
                    'Metric': metric,
                    'Mean': mean,
                    'Std': std
                })
            for metric, (mean, std) in detailed_metric['vae'].items():
                rows.append({
                    'Dataset': dataset_name,
                    'Train_Size': train_size,
                    'Model': 'VAE',
                    'Metric': metric,
                    'Mean': mean,
                    'Std': std
                })
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"\n详细结果已保存到: {output_file}")


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.backends.mps.is_available():
        torch.manual_seed(42)  # MPS 上同样可设置随机种子
    #
    # datasets = [
    #     "ALLAML.mat", "colon.mat", "GLI_85.mat",
    #     "leukemia.mat", "Prostate_GE.mat", "SMK_CAN_187.mat"
    # ]
    datasets = [
        "ALLAML.mat"
    ]
    data_dir = "../data/"
    train_sizes = [0.4, 0.5, 0.6, 0.7, 0.8]

    all_results = []

    for dataset in datasets:
        try:
            file_path = os.path.join(data_dir, dataset)
            X, y = load_data_from_mat(file_path)
            results = evaluate_dataset(X, y, dataset, train_sizes)
            all_results.append(results)
        except Exception as e:
            print(f"处理数据集 {dataset} 时出错: {str(e)}")
            continue

    plot_all_datasets_comparison(all_results)
    save_detailed_results(all_results)

    print("\n===== 实验结果汇总 =====")
    for results in all_results:
        dataset_name = results['dataset_name']
        baseline_mean = np.mean(results['baseline']['mean'])
        vae_mean = np.mean(results['vae']['mean'])
        improvement = ((vae_mean - baseline_mean) / baseline_mean) * 100
        print(f"\n数据集: {dataset_name}")
        print(f"基线模型平均准确率: {baseline_mean:.4f}")
        print(f"VAE增强模型平均准确率: {vae_mean:.4f}")
        print(f"性能提升: {improvement:+.2f}%")
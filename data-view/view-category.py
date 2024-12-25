"""
• 改进效果：改进后的 Transformer 在复杂数据集（如 SMK_CAN_187.mat）上表现更好，显著提高了 Accuracy 和部分 Precision。
 • 不足之处：在部分数据集（如 colon.mat 和 GLI_85.mat）上，改进模型的 Recall 和 AUC 稍有下降，可能对不均衡数据的正类识别稍逊。
 • 适用场景：改进模型更适合应对复杂、不平衡数据集，但在一些对召回率要求较高的场景中，需进一步优化。


"""

import scipy.io as sio
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_analyze_data(file_path):
    """
    加载并分析.mat文件数据
    """
    # 加载数据
    data = sio.loadmat(file_path)

    # 提取数据和标签
    X = data['X'] if 'X' in data else None
    Y = data['Y'] if 'Y' in data else None

    if X is None or Y is None:
        print(f"数据格式错误，可用的键：{data.keys()}")
        return

    # 基本信息
    n_samples, n_features = X.shape
    unique_labels, label_counts = np.unique(Y, return_counts=True)

    # 打印基本信息
    print(f"\n数据集: {os.path.basename(file_path)}")
    print(f"样本数量: {n_samples}")
    print(f"特征数量: {n_features}")
    print("\n类别分布:")
    for label, count in zip(unique_labels.flatten(), label_counts):
        print(f"类别 {label}: {count} 样本 ({count / n_samples * 100:.2f}%)")

    return X, Y, unique_labels, label_counts


def plot_class_distribution(datasets_info):
    """
    绘制类别分布对比图
    """
    plt.figure(figsize=(12, 6))

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    n_datasets = len(datasets_info)
    bar_width = 0.35

    for i, (dataset_name, (labels, counts)) in enumerate(datasets_info.items()):
        x = np.arange(len(labels.flatten()))
        plt.bar(x + i * bar_width, counts, bar_width, label=dataset_name)

        # 添加数值标签
        for j, count in enumerate(counts):
            plt.text(x[j] + i * bar_width, count, str(count),
                     ha='center', va='bottom')

    plt.xlabel('类别')
    plt.ylabel('样本数量')
    plt.title('数据集类别分布对比')
    plt.legend()

    # 设置x轴刻度
    plt.xticks(x + bar_width / 2, [f'类别 {int(label)}' for label in labels.flatten()])

    plt.tight_layout()
    plt.show()


def main():
    # 数据路径
    base_path = "/Users/bootscoder/Documents/岳老师-科研/贾浩轩-Adap-BDCM/paper2/Adap-BDCM-main/data"
    datasets = ['colon.mat', 'GLI_85.mat']

    datasets_info = {}

    # 分析每个数据集
    for dataset in datasets:
        file_path = os.path.join(base_path, dataset)
        if os.path.exists(file_path):
            X, Y, labels, counts = load_and_analyze_data(file_path)
            datasets_info[dataset] = (labels, counts)
        else:
            print(f"文件不存在: {file_path}")

    # 绘制对比图
    plot_class_distribution(datasets_info)

    # 输出数据集之间的对比信息
    print("\n数据集对比:")
    for dataset in datasets:
        if dataset in datasets_info:
            X = sio.loadmat(os.path.join(base_path, dataset))['X']
            print(f"\n{dataset}:")
            print(f"特征维度: {X.shape[1]}")
            print(f"样本数量: {X.shape[0]}")
            print(f"特征值范围: [{X.min():.3f}, {X.max():.3f}]")
            print(f"特征均值: {X.mean():.3f}")
            print(f"特征标准差: {X.std():.3f}")


if __name__ == "__main__":
    main()
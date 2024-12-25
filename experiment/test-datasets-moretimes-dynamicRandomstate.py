import scipy.io as sio
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
import pandas as pd


# 设置中文字体
def set_chinese_font():
    """设置中文字体"""
    try:
        # Mac系统
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    except:
        try:
            # Windows系统
            plt.rcParams['font.sans-serif'] = ['SimHei']
        except:
            print("警告：未能找到合适的中文字体，可能会导致中文显示异常")
    plt.rcParams['axes.unicode_minus'] = False


def load_and_analyze_data(file_path):
    """
    加载并分析.mat文件数据
    返回：数据矩阵、标签、唯一标签值、每个标签的数量
    """
    try:
        # 加载数据
        data = sio.loadmat(file_path)

        # 提取数据和标签
        X = data['X'] if 'X' in data else None
        Y = data['Y'] if 'Y' in data else None

        if X is None or Y is None:
            print(f"数据格式错误，可用的键：{data.keys()}")
            return None, None, None, None

        # 基本信息
        n_samples, n_features = X.shape
        unique_labels, label_counts = np.unique(Y, return_counts=True)

        # 打印基本信息
        print(f"\n数据集: {os.path.basename(file_path)}")
        print(f"样本总数: {n_samples}")
        print(f"特征维度: {n_features}")
        print("\n类别分布:")
        for label, count in zip(unique_labels.flatten(), label_counts):
            percentage = count / n_samples * 100
            print(f"类别 {label}: {count} 样本 ({percentage:.2f}%)")

        return X, Y, unique_labels, label_counts

    except Exception as e:
        print(f"加载数据时出错: {str(e)}")
        return None, None, None, None


def create_dataset_summary_table(datasets_info, feature_stats):
    """创建数据集汇总表"""
    data = []
    for dataset_name, (labels, counts) in datasets_info.items():
        stats = feature_stats[dataset_name]
        row = {
            '数据集': dataset_name.replace('.mat', ''),
            '样本数': sum(counts),
            '特征数': stats['n_features'],
            '类别数': len(counts),
            '类别分布': ' : '.join(map(str, counts)),
            '特征均值': f"{stats['mean']:.3f}",
            '特征标准差': f"{stats['std']:.3f}"
        }
        data.append(row)
    return pd.DataFrame(data)


def plot_sample_feature_distribution(datasets_info, feature_stats):
    """绘制样本数量和特征维度分布图"""
    set_chinese_font()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 准备数据
    names = [name.replace('.mat', '') for name in datasets_info.keys()]
    samples = [sum(counts) for _, (_, counts) in datasets_info.items()]
    features = [stats['n_features'] for stats in feature_stats.values()]

    # 样本数量条形图
    bars1 = ax1.bar(names, samples, color='#2ecc71', alpha=0.8)
    ax1.set_title('各数据集样本数量分布')
    ax1.set_xlabel('数据集')
    ax1.set_ylabel('样本数量')
    ax1.tick_params(axis='x', rotation=45)

    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}',
                 ha='center', va='bottom')

    # 特征维度条形图
    bars2 = ax2.bar(names, features, color='#3498db', alpha=0.8)
    ax2.set_title('各数据集特征维度分布')
    ax2.set_xlabel('数据集')
    ax2.set_ylabel('特征维度')
    ax2.tick_params(axis='x', rotation=45)

    # 添加数值标签
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


def plot_class_distribution(datasets_info):
    """绘制类别分布图"""
    set_chinese_font()
    n_datasets = len(datasets_info)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    colors = plt.cm.Set3(np.linspace(0, 1, 10))

    for i, (dataset_name, (labels, counts)) in enumerate(datasets_info.items()):
        ax = axes[i]

        # 绘制饼图
        wedges, texts, autotexts = ax.pie(counts,
                                          labels=[f'类别 {int(label)}' for label in labels.flatten()],
                                          autopct='%1.1f%%',
                                          colors=colors[:len(counts)])

        ax.set_title(dataset_name.replace('.mat', ''))

    # 删除多余的子图
    if n_datasets < 6:
        for i in range(n_datasets, 6):
            fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()


def main():
    """主函数"""
    # 数据路径
    base_path = "/Users/bootscoder/Documents/岳老师-科研/贾浩轩-Adap-BDCM/paper2/Adap-BDCM-main/data"
    datasets = ["ALLAML.mat", "colon.mat", "GLI_85.mat",
                "leukemia.mat", "Prostate_GE.mat", "SMK_CAN_187.mat"]

    datasets_info = {}
    feature_stats = {}

    # 分析每个数据集
    for dataset in datasets:
        file_path = os.path.join(base_path, dataset)
        if os.path.exists(file_path):
            X, Y, labels, counts = load_and_analyze_data(file_path)
            if X is not None:
                datasets_info[dataset] = (labels, counts)
                # 计算特征统计信息
                feature_stats[dataset] = {
                    'mean': X.mean(),
                    'std': X.std(),
                    'range': (X.min(), X.max()),
                    'n_features': X.shape[1]
                }
        else:
            print(f"文件不存在: {file_path}")

    # 创建汇总表
    if datasets_info:
        print("\n创建数据集汇总表...")
        summary_table = create_dataset_summary_table(datasets_info, feature_stats)
        print("\n数据集汇总信息:")
        print(summary_table.to_string(index=False))

        # 绘制可视化图表
        print("\n生成可视化图表...")
        plot_sample_feature_distribution(datasets_info, feature_stats)
        plot_class_distribution(datasets_info)

        # 导出汇总表为CSV
        summary_table.to_csv('dataset_summary.csv', index=False, encoding='utf-8-sig')
        print("\n汇总表已导出为 dataset_summary.csv")


if __name__ == "__main__":
    main()
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 文件路径
file_path = 'data.csv'
file_name = os.path.splitext(os.path.basename(file_path))[0]  # 提取文件名（不包含扩展名）

# 读取数据
data = pd.read_csv(file_path)

# 1. 数据概要
print("数据概要:")
print(data.info())
print("\n统计摘要:")
print(data.describe())

# 2. 检查缺失值
missing_values = data.isnull().sum()
print("\n每列的缺失值数量:")
print(missing_values[missing_values > 0])

# 3. 标签分布
plt.figure(figsize=(8, 6))
sns.countplot(x='label', data=data)
plt.title(f'Label Distribution - {file_name}')
plt.xlabel('Label')
plt.ylabel('Amount')
plt.savefig(f'label_distribution_{file_name}.png')  # 动态保存文件名
plt.show()
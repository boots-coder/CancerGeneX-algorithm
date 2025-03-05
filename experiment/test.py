from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# 1. 创建简单的二分类数据集
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=2,
                           n_clusters_per_class=1, random_state=42)

# 2. 定义模型
model = RandomForestClassifier(random_state=42)

# 3. 定义五折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 4. 记录日志
fold_results = []

print("开始五折交叉验证...")

for fold, (train_index, test_index) in enumerate(kf.split(X)):
    print(f"\n第 {fold + 1} 折:")

    # 分割数据
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    print(f"训练集大小: {X_train.shape[0]}, 验证集大小: {X_test.shape[0]}")

    # 训练模型
    model.fit(X_train, y_train)
    print("模型训练完成。")

    # 验证模型
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    fold_results.append(accuracy)

    print(f"本折准确率: {accuracy:.4f}")

# 5. 输出整体结果
print("\n五折交叉验证完成。")
print("每折准确率: ", fold_results)
print(f"平均准确率: {np.mean(fold_results):.4f}")
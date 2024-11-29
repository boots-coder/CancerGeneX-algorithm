import numpy as np
import matplotlib.pyplot as plt

# 数据准备
y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 1])  # 真实标签
y_pred_prob = np.array([0.9, 0.1, 0.8, 0.6, 0.2, 0.3, 0.7, 0.5, 0.4, 0.85])  # 预测概率

# 手动指定阈值
thresholds = [0.2, 0.1, 0.3, 0.4, 0.9]

# 初始化列表存储 FPR 和 TPR
fpr_list = []
tpr_list = []

# 逐个计算每个阈值下的 FPR 和 TPR
for threshold in thresholds:
    y_pred = (y_pred_prob >= threshold).astype(int)  # 根据当前阈值分类
    TP = np.sum((y_pred == 1) & (y_true == 1))  # 真阳性
    FP = np.sum((y_pred == 1) & (y_true == 0))  # 假阳性
    TN = np.sum((y_pred == 0) & (y_true == 0))  # 真阴性
    FN = np.sum((y_pred == 0) & (y_true == 1))  # 假阴性

    # 计算 FPR 和 TPR
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0  # 避免除零错误
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    fpr_list.append(FPR)
    tpr_list.append(TPR)

# 打印计算结果
print("Thresholds:", thresholds)
print("FPR:", fpr_list)
print("TPR:", tpr_list)

# 绘制 ROC 曲线
plt.figure()
plt.plot(fpr_list, tpr_list, marker='o', linestyle='-', color='blue', label='ROC Curve')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')
plt.title('ROC Curve (Manual Thresholds)')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend(loc='lower right')
plt.show()

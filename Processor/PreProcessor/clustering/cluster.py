# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
# from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import cross_val_score
# import seaborn as sns
#
#
# class FeatureClusterExecutor:
#     def __init__(self, cluster_threshold=50, cv=5, scorer='accuracy', max_iter=1000):
#         """
#         :param cluster_threshold: fcluster使用的阈值，控制聚类颗粒度
#         :param cv: 交叉验证折数
#         :param scorer: 模型评分标准
#         :param max_iter: 逻辑回归最大迭代次数
#         """
#         self.cluster_threshold = cluster_threshold
#         self.cv = cv
#         self.scorer = scorer
#         self.max_iter = max_iter
#
#     def __call__(self, data):
#         """
#         对data['X']特征进行层次聚类，找到代表性特征，并将代表性特征信息加入data字典中。
#
#         data字典要求：
#         data['Original']['X_train'], data['Original']['y_train'] 必须有
#         如果有验证集：data['Original']['X_val']
#         如果有测试集：data['Original']['X_test']
#
#         运行结束后:
#         data['representative_features'] -> 每个簇的代表性特征索引列表
#         data['cluster_details'] -> 每个簇的详细信息字典
#         """
#         print("================= 数据池优秀代表“基因”选择 ==================")
#         X = data['Original']['X_train']
#         y = data['Original']['y_train'].ravel()
#
#         X_val = data['Original'].get('X_val', None)
#         X_test = data['Original'].get('X_test', None)
#
#         # 特征标准化
#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(X)
#
#         # 层次聚类
#         Z = linkage(X_scaled.T, method='ward', metric='euclidean')
#
#         # 根据阈值分簇
#         cluster_labels = fcluster(Z, t=self.cluster_threshold, criterion='distance')
#
#         num_clusters = len(np.unique(cluster_labels))
#         print("总共得到的特征簇数量:", num_clusters)
#
#         representative_features = []
#         cluster_details = {}
#
#         for c in np.unique(cluster_labels):
#             cluster_feature_indices = np.where(cluster_labels == c)[0]
#
#             best_score = -np.inf
#             best_feature = None
#
#             # 对每个特征进行性能评估(单特征分类)
#             for f_idx in cluster_feature_indices:
#                 X_single_feature = X_scaled[:, f_idx].reshape(-1, 1)
#                 clf = LogisticRegression(solver='lbfgs', max_iter=self.max_iter)
#                 scores = cross_val_score(clf, X_single_feature, y, cv=self.cv, scoring=self.scorer)
#                 mean_score = scores.mean()
#
#                 if mean_score > best_score:
#                     best_score = mean_score
#                     best_feature = f_idx
#
#             cluster_details[c] = {
#                 'cluster_features': cluster_feature_indices,
#                 'representative_feature': best_feature,
#                 'representative_score': best_score,
#                 'representative_feature_values_train': X[:, best_feature].reshape(-1, 1)
#             }
#
#             if X_val is not None:
#                 cluster_details[c]['representative_feature_values_val'] = X_val[:, best_feature].reshape(-1, 1)
#             if X_test is not None:
#                 cluster_details[c]['representative_feature_values_predict'] = X_test[:, best_feature].reshape(-1, 1)
#
#             representative_features.append(best_feature)
#
#         data['representative_features'] = representative_features
#         data['cluster_details'] = cluster_details
#
#         print("数据池持久化结束")
#         return data, Z, cluster_labels
#
#
# if __name__ == '__main__':
#     # =============================
#     # 这里使用随机数据来演示流程
#     # 用户可替换成真实数据
#     # =============================
#     np.random.seed(42)
#     # 假设有100条样本，每条样本50个特征（可根据需要修改）
#     X_train = np.random.rand(100, 50)
#     y_train = np.random.randint(0, 2, size=(100, 1))
#
#     data = {
#         'Original': {
#             'X_train': X_train,
#             'y_train': y_train
#         }
#     }
#
#     executor = FeatureClusterExecutor(cluster_threshold=50, cv=5, scorer='accuracy', max_iter=1000)
#     data, Z, cluster_labels = executor(data)
#
#     # 作图1：绘制层次聚类的树状图（dendrogram）
#     plt.figure(figsize=(10, 6))
#     dendrogram(Z,
#                labels=[f'F{i}' for i in range(X_train.shape[1])],
#                leaf_rotation=90,
#                leaf_font_size=8)
#     plt.title("Feature Hierarchical Clustering Dendrogram")
#     plt.xlabel("Feature Index")
#     plt.ylabel("Distance")
#     plt.tight_layout()
#     plt.savefig("dendrogram.png", dpi=300)
#     plt.close()
#     print("已保存 dendrogram.png")
#
#     # 作图2：代表性特征的评分柱状图
#     cluster_details = data['cluster_details']
#     # 提取代表特征及其评分
#     cluster_ids = list(cluster_details.keys())
#     representative_scores = [cluster_details[c]['representative_score'] for c in cluster_ids]
#     representative_features = [cluster_details[c]['representative_feature'] for c in cluster_ids]
#
#     plt.figure(figsize=(10, 6))
#     sns.barplot(x=representative_features, y=representative_scores)
#     plt.title("Representative Features and Their Scores")
#     plt.xlabel("Feature Index")
#     plt.ylabel("Cross-Validation Score")
#     plt.tight_layout()
#     plt.savefig("representative_features.png", dpi=300)
#     plt.close()
#     print("已保存 representative_features.png")
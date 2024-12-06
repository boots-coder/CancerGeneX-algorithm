import time
import scipy.io as sio
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

class FeatureClusterExecutor:
    def __init__(self, cluster_threshold=50, cv=5, scorer='accuracy', max_iter=1000):
        """
        :param cluster_threshold: 用于 fcluster 的分簇距离阈值
        :param cv: 交叉验证折数
        :param scorer: 评分标准
        :param max_iter: 逻辑回归最大迭代次数
        """
        self.cluster_threshold = cluster_threshold
        self.cv = cv
        self.scorer = scorer
        self.max_iter = max_iter

    def __call__(self, data):
        """
        对data中的X进行层次聚类，自动找到代表性特征，并将结果添加回data中。

        data中应包含:
        data['X'] -> 特征矩阵, shape (n_samples, n_features)
        data['y'] -> 标签向量, shape (n_samples,)

        更新后的data中将包含:
        data['representative_features'] -> 每个簇的代表性特征索引列表
        data['cluster_details'] -> 每个簇的详细信息（包含簇特征列表、代表性特征、该特征的评分等）
        """
        X = data['x']
        y = data['Y'].ravel()

        # 对特征进行标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 层次聚类对特征做聚类，注意特征在X中是列，因此对X_scaled.T操作
        Z = linkage(X_scaled.T, method='ward', metric='euclidean')

        # 对特征进行分簇
        cluster_labels = fcluster(Z, t=self.cluster_threshold, criterion='distance')

        # 找出有多少个簇
        num_clusters = len(np.unique(cluster_labels))
        print("总共得到的簇数量:", num_clusters)

        representative_features = []
        cluster_details = {}

        # 遍历每个簇，寻找代表性特征
        for c in np.unique(cluster_labels):
            # 找到该簇的特征索引
            cluster_feature_indices = np.where(cluster_labels == c)[0]

            best_score = -np.inf
            best_feature = None

            # 对每个特征进行性能评估
            for f_idx in cluster_feature_indices:
                X_single_feature = X_scaled[:, f_idx].reshape(-1, 1)

                clf = LogisticRegression(solver='lbfgs', max_iter=self.max_iter)
                scores = cross_val_score(clf, X_single_feature, y, cv=self.cv, scoring=self.scorer)
                mean_score = scores.mean()

                if mean_score > best_score:
                    best_score = mean_score
                    best_feature = f_idx

            # 保存该簇的信息
            cluster_details[c] = {
                'cluster_features': cluster_feature_indices,
                'representative_feature': best_feature,
                'representative_score': best_score
            }

            representative_features.append(best_feature)

        # 将结果添加回data
        data['representative_features'] = representative_features
        data['cluster_details'] = cluster_details

        return data


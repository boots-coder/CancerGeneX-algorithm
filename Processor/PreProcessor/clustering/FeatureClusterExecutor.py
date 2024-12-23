import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

class FeatureClusterExecutor:
    def __init__(self, cluster_threshold=50, cv=5, scorer='accuracy', max_iter=1000):
        """
        :param cluster_threshold: fcluster使用的阈值
        :param cv: 交叉验证折数
        :param scorer: 模型评分标准
        :param max_iter: 逻辑回归最大迭代次数
        """
        self.cluster_threshold = cluster_threshold
        self.cv = cv
        self.scorer = scorer
        self.max_iter = max_iter

    def __call__(self, data):
        """
        对data['X']特征进行层次聚类，找到代表性特征，并将代表性特征信息加入data字典中。
        要求:
        data['Original']['X_train'], data['Original']['y_train']
        若有验证集：data['Original']['X_val']
        若有测试集：data['Original']['X_test']

        运行结束后:
        data['representative_features'] -> 每个簇的代表性特征索引列表
        data['cluster_details'] -> 每个簇的详细信息字典，其中包含train/val/test对应的代表特征的值
        """
        print("=================数据池优秀代表“基因”选择==================")
        X = data['Original']['X_train']
        y = data['Original']['y_train'].ravel()

        X_val = data['Original'].get('X_val', None)
        X_test = data['Original'].get('X_test', None)

        # 特征标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 层次聚类
        Z = linkage(X_scaled.T, method='ward', metric='euclidean')

        # 分簇
        cluster_labels = fcluster(Z, t=self.cluster_threshold, criterion='distance')

        # 找出多少个簇
        num_clusters = len(np.unique(cluster_labels))
        print("总共得到的特征簇数量:", num_clusters)

        representative_features = []
        cluster_details = {}

        for c in np.unique(cluster_labels):
            cluster_feature_indices = np.where(cluster_labels == c)[0]

            best_score = -np.inf
            best_feature = None

            # 对每个特征进行性能评估(单特征分类)
            for f_idx in cluster_feature_indices:
                X_single_feature = X_scaled[:, f_idx].reshape(-1, 1)

                clf = LogisticRegression(solver='lbfgs', max_iter=self.max_iter)
                scores = cross_val_score(clf, X_single_feature, y, cv=self.cv, scoring=self.scorer)
                mean_score = scores.mean()

                if mean_score > best_score:
                    best_score = mean_score
                    best_feature = f_idx
            # 存储代表性特征信息
            # 注意这里的best_feature是特征列的索引，我们直接存储该列特征的原始值以避免后续融合过程中的越界问题。
            cluster_details[c] = {
                'cluster_features': cluster_feature_indices,
                'representative_feature': best_feature,
                'representative_score': best_score,
                'representative_feature_values_train': X[:, best_feature].reshape(-1, 1)
            }

            # 如果存在验证集或测试集，也将对应的列提取出来
            if X_val is not None:
                cluster_details[c]['representative_feature_values_val'] = X_val[:, best_feature].reshape(-1, 1)
            if X_test is not None:
                cluster_details[c]['representative_feature_values_predict'] = X_test[:, best_feature].reshape(-1, 1)

            representative_features.append(best_feature)

        # 将结果写入data
        data['representative_features'] = representative_features
        data['cluster_details'] = cluster_details
        print("数据池持久化结束")
        return data
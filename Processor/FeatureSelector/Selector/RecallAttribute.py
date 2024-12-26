import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import (
    mutual_info_classif,
    VarianceThreshold,
    f_classif,
    chi2
)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.svm import LinearSVC
from collections import Counter, defaultdict
from Processor.Common.Template import FeatureSelectorTemplate


from Processor.Common.Template import FeatureSelectorTemplate
def get_attribute_recall_method(name, kwargs):
    # 从 kwargs 中获取 'name' 的值
    kwargs_name = kwargs.get("name", None)
    if kwargs_name == "RecallAttribute":
        return RecallAttribute(name, kwargs)
    if kwargs_name == "TwoStageRecall":
        return TwoStageRecall(name, kwargs)

class RecallAttribute(FeatureSelectorTemplate):

    def __init__(self, name, configs=None):
        self.name = name
        self.recall_ratio = configs.get("RecallRatio", 0.1)
        self.default_recall_ratio = configs.get("DefualtRecallRatio", 0.1)
        self.is_encapsulated = configs.get("IsEncapsulated", True)

    def fit_executable(self, layer):
        return layer >= 2

    def fit_excecute(self, f_select_ids, f_select_infos, layer):
        assert f_select_ids != None, "当前层没有进行特征筛选模块，无法进行属性召回"

        # 总特征数量
        totall_feature_num = f_select_infos.get("Dim", None)

        f_select_num = len(f_select_ids)
        recall_ratio = self._obtain_recall_ratio(layer)
        # 进行特征召回的具体数量
        recall_num = int(recall_ratio * (totall_feature_num - f_select_num))

        all_attribute_ids = set(range(totall_feature_num))
        no_selected_ids = all_attribute_ids - set(f_select_ids)
        assert 0 <= recall_num <= len(no_selected_ids), "召回特征的数量不能超过未选择的特征数量"

        recall_ids = random.sample(list(no_selected_ids), recall_num)
        f_select_ids = recall_ids + f_select_ids

        f_select_infos["RecallNum"] = recall_num

        return f_select_ids, f_select_infos

    def _obtain_recall_ratio(self, layer=None):
        if isinstance(self.recall_ratio, float):
            assert 0 <= self.recall_ratio <= 1 , "召回的比率不在 0 - 1 之间"
            return self.recall_ratio

        if isinstance(self.recall_ratio, dict):
            current_recall_ratio = self.recall_ratio.get(layer, None)
            if current_recall_ratio == None :
                current_recall_ratio = self.recall_ratio.get("default", self.default_recall_ratio)
                print("请注意当前层的特征召回比率是默认值", self.default_recall_ratio)
            assert 0 <= current_recall_ratio <= 1, "召回的比率不在 0 - 1 之间"
            return current_recall_ratio

# 并集+交集
# class SelectorBasedRecall(FeatureSelectorTemplate):
#     """基于多个特征选择器的特征召回方法，重复特征代表更高重要性"""
#
#     def __init__(self, name, configs=None):
#         self.name = name
#         self.is_encapsulated = configs.get("IsEncapsulated", True)
#
#         # 特征选择器的参数
#         self.mi_percentile = configs.get("MutualInfoPercentile", 10)
#         self.variance_threshold = configs.get("VarianceThreshold", 0.1)
#         self.rf_importance_percentile = configs.get("RFImportancePercentile", 10)
#
#     def fit_executable(self, layer):
#         return layer >= 2
#
#     def fit_excecute(self, f_select_ids, f_select_infos, layer):
#         assert f_select_ids is not None, "当前层没有进行特征筛选模块，无法进行属性召回"
#
#         # 获取数据
#         X_train = f_select_infos.get("X_train")
#         y_train = f_select_infos.get("y_train")
#         totall_feature_num = f_select_infos.get("Dim")
#
#         # 获取未被选择的特征索引
#         all_attribute_ids = set(range(totall_feature_num))
#         no_selected_ids = list(all_attribute_ids - set(f_select_ids))
#
#         if len(no_selected_ids) == 0:
#             return f_select_ids, f_select_infos
#
#         # 构建用于特征选择的数据
#         X_unselected = X_train[:, no_selected_ids]
#
#         # 使用多个特征选择器，收集所有选择的特征（允许重复）
#         selected_features = []
#
#         # 1. 互信息特征选择
#         try:
#             mi_scores = mutual_info_classif(X_unselected, y_train)
#             mi_threshold = np.percentile(mi_scores, 100 - self.mi_percentile)
#             mi_selected = np.where(mi_scores >= mi_threshold)[0]
#             selected_features.extend(mi_selected)
#             f_select_infos["MI_Selected_Count"] = len(mi_selected)
#         except Exception as e:
#             print(f"互信息特征选择失败: {str(e)}")
#             f_select_infos["MI_Selected_Count"] = 0
#
#         # 2. 方差阈值特征选择
#         try:
#             selector = VarianceThreshold(threshold=self.variance_threshold)
#             selector.fit(X_unselected)
#             var_selected = np.where(selector.get_support())[0]
#             selected_features.extend(var_selected)
#             f_select_infos["Variance_Selected_Count"] = len(var_selected)
#         except Exception as e:
#             print(f"方差阈值特征选择失败: {str(e)}")
#             f_select_infos["Variance_Selected_Count"] = 0
#
#         # 3. 随机森林特征重要性
#         try:
#             rf = RandomForestClassifier(n_estimators=100, random_state=42)
#             rf.fit(X_unselected, y_train)
#             importances = rf.feature_importances_
#             rf_threshold = np.percentile(importances, 100 - self.rf_importance_percentile)
#             rf_selected = np.where(importances >= rf_threshold)[0]
#             selected_features.extend(rf_selected)
#             f_select_infos["RF_Selected_Count"] = len(rf_selected)
#         except Exception as e:
#             print(f"随机森林特征选择失败: {str(e)}")
#             f_select_infos["RF_Selected_Count"] = 0
#
#         # 统计每个特征被选中的次数
#         feature_counts = Counter(selected_features)
#
#         # 将特征按被选中次数排序，次数相同的特征随机排序
#         sorted_features = sorted(
#             feature_counts.items(),
#             key=lambda x: (x[1], random.random()),
#             reverse=True
#         )
#
#         # 将选择器的索引映射回原始特征空间
#         recall_ids = []
#
#         # 添加所有被选择器选中的特征，按照被选中次数排序
#         for feat_idx, _ in sorted_features:
#             recall_ids.append(no_selected_ids[feat_idx])
#
#         # 合并原有的特征选择结果和召回的特征
#         f_select_ids = recall_ids + f_select_ids
#
#         # 记录召回信息
#         f_select_infos["RecallNum"] = len(recall_ids)
#         f_select_infos["MultiSelected_Features"] = sum(1 for _, count in feature_counts.items() if count > 1)
#         f_select_infos["SingleSelected_Features"] = sum(1 for _, count in feature_counts.items() if count == 1)
#
#         return f_select_ids, f_select_infos


# 交集
# class SelectorBasedRecall(FeatureSelectorTemplate):
#     """基于多个特征选择器交集的特征召回方法"""
#
#     def __init__(self, name, configs=None):
#         self.name = name
#         self.is_encapsulated = configs.get("IsEncapsulated", True)
#
#         # 特征选择器的参数
#         self.mi_percentile = configs.get("MutualInfoPercentile", 10)
#         self.variance_threshold = configs.get("VarianceThreshold", 0.1)
#         self.rf_importance_percentile = configs.get("RFImportancePercentile", 10)
#
#     def fit_executable(self, layer):
#         return layer >= 2
#
#     def fit_excecute(self, f_select_ids, f_select_infos, layer):
#         assert f_select_ids is not None, "当前层没有进行特征筛选模块，无法进行属性召回"
#
#         # 获取数据
#         X_train = f_select_infos.get("X_train")
#         y_train = f_select_infos.get("y_train")
#         totall_feature_num = f_select_infos.get("Dim")
#
#         # 获取未被选择的特征索引
#         all_attribute_ids = set(range(totall_feature_num))
#         no_selected_ids = list(all_attribute_ids - set(f_select_ids))
#
#         if len(no_selected_ids) == 0:
#             return f_select_ids, f_select_infos
#
#         # 构建用于特征选择的数据
#         X_unselected = X_train[:, no_selected_ids]
#
#         # 存储每个选择器选择的特征集合
#         selected_feature_sets = []
#
#         # 1. 互信息特征选择
#         try:
#             mi_scores = mutual_info_classif(X_unselected, y_train)
#             mi_threshold = np.percentile(mi_scores, 100 - self.mi_percentile)
#             mi_selected = set(np.where(mi_scores >= mi_threshold)[0])
#             selected_feature_sets.append(mi_selected)
#             f_select_infos["MI_Selected_Count"] = len(mi_selected)
#         except Exception as e:
#             print(f"互信息特征选择失败: {str(e)}")
#             f_select_infos["MI_Selected_Count"] = 0
#             return f_select_ids, f_select_infos  # 如果任一选择器失败，直接返回原特征集
#
#         # 2. 方差阈值特征选择
#         try:
#             selector = VarianceThreshold(threshold=self.variance_threshold)
#             selector.fit(X_unselected)
#             var_selected = set(np.where(selector.get_support())[0])
#             selected_feature_sets.append(var_selected)
#             f_select_infos["Variance_Selected_Count"] = len(var_selected)
#         except Exception as e:
#             print(f"方差阈值特征选择失败: {str(e)}")
#             f_select_infos["Variance_Selected_Count"] = 0
#             return f_select_ids, f_select_infos
#
#         # 3. 随机森林特征重要性
#         try:
#             rf = RandomForestClassifier(n_estimators=100, random_state=42)
#             rf.fit(X_unselected, y_train)
#             importances = rf.feature_importances_
#             rf_threshold = np.percentile(importances, 100 - self.rf_importance_percentile)
#             rf_selected = set(np.where(importances >= rf_threshold)[0])
#             selected_feature_sets.append(rf_selected)
#             f_select_infos["RF_Selected_Count"] = len(rf_selected)
#         except Exception as e:
#             print(f"随机森林特征选择失败: {str(e)}")
#             f_select_infos["RF_Selected_Count"] = 0
#             return f_select_ids, f_select_infos
#
#         # 计算所有选择器的交集
#         if selected_feature_sets:
#             intersection_features = reduce(and_, selected_feature_sets)
#
#             # 将索引映射回原始特征空间
#             recall_ids = [no_selected_ids[i] for i in intersection_features]
#
#             # 合并原有的特征选择结果和召回的特征
#             f_select_ids = recall_ids + f_select_ids
#
#             # 记录召回信息
#             f_select_infos["RecallNum"] = len(recall_ids)
#             f_select_infos["IntersectionFeatures"] = len(intersection_features)
#
#             # 记录各选择器之间的两两交集大小
#             if len(selected_feature_sets) >= 2:
#                 for i in range(len(selected_feature_sets)):
#                     for j in range(i + 1, len(selected_feature_sets)):
#                         key = f"Intersection_{i + 1}_{j + 1}"
#                         f_select_infos[key] = len(selected_feature_sets[i] & selected_feature_sets[j])
#
#         return f_select_ids, f_select_infos


"""
交集
"""

# class SelectorBasedRecall(FeatureSelectorTemplate):
#     """基于多个特征选择器交集的特征召回方法"""
#
#     def __init__(self, name, configs=None):
#         self.name = name
#         self.is_encapsulated = configs.get("IsEncapsulated", True)
#
#         # 特征选择器的参数
#         selector_configs = configs.get("SelectorConfigs", {})
#         self.mi_percentile = selector_configs.get("MutualInfoPercentile", 10)
#         self.variance_threshold = selector_configs.get("VarianceThreshold", 0.1)
#         self.rf_importance_percentile = selector_configs.get("RFImportancePercentile", 10)
#         self.f_score_percentile = selector_configs.get("FScorePercentile", 10)
#         self.chi2_percentile = selector_configs.get("Chi2Percentile", 10)
#         self.correlation_percentile = selector_configs.get("CorrelationPercentile", 10)
#         self.lasso_alpha = selector_configs.get("LassoAlpha", 0.01)
#         self.gbdt_importance_percentile = selector_configs.get("GBDTImportancePercentile", 10)
#         self.l1_svc_percentile = selector_configs.get("L1SVCPercentile", 10)
#         self.logistic_l1_percentile = selector_configs.get("LogisticL1Percentile", 10)
#
#     def fit_executable(self, layer):
#         return layer >= 2
#
#     def fit_excecute(self, f_select_ids, f_select_infos, layer):
#         assert f_select_ids is not None, "当前层没有进行特征筛选模块，无法进行属性召回"
#
#         # 获取数据
#         X_train = f_select_infos.get("X_train")
#         y_train = f_select_infos.get("y_train")
#         totall_feature_num = f_select_infos.get("Dim")
#
#         # 获取未被选择的特征索引
#         all_attribute_ids = set(range(totall_feature_num))
#         no_selected_ids = list(all_attribute_ids - set(f_select_ids))
#
#         if len(no_selected_ids) == 0:
#             return f_select_ids, f_select_infos
#
#         # 构建用于特征选择的数据
#         X_unselected = X_train[:, no_selected_ids]
#
#         # 对数据进行归一化，用于某些需要归一化的特征选择器
#         scaler = MinMaxScaler()
#         X_normalized = scaler.fit_transform(X_unselected)
#
#         # 存储每个选择器选择的特征集合
#         selected_feature_sets = []
#         selector_names = []
#
#         # 1. 互信息特征选择
#         try:
#             mi_scores = mutual_info_classif(X_unselected, y_train)
#             mi_threshold = np.percentile(mi_scores, 100 - self.mi_percentile)
#             mi_selected = set(np.where(mi_scores >= mi_threshold)[0])
#             selected_feature_sets.append(mi_selected)
#             selector_names.append("MutualInfo")
#             f_select_infos["MI_Selected_Count"] = len(mi_selected)
#         except Exception as e:
#             print(f"互信息特征选择失败: {str(e)}")
#
#         # 2. 方差阈值特征选择
#         try:
#             selector = VarianceThreshold(threshold=self.variance_threshold)
#             selector.fit(X_unselected)
#             var_selected = set(np.where(selector.get_support())[0])
#             selected_feature_sets.append(var_selected)
#             selector_names.append("Variance")
#             f_select_infos["Variance_Selected_Count"] = len(var_selected)
#         except Exception as e:
#             print(f"方差阈值特征选择失败: {str(e)}")
#
#         # 3. 随机森林特征重要性
#         try:
#             rf = RandomForestClassifier(n_estimators=100, random_state=42)
#             rf.fit(X_unselected, y_train)
#             importances = rf.feature_importances_
#             rf_threshold = np.percentile(importances, 100 - self.rf_importance_percentile)
#             rf_selected = set(np.where(importances >= rf_threshold)[0])
#             selected_feature_sets.append(rf_selected)
#             selector_names.append("RandomForest")
#             f_select_infos["RF_Selected_Count"] = len(rf_selected)
#         except Exception as e:
#             print(f"随机森林特征选择失败: {str(e)}")
#
#         # 4. F-score特征选择
#         try:
#             f_scores, _ = f_classif(X_unselected, y_train)
#             f_threshold = np.percentile(f_scores, 100 - self.f_score_percentile)
#             f_selected = set(np.where(f_scores >= f_threshold)[0])
#             selected_feature_sets.append(f_selected)
#             selector_names.append("FScore")
#             f_select_infos["FScore_Selected_Count"] = len(f_selected)
#         except Exception as e:
#             print(f"F-score特征选择失败: {str(e)}")
#
#         # 5. 卡方检验特征选择
#         try:
#             chi2_scores, _ = chi2(X_normalized, y_train)
#             chi2_threshold = np.percentile(chi2_scores, 100 - self.chi2_percentile)
#             chi2_selected = set(np.where(chi2_scores >= chi2_threshold)[0])
#             selected_feature_sets.append(chi2_selected)
#             selector_names.append("Chi2")
#             f_select_infos["Chi2_Selected_Count"] = len(chi2_selected)
#         except Exception as e:
#             print(f"卡方检验特征选择失败: {str(e)}")
#
#         # 6. LASSO特征选择
#         try:
#             lasso = Lasso(alpha=self.lasso_alpha, random_state=42)
#             lasso.fit(X_normalized, y_train)
#             lasso_selected = set(np.where(np.abs(lasso.coef_) > 0)[0])
#             selected_feature_sets.append(lasso_selected)
#             selector_names.append("Lasso")
#             f_select_infos["Lasso_Selected_Count"] = len(lasso_selected)
#         except Exception as e:
#             print(f"LASSO特征选择失败: {str(e)}")
#
#         # 7. GBDT特征重要性
#         try:
#             gbdt = GradientBoostingClassifier(n_estimators=100, random_state=42)
#             gbdt.fit(X_unselected, y_train)
#             importances = gbdt.feature_importances_
#             gbdt_threshold = np.percentile(importances, 100 - self.gbdt_importance_percentile)
#             gbdt_selected = set(np.where(importances >= gbdt_threshold)[0])
#             selected_feature_sets.append(gbdt_selected)
#             selector_names.append("GBDT")
#             f_select_infos["GBDT_Selected_Count"] = len(gbdt_selected)
#         except Exception as e:
#             print(f"GBDT特征选择失败: {str(e)}")
#
#         # 8. L1-SVM特征选择
#         try:
#             svc = LinearSVC(penalty='l1', dual=False, random_state=42)
#             svc.fit(X_normalized, y_train)
#             l1_svc_selected = set(np.where(np.abs(svc.coef_[0]) > 0)[0])
#             selected_feature_sets.append(l1_svc_selected)
#             selector_names.append("L1SVC")
#             f_select_infos["L1SVC_Selected_Count"] = len(l1_svc_selected)
#         except Exception as e:
#             print(f"L1-SVM特征选择失败: {str(e)}")
#
#         # 9. L1正则化Logistic回归特征选择
#         try:
#             lr = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
#             lr.fit(X_normalized, y_train)
#             lr_selected = set(np.where(np.abs(lr.coef_[0]) > 0)[0])
#             selected_feature_sets.append(lr_selected)
#             selector_names.append("LogisticL1")
#             f_select_infos["LogisticL1_Selected_Count"] = len(lr_selected)
#         except Exception as e:
#             print(f"L1正则化Logistic回归特征选择失败: {str(e)}")
#
#         # 计算所有选择器的交集
#         if selected_feature_sets:
#             intersection_features = reduce(and_, selected_feature_sets)
#
#             # 将索引映射回原始特征空间
#             recall_ids = [no_selected_ids[i] for i in intersection_features]
#
#             # 合并原有的特征选择结果和召回的特征
#             f_select_ids = recall_ids + f_select_ids
#
#             # 记录召回信息
#             f_select_infos["RecallNum"] = len(recall_ids)
#             f_select_infos["IntersectionFeatures"] = len(intersection_features)
#             f_select_infos["ActiveSelectors"] = len(selected_feature_sets)
#             f_select_infos["SelectorNames"] = selector_names
#
#             # 记录各选择器之间的两两交集大小
#             if len(selected_feature_sets) >= 2:
#                 for i in range(len(selected_feature_sets)):
#                     for j in range(i + 1, len(selected_feature_sets)):
#                         key = f"Intersection_{selector_names[i]}_{selector_names[j]}"
#                         f_select_infos[key] = len(selected_feature_sets[i] & selected_feature_sets[j])
#
#         return f_select_ids, f_select_infos

"""
vote
"""


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=64):
        super(AutoEncoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim * 2),
            nn.ReLU(),
            nn.Linear(encoding_dim * 2, encoding_dim),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim * 2),
            nn.ReLU(),
            nn.Linear(encoding_dim * 2, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_encoded_features(self, x):
        return self.encoder(x)

class TwoStageRecall(FeatureSelectorTemplate):
    """两阶段特征召回方法：随机选择 + 多选择器投票"""

    def __init__(self, name, configs=None):
        self.name = name
        self.is_encapsulated = configs.get("IsEncapsulated", True)

        # 第一阶段：随机选择参数
        self.random_select_ratio = configs.get("RandomSelectRatio", 0.3)  # 随机选择的特征比例

        # 第二阶段：特征选择器参数
        selector_configs = configs.get("SelectorConfigs", {})
        # 有监督方法参数
        self.mi_percentile = selector_configs.get("MutualInfoPercentile", 10)
        self.variance_threshold = selector_configs.get("VarianceThreshold", 0.1)
        self.rf_importance_percentile = selector_configs.get("RFImportancePercentile", 10)
        self.f_score_percentile = selector_configs.get("FScorePercentile", 10)
        self.chi2_percentile = selector_configs.get("Chi2Percentile", 10)
        self.lasso_alpha = selector_configs.get("LassoAlpha", 0.01)
        self.gbdt_importance_percentile = selector_configs.get("GBDTImportancePercentile", 10)

        # 无监督方法参数
        self.pca_variance_ratio = selector_configs.get("PCAVarianceRatio", 0.9)
        self.ae_encoding_dim = selector_configs.get("AEEncodingDim", 64)
        self.ae_epochs = selector_configs.get("AEEpochs", 50)
        self.ae_batch_size = selector_configs.get("AEBatchSize", 256)
        self.ae_learning_rate = selector_configs.get("AELearningRate", 0.001)
        self.ae_reconstruction_threshold = selector_configs.get("AEReconstructionThreshold", 0.1)

        # 投票阈值
        self.min_votes = selector_configs.get("MinVotes", 2)

    def train_autoencoder(self, X, input_dim):
        """训练AutoEncoder并返回重要特征"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 创建和训练AutoEncoder
        model = AutoEncoder(input_dim, self.ae_encoding_dim).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.ae_learning_rate)

        X_tensor = torch.FloatTensor(X).to(device)
        dataset = torch.utils.data.TensorDataset(X_tensor, X_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.ae_batch_size, shuffle=True
        )

        model.train()
        for epoch in range(self.ae_epochs):
            total_loss = 0
            for batch_x, _ in dataloader:
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            reconstructed = model(X_tensor)
            reconstruction_errors = torch.mean(
                torch.abs(X_tensor - reconstructed), dim=0
            ).cpu().numpy()

        important_features = np.where(
            reconstruction_errors < np.percentile(
                reconstruction_errors,
                self.ae_reconstruction_threshold * 100
            )
        )[0]

        return important_features, reconstruction_errors

    def fit_executable(self, layer):
        return layer >= 2

    def fit_excecute(self, f_select_ids, f_select_infos, layer):
        assert f_select_ids is not None, "当前层没有进行特征筛选模块，无法进行属性召回"

        # 获取数据
        X_train = f_select_infos.get("X_train")
        y_train = f_select_infos.get("y_train")
        totall_feature_num = f_select_infos.get("Dim")
        feature_names = f_select_infos.get("FeatureNames", list(range(totall_feature_num)))

        # 获取未被选择的特征索引
        all_attribute_ids = set(range(totall_feature_num))
        no_selected_ids = list(all_attribute_ids - set(f_select_ids))

        if len(no_selected_ids) == 0:
            return f_select_ids, f_select_infos

        # === 第一阶段：随机选择 ===
        random_select_num = int(len(no_selected_ids) * self.random_select_ratio)
        random_selected_ids = random.sample(no_selected_ids, random_select_num)

        # 构建用于特征选择的数据
        X_random_selected = X_train[:, random_selected_ids]

        # 数据标准化
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X_random_selected)

        # === 第二阶段：多选择器投票 ===
        selected_features = []
        selector_names = []
        selector_results = defaultdict(dict)

        # 1. 互信息特征选择
        try:
            mi_scores = mutual_info_classif(X_random_selected, y_train)
            mi_threshold = np.percentile(mi_scores, 100 - self.mi_percentile)
            mi_selected = np.where(mi_scores >= mi_threshold)[0]
            selected_features.extend(mi_selected)
            selector_names.append("MutualInfo")

            selector_results["MutualInfo"] = {
                "selected_features": mi_selected,
                "importance_scores": mi_scores,
                "threshold": mi_threshold
            }
            f_select_infos["MI_Selected_Count"] = len(mi_selected)
        except Exception as e:
            print(f"互信息特征选择失败: {str(e)}")

        # 2. 方差阈值特征选择
        try:
            selector = VarianceThreshold(threshold=self.variance_threshold)
            selector.fit(X_random_selected)
            var_selected = np.where(selector.get_support())[0]
            selected_features.extend(var_selected)
            selector_names.append("Variance")

            selector_results["Variance"] = {
                "selected_features": var_selected,
                "variances": selector.variances_,
                "threshold": self.variance_threshold
            }
            f_select_infos["Variance_Selected_Count"] = len(var_selected)
        except Exception as e:
            print(f"方差阈值特征选择失败: {str(e)}")

        # 3. 随机森林特征重要性
        try:
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_random_selected, y_train)
            importances = rf.feature_importances_
            rf_threshold = np.percentile(importances, 100 - self.rf_importance_percentile)
            rf_selected = np.where(importances >= rf_threshold)[0]
            selected_features.extend(rf_selected)
            selector_names.append("RandomForest")

            selector_results["RandomForest"] = {
                "selected_features": rf_selected,
                "importance_scores": importances,
                "threshold": rf_threshold
            }
            f_select_infos["RF_Selected_Count"] = len(rf_selected)
        except Exception as e:
            print(f"随机森林特征选择失败: {str(e)}")

        # 4. PCA特征选择
        try:
            pca = PCA(n_components=self.pca_variance_ratio, svd_solver='full')
            pca.fit(X_normalized)

            feature_importance = np.sum(np.abs(pca.components_), axis=0)
            pca_threshold = np.percentile(feature_importance, 100 - self.mi_percentile)
            pca_selected = np.where(feature_importance >= pca_threshold)[0]

            selected_features.extend(pca_selected)
            selector_names.append("PCA")

            selector_results["PCA"] = {
                "selected_features": pca_selected,
                "importance_scores": feature_importance,
                "threshold": pca_threshold,
                "explained_variance_ratio": pca.explained_variance_ratio_,
                "n_components": pca.n_components_
            }
            f_select_infos["PCA_Selected_Count"] = len(pca_selected)
        except Exception as e:
            print(f"PCA特征选择失败: {str(e)}")

        # 5. AutoEncoder特征选择
        try:
            ae_selected, reconstruction_errors = self.train_autoencoder(
                X_normalized, X_random_selected.shape[1]
            )
            selected_features.extend(ae_selected)
            selector_names.append("AutoEncoder")

            selector_results["AutoEncoder"] = {
                "selected_features": ae_selected,
                "reconstruction_errors": reconstruction_errors,
                "threshold": np.percentile(
                    reconstruction_errors,
                    self.ae_reconstruction_threshold * 100
                )
            }
            f_select_infos["AE_Selected_Count"] = len(ae_selected)
        except Exception as e:
            print(f"AutoEncoder特征选择失败: {str(e)}")

        # 统计投票结果
        if selected_features:
            feature_votes = Counter(selected_features)

            # 获取被足够多选择器选中的特征
            final_selected_features = [feat for feat, votes in feature_votes.items()
                                       if votes >= self.min_votes]

            # 将索引映射回原始特征空间
            recall_ids = [random_selected_ids[i] for i in final_selected_features]

            # 获取召回特征的原始数据
            X_recalled = X_train[:, recall_ids]

            # 更新特征选择结果
            f_select_ids = recall_ids + f_select_ids

            # 构建每个召回特征的详细信息
            recalled_feature_info = []
            for feat_idx in final_selected_features:
                feat_info = {
                    "original_index": random_selected_ids[feat_idx],
                    "feature_name": feature_names[random_selected_ids[feat_idx]],
                    "vote_count": feature_votes[feat_idx],
                    "selected_by": []
                }

                # 记录每个特征被哪些选择器选中
                for selector_name, result in selector_results.items():
                    if feat_idx in result["selected_features"]:
                        feat_info["selected_by"].append(selector_name)
                        if "importance_scores" in result:
                            feat_info[f"{selector_name}_score"] = result["importance_scores"][feat_idx]
                        elif "reconstruction_errors" in result:
                            feat_info[f"{selector_name}_error"] = result["reconstruction_errors"][feat_idx]
                        elif "variances" in result:
                            feat_info[f"{selector_name}_variance"] = result["variances"][feat_idx]

                recalled_feature_info.append(feat_info)

            # 记录召回信息
            f_select_infos.update({
                "RandomSelectedNum": random_select_num,
                "FinalRecallNum": len(recall_ids),
                "ActiveSelectors": len(selector_names),
                "SelectorNames": selector_names,
                "RecalledFeatures": recalled_feature_info,
                "RecalledData": X_recalled,
                "SelectorResults": selector_results
            })

            # 记录投票统计信息
            vote_counts = Counter(feature_votes.values())
            for votes, count in vote_counts.items():
                f_select_infos[f"Features_With_{votes}_Votes"] = count

        return f_select_ids, f_select_infos
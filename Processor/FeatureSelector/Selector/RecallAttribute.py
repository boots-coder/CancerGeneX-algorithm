import random

from Processor.Common.Template import FeatureSelectorTemplate

import numpy as np
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from Processor.Common.Template import FeatureSelectorTemplate

def get_attribute_recall_method(name, kwargs):
    # 从 kwargs 中获取 'name' 的值
    kwargs_name = kwargs.get("name", None)
    if kwargs_name == "RecallAttribute":
        return RecallAttribute(name, kwargs)
    if kwargs_name == "SelectorBasedRecall":
        return SelectorBasedRecall(name, kwargs)

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


class SelectorBasedRecall(FeatureSelectorTemplate):
    """基于多个特征选择器的特征召回方法，重复特征代表更高重要性"""

    def __init__(self, name, configs=None):
        self.name = name
        self.is_encapsulated = configs.get("IsEncapsulated", True)

        # 特征选择器的参数
        self.mi_percentile = configs.get("MutualInfoPercentile", 10)
        self.variance_threshold = configs.get("VarianceThreshold", 0.1)
        self.rf_importance_percentile = configs.get("RFImportancePercentile", 10)

    def fit_executable(self, layer):
        return layer >= 2

    def fit_excecute(self, f_select_ids, f_select_infos, layer):
        assert f_select_ids is not None, "当前层没有进行特征筛选模块，无法进行属性召回"

        # 获取数据
        X_train = f_select_infos.get("X_train")
        y_train = f_select_infos.get("y_train")
        totall_feature_num = f_select_infos.get("Dim")

        # 获取未被选择的特征索引
        all_attribute_ids = set(range(totall_feature_num))
        no_selected_ids = list(all_attribute_ids - set(f_select_ids))

        if len(no_selected_ids) == 0:
            return f_select_ids, f_select_infos

        # 构建用于特征选择的数据
        X_unselected = X_train[:, no_selected_ids]

        # 使用多个特征选择器，收集所有选择的特征（允许重复）
        selected_features = []

        # 1. 互信息特征选择
        try:
            mi_scores = mutual_info_classif(X_unselected, y_train)
            mi_threshold = np.percentile(mi_scores, 100 - self.mi_percentile)
            mi_selected = np.where(mi_scores >= mi_threshold)[0]
            selected_features.extend(mi_selected)
            f_select_infos["MI_Selected_Count"] = len(mi_selected)
        except Exception as e:
            print(f"互信息特征选择失败: {str(e)}")
            f_select_infos["MI_Selected_Count"] = 0

        # 2. 方差阈值特征选择
        try:
            selector = VarianceThreshold(threshold=self.variance_threshold)
            selector.fit(X_unselected)
            var_selected = np.where(selector.get_support())[0]
            selected_features.extend(var_selected)
            f_select_infos["Variance_Selected_Count"] = len(var_selected)
        except Exception as e:
            print(f"方差阈值特征选择失败: {str(e)}")
            f_select_infos["Variance_Selected_Count"] = 0

        # 3. 随机森林特征重要性
        try:
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_unselected, y_train)
            importances = rf.feature_importances_
            rf_threshold = np.percentile(importances, 100 - self.rf_importance_percentile)
            rf_selected = np.where(importances >= rf_threshold)[0]
            selected_features.extend(rf_selected)
            f_select_infos["RF_Selected_Count"] = len(rf_selected)
        except Exception as e:
            print(f"随机森林特征选择失败: {str(e)}")
            f_select_infos["RF_Selected_Count"] = 0

        # 统计每个特征被选中的次数
        feature_counts = Counter(selected_features)

        # 将特征按被选中次数排序，次数相同的特征随机排序
        sorted_features = sorted(
            feature_counts.items(),
            key=lambda x: (x[1], random.random()),
            reverse=True
        )

        # 将选择器的索引映射回原始特征空间
        recall_ids = []

        # 添加所有被选择器选中的特征，按照被选中次数排序
        for feat_idx, _ in sorted_features:
            recall_ids.append(no_selected_ids[feat_idx])

        # 合并原有的特征选择结果和召回的特征
        f_select_ids = recall_ids + f_select_ids

        # 记录召回信息
        f_select_infos["RecallNum"] = len(recall_ids)
        f_select_infos["MultiSelected_Features"] = sum(1 for _, count in feature_counts.items() if count > 1)
        f_select_infos["SingleSelected_Features"] = sum(1 for _, count in feature_counts.items() if count == 1)

        return f_select_ids, f_select_infos

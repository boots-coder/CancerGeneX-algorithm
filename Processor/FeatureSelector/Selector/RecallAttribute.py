import random

from Processor.Common.Template import FeatureSelectorTemplate


def get_attribute_recall_method(name, kwargs):
    if name == "RecallAttribute":
        return RecallAttribute(name, kwargs)


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
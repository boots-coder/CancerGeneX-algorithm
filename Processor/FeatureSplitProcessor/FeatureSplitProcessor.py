import numpy as np

from Processor.Common.Template import FeatureSplitProcessorTemplate


class FeatureSplitProcessorWrapper(FeatureSplitProcessorTemplate):

    def __init__(self, name, est_type, config):
        self.name= name

    def executable(self, layer):
        return True

    def fit_excecute(self, original_train, original_val, layer):
       pass

    def split_method(self, original_X):
        pass


def get_lobal_feature_split_processor(name, est_type, config):
    if est_type == "AverageSplit":
        return AverageSplitProcessor(name, est_type, config)

class AverageSplitProcessor(FeatureSplitProcessorWrapper):
    def __init__(self, name, est_type, config):
        self.name = name
        self.est_type = est_type
        self.split_num = config.get("SplitNum", 3)

    def fit_excecute(self, original_train, original_val, layer):
        dim = original_train.shape[1]
        if dim > self.split_num :
            split_train = self.split_method(original_train, layer)
            split_val = self.split_method(original_val, layer)
            split_finfo = dict(isSuccess=True, FailureInfo=None)
            return split_train, split_val, split_finfo
        else:
            split_finfo = dict(isSuccess=False, FailureInfo="数据的维度应该超过局部变量, 此时不使用特征划分方式")
            return None, None, split_finfo

    def predict_execute(self, original_X, layer):
        dim = original_X.shape[1]
        if dim > self.split_num:
            split_X = self.split_method(original_X, layer)
            split_finfo = dict(isSuccess=True, FailureInfo=None)
            return split_X, split_finfo
        else:
            split_finfo = dict(isSuccess=False, FailureInfo="数据的维度应该超过局部变量, 此时不使用特征划分方式")
            return None, split_finfo

    # todo 解释性 按照功能进行划分- 或更好
    def split_method(self, original_X, layer):
        split_X = np.array_split(original_X,  self.split_num,  axis=1)
        return split_X
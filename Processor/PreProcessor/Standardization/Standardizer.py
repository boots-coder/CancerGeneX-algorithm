import numpy as np

from Processor.Common.Template import PreProcessorTemplate


def get_pre_standardization_processor(name, est_type, configs):
    est_method = configs.get("Method")
    if est_method == "MinMax":
        return MinMaxFeaturesPreProcessor(name,  configs)
    elif est_method == "Zscore":
        return ZscoreFeaturesProcessor(name,  configs)
    elif est_method == "Robust":
        return RobustFeaturesProcessor(name, configs)
    elif est_method == "UnitVector":
        return UnitVectorFeaturesProcessor(name, configs)
    elif est_method == "DecimalScale":
        return DecimalScaleFeaturesProcessor(name, configs)
    else:
        raise ""

class StandardizationPreProcessorWrapper(PreProcessorTemplate):

    def __init__(self, name, config):
        self.name = name
        self.est_type = config.get("EstType", [])
        self.builder_type = config.get("BuilderType", [])
        self.feat_type = config.get("FeaturesType", [])

    def executable(self):
        return True

    def fit_excecute(self, datas):
        proceesed_data = datas["Original"]
        features_train = proceesed_data["X_train"]
        #todo 这里方法是空， 可以方便我们后期对数据进行预处理；；
        #todo 现在写的乱七八糟的
        featutres_train = self.excecute_feature_processor(features_train)
        datas["Original"]["X_train"] = featutres_train

        features_val = proceesed_data.get("X_val", None)
        if features_val is not None:
            featutres_val = self.excecute_feature_processor(features_val)
            datas["Original"]["X_val"] = featutres_val

        return datas

    def predict_excecute(self, datas):
        proceesed_data = datas["Original"]
        features_X = proceesed_data["X"]
        featutres_X = self.excecute_feature_processor(features_X)
        datas["Original"]["X"] = featutres_X
        return datas

    def excecute_feature_processor(self, feature):
        pass



class MinMaxFeaturesPreProcessor(StandardizationPreProcessorWrapper):

    def excecute_feature_processor(self, feature):
        feature_min = feature.min()
        feature_max = feature.max()
        feature = (feature - feature_min) / (feature_max - feature_min)
        return feature

    def modify_features(self, feature, fmin, fmax):
        gap = fmax - fmin
        if (gap - 0.0) <= (10 ** -5):
            feature = (feature - fmin)
        else:
            feature = (feature - fmin) / (fmax - fmin)
        return feature

class ZscoreFeaturesProcessor(StandardizationPreProcessorWrapper):

    def excecute_feature_processor(self, feature):
        mean = np.mean(feature)
        std_dev = np.std(feature)

        feature = (feature - mean) / std_dev
        return feature

    def modify_features(self, feature, mean, std_dev):
        if (std_dev - 0.0) <= (10 ** -5):
            feature = feature - mean
        else:
            feature = (feature - mean) / std_dev
        return feature

class RobustFeaturesProcessor(StandardizationPreProcessorWrapper):

    def excecute_feature_processor(self, feature):

        median = np.median(feature)
        q1 = np.percentile(feature, 25)
        q3 = np.percentile(feature, 75)

        feature = self.modify_features(feature, median, q1, q3)
        return feature

    def modify_features(self, feature, median, q1, q3):
        gap = q3 - q1
        if (gap - 0.0) <= (10 ** -5):
            feature = feature - median
        else:
            feature = (feature - median) / gap
        return feature

class UnitVectorFeaturesProcessor(StandardizationPreProcessorWrapper):
    def excecute_feature_processor(self, feature):

        norm = np.linalg.norm(feature)

        feature = self.modify_features(feature, norm)
        return feature

    def modify_features(self, feature, norm):

        if (norm - 0.0) <= (10 ** -5):
            feature = feature
        else:
            feature = feature / norm
        return feature

class DecimalScaleFeaturesProcessor(StandardizationPreProcessorWrapper):
    def excecute_feature_processor(self, feature):

        max_abs = np.max(np.abs(feature))

        feature = self.modify_features(feature, max_abs)
        return feature

    def modify_features(self, feature, max_abs):

        if (max_abs - 0.0) <= (10 ** -5):
            feature = feature
        else:
            feature = feature / max_abs
        return feature
import numpy as np
from Processor.Common.Template import FeaturesProcessorTemplate

def get_standardizer(name, est_type, configs):
    est_method = configs.get("Method")
    if est_method == "MinMax":
        return MinMaxFeaturesProcessor(name,  configs)
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

class FeaturesProcessorWrapper(FeaturesProcessorTemplate):

    def __init__(self, name, config):
        self.name = name
        self.est_type = config.get("EstType", [])
        self.builder_type = config.get("BuilderType", [])
        self.feat_type = config.get("FeaturesType", [])

    def executable(self, layer):
        return True

    def fit_excecute(self, finfos, layer):
        for name, finfo in finfos.items():
            if self.determine_processor_finfos(name, finfo, layer):

                features_train = finfo["Feature_train"]
                features_train = self.excecute_feature_processor(features_train)
                finfo["Feature_train"] = features_train

                features_val = finfo.get("Feature_val")
                if features_val is not None:
                    features_val = self.excecute_feature_processor(features_val)
                    finfo["Feature_val"] = features_val

        return finfos

    def predict_excecute(self, finfos, layer):
        for name, finfo in finfos.items():
            if self.determine_processor_finfos(name, finfo, layer):
                features_X = finfo.get("Feature_X")
                features_X = self.excecute_feature_processor(features_X)
                finfo["Feature_X"] = features_X
        return finfos

    def determine_processor_finfos(self, name, finfo, layer):
        if finfo.get("BuilderType") in self.builder_type:
            return True
        if finfo.get("EstType") in self.feat_type:
            return True
        return False

    def excecute_feature_processor(self, features_X):
        pass


class MinMaxFeaturesProcessor(FeaturesProcessorWrapper):

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

class ZscoreFeaturesProcessor(FeaturesProcessorWrapper):

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

class RobustFeaturesProcessor(FeaturesProcessorWrapper):

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

class UnitVectorFeaturesProcessor(FeaturesProcessorWrapper):
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

class DecimalScaleFeaturesProcessor(FeaturesProcessorWrapper):
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
import numpy as np

from Processor.Common.Template import FeatFusionTemplate

def get_feature_concatenation_method(name, est_type, configs):
    if est_type == "FeatureConcatenation":
        return FeatureConcatenation(name, est_type, configs)
    else:
        raise "不支持其他向量拼接方法"

class FeatureConcatenation(FeatFusionTemplate):

    def __init__(self, name, est_type, configs):
        self.name = name
        self.est_type = est_type
        self.builder_type = configs.get("BuilderType", [])
        self.est_type = configs.get("EstType", [])
        self.data_type = configs.get("DataType", [])
        self.components = dict()

    def executable(self, layer):
        return layer >= 2

    def obtain_finfos_by_layers(self, finfos, layer):
        need_layer = layer - 1
        return finfos.get(need_layer)

    def obtain_need_finfos(self, finfos_layers):
        need_finfos = []
        for name, finfo in finfos_layers.items():
            if finfo.get("BuilderType") in self.builder_type:
                need_finfos.append(finfo)
                continue
            if finfo.get("EstType") in self.est_type:
                need_finfos.append(finfo)
                continue
            if finfo.get("DataType") in self.data_type:
                need_finfos.append(finfo)
                continue
        return need_finfos

    def obtain_need_train_finfos(self, f_infos):
        need_f_infos = []
        for f_info in f_infos:
            if f_info.get("Feature_train") is not None:
                need_f_infos.append(f_info)
        return need_f_infos

    def obtain_need_predict_finfos(self, finfos):
        need_finfos = []
        for finfo in finfos:
            if finfo.get("Feature_X") is not None:
                need_finfos.append(finfo)
        return need_finfos

    def obtain_final_fit_features(self, need_finfos):
        final_features_train, final_features_val = [], []
        for need_finfo in need_finfos:
            final_features_train.append(need_finfo.get("Feature_train"))
            final_features_val.append(need_finfo.get("Feature_val"))
        final_features_train = np.concatenate(final_features_train, axis=1)
        final_features_val = np.concatenate(final_features_val, axis=1)
        return final_features_train, final_features_val

    def fit_excecute(self, original_train, original_val, f_infos, layer):

        f_infos_layers = self.obtain_finfos_by_layers(f_infos, layer)
        need_f_infos = self.obtain_need_finfos(f_infos_layers)
        need_f_infos = self.obtain_need_train_finfos(need_f_infos)
        features_train, features_val = self.obtain_final_fit_features(need_f_infos)

        fusions_train = np.concatenate([original_train, features_train], axis=1)
        fusions_val = np.concatenate([original_val, features_val], axis=1)

        return fusions_train, fusions_val

    def obtain_final_predict_features(self, need_finfos):
        final_features_X = []
        for need_finfo in need_finfos:
            final_features_X.append(need_finfo.get("Feature_X"))
        final_features_X = np.concatenate(final_features_X, axis=1)
        return final_features_X

    def predict_excecute(self, X, finfos, layer):

        finfos_layers = self.obtain_finfos_by_layers(finfos, layer)
        need_finfos = self.obtain_need_finfos(finfos_layers)
        need_finfos = self.obtain_need_predict_finfos(need_finfos)

        features_X = self.obtain_final_predict_features(need_finfos)
        fusions_X = np.concatenate([X, features_X], axis=1)

        return fusions_X

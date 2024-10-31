import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from Processor.Common.Template import MetricProcessorTemplate


def get_metric_processor(name, type, configs):
    if type == "AvgMetricProcessor":
        return AvgMetricProcessor(name, type, configs)
    else:
        raise "不支持其他向量拼接方法"

class MetricProcessorWrapper(MetricProcessorTemplate):

    def __init__(self, name, est_type, configs):
        self.name = name
        self.type = est_type
        self.classifier_method = configs.get("MetricMethod", "acc")
        self.builder_type = configs.get("BuilderType", [])
        self.est_type = configs.get("EstType", [])
        self.data_type = configs.get("DateType", [])
        assert self.classifier_method != None, "分类器的方法设置不能为空"

    def executable(self, layer):
        return True

    def obtain_need_finfos(self, finfos_layers):
        need_finfos = []
        for name, finfo in finfos_layers.items():
            if finfo.get("Probs") is None:
                continue
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


    def obtain_finfos_by_layers(self, finfos, layer):
        return finfos.get(layer)

    def obtain_final_probs(self, need_finfos):
        pass

    def calucate_metric(self, x1, x2, method_name):
        if isinstance(self.classifier_method, str):
            if method_name.lower() in ["accuracy", "acc"]:
                return accuracy_score(x1, x2)
            if method_name.lower() in ["precision", "pre"]:
                return precision_score(x1, x2)
            if method_name.lower() in ["recall"]:
                return recall_score(x1, x2)
            if method_name.lower() in ["f1_score", "f1", "f1-score"]:
                return f1_score(x1, x2)
        elif callable(self.classifier_method):
            return self.classifier_method(x1, x2)

class AvgMetricProcessor(MetricProcessorWrapper):

    def fit_excecute(self, data, layer):
        finfos = data.get("Finfos")
        y_val = data.get("Original").get("y_val")

        finfos_layers = self.obtain_finfos_by_layers(finfos, layer)
        # 按照一定要求筛选一些finfos
        need_finfos = self.obtain_need_finfos(finfos_layers)
        # 获得最终的预测概率值
        final_probs = self.obtain_final_probs(need_finfos)

        final_preds = np.argmax(final_probs, axis=1)
        metric = self.calucate_metric(final_preds, y_val, self.classifier_method)
        return metric

    def obtain_final_probs(self, need_finfos):
        final_probs = []
        for need_finfo in need_finfos:
            final_probs.append(need_finfo.get("Probs"))
        return np.mean(final_probs, axis=0)

    def predict_execute(self, data, layer):
        finfos = data.get("Finfos")

        finfos_layers = self.obtain_finfos_by_layers(finfos, layer)
        # 按照一定要求筛选一些finfos
        need_finfos = self.obtain_need_finfos(finfos_layers)
        # 获得最终的预测概率值
        final_probs = self.obtain_final_probs(need_finfos)

        return final_probs

class WeightMetricProcessor(MetricProcessorWrapper):

    def __init__(self, name, type, configs):
        super(WeightMetricProcessor, self).__init__(name, type, configs)
        self.finfos_weight_method = configs.get("WeightMethod", "acc")
        self.finfos_weights = None

    def fit_excecute(self, data, layer):
        finfos = data.get("Finfos")
        y_val = data.get("Original").get("y_val")

        finfos_layers = self.obtain_finfos_by_layers(finfos, layer)

        need_finfos = self.obtain_need_finfos(finfos_layers)
        finfo_weights = self.calucate_finfos_weight(finfos, y_val, self.finfos_weight_method)
        self.save_finfos_weights(finfo_weights)

        final_probs = self.obtain_final_probs(need_finfos, finfo_weights)

        final_preds = np.argmax(final_probs, axis=1)
        metric = self.calucate_metric(final_preds, y_val, self.classifier_method)
        return metric

    def save_finfos_weights(self, finfo_weights):
        self.finfos_weights = finfo_weights

    def obtain_final_probs(self, need_finfos, finfo_weights):
        final_probs = []
        for need_finfo, finfo_weight in zip(need_finfos, finfo_weights):
            final_probs.append(need_finfo.get("Probs") * finfo_weight)
        return np.sum(final_probs, axis=0)

    def calucate_finfos_weight(self, finfos, y_val, weight_method):
        finfo_weights = []
        for name, finfo in finfos.items():
            y_pred = finfo.get("Predict")
            finfo_weights.append(self.calucate_metric(y_pred, y_val, weight_method))
        finfo_weights = self.normalize_weights(finfo_weights)
        return finfo_weights

    def normalize_weights(self, weights):
        weights_sum = sum(weights)
        weights = [weight / weights_sum for weight in weights]
        return weights

    def predict_execute(self, data, layer):
        finfos = data.get("Finfos")

        finfos_layers = self.obtain_finfos_by_layers(finfos, layer)
        need_finfos = self.obtain_need_finfos(finfos_layers)
        final_probs = self.obtain_final_probs(need_finfos, self.finfos_weights)

        return final_probs


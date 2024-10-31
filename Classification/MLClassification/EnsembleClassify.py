import logging

import numpy as np
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score

from Classification.Common.ClassifierTemplate import ClassifierTemplate
from Classification.MLClassification.BaseClassify import get_ml_base_classifier

def get_ens_classifier(name, configs, layer, default=False):
    if name == "EnsembleClassify":
        return EnsembleClassify(name, configs, layer)
    elif name == "WeightEnsembleClassify":
        return WeightEnsembleClassify(name, configs, layer)
    elif name == "AdaptiveEnsembleClassifyByNum":
        return AdaptiveEnsembleClassifyByNum(name, configs, layer)
    elif name == "AdaptiveEnsembleClassifyByMid":
        return AdaptiveEnsembleClassifyByMid(name, configs, layer)
    elif name == "AdaptiveEnsembleClassifyByAvg":
        return AdaptiveEnsembleClassifyByAvg(name, configs, layer)
    else:
        if default:
            return None
        else:
            raise "暂时不支持" + name + "分类器"

class EnsembleClassify(ClassifierTemplate):

    def __init__(self, name, configs, layer):

        super(EnsembleClassify, self).__init__(name, configs)

        self.BaseClassifierConfig = configs.get("BaseClassifier", None)
        assert self.BaseClassifierConfig != None, "基分类器必须配置"

        self.init_base_classifiers(layer)
        assert self.BaseClassifierNum != 0, "基分类器的数量不能为空！"

        self.is_encapsulated = configs.get("IsEncapsulated", True)

        if self.is_encapsulated:
            print("使用的集成分类器的名称:", self.name)
            print("初始化的分类器数量:", self.BaseClassifierNum)
            print("初始化的基分类器名称", self.obtain_est_name())

    def init_base_classifiers(self, layer):
        self.BaseClassifierIntances = self.init_base_classifiers_instance(layer)
        self.BaseClassifierNum = len(self.BaseClassifierIntances)

    def init_base_classifiers_instance(self, layer):
        base_classifier_intances = {}
        for name, config in self.BaseClassifierConfig.items():
            if self.check_classifier_init(config, layer):
                base_classifier_intances[name] = get_ml_base_classifier(name, config)
        return base_classifier_intances

    def fit(self, X_train, y_train, X_test=None, y_test=None):
        for name, est in self.BaseClassifierIntances.items():
            est.fit(X_train, y_train)

    def predict_all_probs(self, X):
        probs = {}
        for name, est in self.BaseClassifierIntances.items():
            probs[name] = est.predict_proba(X)
        return probs

    def predict_probs(self, X):
        probs = self.predict_all_probs(X)
        return np.mean(np.stack(probs.values()), axis=0)

    def predict(self, X):
        return np.argmax(self.predict_probs(X), axis=1)

    def obtain_features(self, X):
        features = self.predict_all_probs(X)
        return np.concatenate(features.values(), axis=1)

    def obtain_est_name(self):
        return list(self.BaseClassifierIntances.keys())

    def obtain_est_instances(self):
        return list(self.BaseClassifierIntances.values())

    def check_classifier_init(self, config, layer):
        # 这个方法是用于判断是否当前层需要初始化哪些基分类器
        need_layers = config.get("Layer", None)
        if need_layers == None:
            return True
        if layer in need_layers:
            return True
        return False

class WeightEnsembleClassify(EnsembleClassify):

    def __init__(self, name, configs, layer):
        super(WeightEnsembleClassify, self).__init__(name, configs, layer)

        self.weight_method = configs.get("WeightMetric", "acc")

        self.ClassifierWeights = {}

    def _calculate_weight_metric(self, X_test, y_test, est):
        if callable(self.weight_method):
            return self.weight_method(X_test, y_test, est)
        elif isinstance(self.weight_method, str):
            y_pre = est.predict(X_test)
            return self.obtain_built_in_weight_method(y_pre, y_test)
        else:
            raise "出错"

    def normalize_weights(self, weights):
        weights_sum = sum(weights.values())
        for name, _ in weights.items():
            weights[name] = weights[name] / weights_sum
        return weights

    def calculate_weight_metrics(self, X_test, y_test):
        weight_metrics = {}
        for name, est in self.BaseClassifierIntances.items():
            weight_metrics[name] = self._calculate_weight_metric(X_test, y_test, est)
        return self.normalize_weights(weight_metrics)

    def fit(self, X_train, y_train, X_test=None, y_test=None):
        super(WeightEnsembleClassify, self).fit(X_train, y_train, X_test=None, y_test=None)
        self.ClassifierWeights = self.calculate_weight_metrics(X_test, y_test)

    def predict_probs(self, X):
        features = self.predict_all_probs(X)
        return self.obtain_probs_by_weight(features)

    def obtain_features(self, X):
        features = self.predict_all_probs(X)
        return self.obtain_probs_by_weight(features)

    def obtain_weights(self):
        return self.ClassifierWeights

    def obtain_probs_by_weight(self, features):
        return np.sum([weight * features[name] for name, weight in self.ClassifierWeights.items()], axis=0)

    def obtain_built_in_weight_method(self, x1, x2):
        if self.weight_method.lower() in ["accuracy", "acc"]:
            return accuracy_score(x1, x2)
        if self.weight_method.lower() in ["precision", "pre"]:
            return precision_score(x1, x2)
        if self.weight_method.lower() in ["recall"]:
            return recall_score(x1, x2)
        if self.weight_method.lower() in ["f1_score", "f1", "f1-score"]:
            return f1_score(x1, x2)

    def set_weight_method(self, weight_method):
        if callable(weight_method):
            self.weight_method = weight_method
        else:
            logging.error("设置的分类器必须要是可调用的")

class AdaptiveEnsembleClassify(EnsembleClassify):

    def __init__(self, name, configs, layer):
        super(AdaptiveEnsembleClassify, self).__init__(name, configs, layer)

        self.RetainedClassifier = None
        self.ClassifierMetrics = {}

        self.metric_method = configs.get("CaluateMetric", "acc")

    def calculate_adaptive_metrics(self, X_test, y_test):
        classifier_metrics = {}
        for name, est in self.BaseClassifierIntances.items():
            classifier_metrics[name] = self._calculate_adaptive_metric(X_test, y_test, est)
            classifier_metrics[name] = self._calculate_adaptive_metric(X_test, y_test, est)
        return classifier_metrics

    def print_classifier_metrics(self):
        print("当前层不同分类器的指标:", end=" ")
        for name, metric in self.ClassifierMetrics.items():
            print(name, ":", format(metric, ".4f"), end=", ")
        print()


    def fit(self, X_train, y_train, X_test=None, y_test=None):
        super(AdaptiveEnsembleClassify, self).fit(X_train, y_train, X_test=None, y_test=None)
        self.ClassifierMetrics = self.calculate_adaptive_metrics(X_test, y_test)
        self.RetainedClassifier = self.complete_adaptive_method(self.ClassifierMetrics)
        if self.is_encapsulated:
            self.print_classifier_metrics()
            print("筛选出的基分类器:", self.RetainedClassifier)

    def predict_retained_probs(self, X):
        features = []
        for name in self.RetainedClassifier:
            est = self.BaseClassifierIntances[name]
            features.append(est.predict_proba(X))
        return features

    def predict_probs(self, X):
        probs = self.predict_retained_probs(X)
        return np.mean(np.stack(probs), axis=0)

    def obtain_features(self, X):
        features = self.predict_retained_probs(X)
        return np.concatenate(features, axis=1)

    def obtain_retained_ens_name(self):
        est_names_temp = []
        for is_retained, est_name in zip(self.retained, self.ensemble_names):
            if is_retained:
                est_names_temp.append(est_name)
        return est_names_temp

    def obtain_retained_ens_instances(self):
        est_instances_temp = []
        for is_retained, est in zip(self.retained, self.ensembles):
            if is_retained:
                est_instances_temp.append(est)
        return est_instances_temp

    def _calculate_adaptive_metric(self, X_test, y_test, ens=None):
        if callable(self.metric_method):
            return self.metric_method(X_test, y_test, ens)
        elif isinstance(self.metric_method, str):
            y_pre = ens.predict(X_test)
            return self.obtain_built_in_metric(y_pre, y_test)
        else:
            raise "出错"


    def set_metric_method(self, metric_method):
        if callable(metric_method):
            self.metric_method = metric_method
        else:
            logging.error("设置的分类器必须要是可调用的")

    def complete_adaptive_method(self):
        pass

    def obtain_built_in_metric(self, x1, x2):
        if self.metric_method.lower() in ["accuracy", "acc"]:
            return accuracy_score(x1, x2)
        if self.metric_method.lower() in ["precision", "pre"]:
            return precision_score(x1, x2)
        if self.metric_method.lower() in ["recall"]:
            return recall_score(x1, x2)
        if self.metric_method.lower() in ["f1_score", "f1", "f1-score"]:
            return f1_score(x1, x2)

class AdaptiveEnsembleClassifyByNum(AdaptiveEnsembleClassify):

    def __init__(self, name, configs, layer):

        super(AdaptiveEnsembleClassifyByNum, self).__init__(name, configs, layer)

        self.retained_num = configs.get("RetainedNum", 3)
        print("筛选后保留的基分类器的数量:", self.retained_num)

        assert len(self.BaseClassifierIntances) >= self.retained_num, "基分类器的数量必须要大于保留的数量"

    def complete_adaptive_method(self, classifier_metrics):
        return sorted(classifier_metrics, key=classifier_metrics.get, reverse=True)[:self.retained_num]

class AdaptiveEnsembleClassifyByMid(AdaptiveEnsembleClassify):
    def __init__(self, name, configs, layer):
        super(AdaptiveEnsembleClassifyByMid, self).__init__(name, configs, layer)

    def complete_adaptive_method(self):
        mid = np.median(self.ClassifierMetrics.values())
        self.RetainedClassifier = [name for name, metric in self.ClassifierMetrics.items() if metric > mid] #todo 大于基分类器
        self.retained_num = len(self.RetainedClassifier)

class AdaptiveEnsembleClassifyByAvg(AdaptiveEnsembleClassify):
    def __init__(self, name, configs, layer):
        super(AdaptiveEnsembleClassifyByAvg, self).__init__(name, configs, layer)

    def complete_adaptive_method(self):
        mn = np.mean(self.ClassifierMetrics.values())
        self.RetainedClassifier = [name for name, metric in self.ClassifierMetrics.items() if metric > mn]
        self.retained_num = len(self.RetainedClassifier)

class AdaptiveWeightEnsembleClassify(AdaptiveEnsembleClassify, WeightEnsembleClassify):
    def __init__(self, name, configs, layer):
        EnsembleClassify.__init__(self, name, configs, layer)

        self.RetainedClassifier = None
        self.ClassifierWeights = {}

        self.ClassifierMetrics = {}
        self.weight_method = configs.get("WeightMetric", "acc")
        self.metric_method = configs.get("CaluateMetric", "acc")

    def fit(self, X_train, y_train, X_test=None, y_test=None):
        EnsembleClassify.fit(X_train, y_train, X_test=None, y_test=None)
        self.ClassifierMetrics = self.calculate_adaptive_metrics(X_test, y_test)
        self.RetainedClassifier = self.complete_adaptive_method(self.ClassifierMetrics)
        self.ClassifierWeights = self.calculate_weight_metrics(X_test, y_test, self.RetainedClassifier)
        if self.is_encapsulated:
            self.print_classifier_metrics()
            print("筛选出的基分类器:", self.RetainedClassifier)

    def predict_proba(self, X):
        probs = self.predict_retained_probs(X)
        return self.obtain_probs_by_weight(probs)

    def obtain_features(self, X):
        features = self.predict_retained_probs(X)
        return self.obtain_probs_by_weight(features)

    def calculate_weight_metrics(self, X_test, y_test, retained_classifier):
        weight_metrics = {}
        for name in retained_classifier:
            est = self.BaseClassifierIntances[name]
            weight_metrics[name] = self._calculate_weight_metric(X_test, y_test, est)
        return self.normalize_weights(weight_metrics)

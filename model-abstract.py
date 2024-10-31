import copy
import time
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
import warnings
from abc import ABC, abstractmethod

from Processor.ProcessorDispatcher.Dispatcher import FeatFusionDispatcher, FeatSelectorDispatcher, \
    FeaturesProcessorDispatcher, MetricProcessorDispatcher, ClassifierDispatcher, CategoryImbalanceDispatcher, \
    PreProcessorDispatcher, PostProcessorDispatcher, FusionFeatDispatcher, SplitFeatureDispatcher, \
    FeatureSplitDispatcher

warnings.filterwarnings("ignore")


class UnimodalModel(ABC):

    def __init__(self, config):
        """
        初始化模型，并根据传入的配置字典设定参数，如最大迭代次数、终止层、类别数等。
        """
        assert config != None, "单模态级联模型的配置信息不能为空"
        self.config = config
        self.max_num_iterations = config.get("MaxNumIterations", 20)
        self.termination_layer = config.get("TerminationLayer", 3)
        self.class_num = config.get("ClassNum", None)
        self.debug = config.get("Debug", True)
        self.feature_types = config.get("DataType", set("Global"))
        self.classifier_instances = dict()
        self.f_select_ids = {}
        self._init_components(config)
        self.all_feature_split_processors = dict()
        self.all_feature_fusions_processors = dict()
        self.all_split_feature_processors = dict()
        self.all_fusion_feature_processors = dict()
        self.all_feature_processors = dict()
        self.all_metrics_processor = dict()
        self.all_feature_types = dict()

    @abstractmethod
    def _init_components(self, config):
        """根据配置初始化模型组件。"""
        pass

    @abstractmethod
    def _init_pre_processor(self, configs):
        """初始化数据的预处理步骤。"""
        pass

    @abstractmethod
    def _init_feature_selectors(self, configs):
        """根据配置初始化特征选择器。"""
        pass

    @abstractmethod
    def _init_data_and_feature_fusion(self, configs):
        """设置特征融合方法。"""
        pass

    @abstractmethod
    def _init_fusion_feature_processors(self, configs):
        """初始化特征融合处理器。"""
        pass

    @abstractmethod
    def _init_feature_split_processor(self, configs):
        """初始化特征切分处理器。"""
        pass

    @abstractmethod
    def _init_split_feature_processors(self, configs):
        """初始化切分特征处理器。"""
        pass

    @abstractmethod
    def _init_category_imbalance_processors(self, config):
        """初始化类别不平衡处理器。"""
        pass

    @abstractmethod
    def _init_cascade_classifier_builder(self, configs):
        """设置级联分类器的构建器。"""
        pass

    @abstractmethod
    def _init_cascade_features_processors(self, config):
        """初始化级联特征处理器。"""
        pass

    @abstractmethod
    def _init_cascade_metrics_processor(self, config):
        """设置用于评估每层模型的指标处理器。"""
        pass

    @abstractmethod
    def _init_post_processor(self, config):
        """初始化模型的后处理步骤。"""
        pass

    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """使用训练数据和验证数据对模型进行训练。"""
        pass

    @abstractmethod
    def _fit(self, X_train, y_train, X_val, y_val):
        """模型内部训练方法，通过层级迭代进行训练。"""
        pass

    @abstractmethod
    def save_relevant_fit_to_data(self, all_finfos, data, layer):
        """在每层训练后保存相关信息。"""
        pass

    @abstractmethod
    def execute_before_fit(self, X_train, y_train, X_val, y_val):
        """执行训练前的准备步骤，例如准备数据。"""
        pass

    @abstractmethod
    def execute_pre_fit_processor(self, data):
        """在训练前应用预处理器。"""
        pass

    @abstractmethod
    def pre_fit_cascade_data_and_infos(self, data, layer):
        """在每次迭代前准备级联数据，并更新特征选择、融合方法等。"""
        pass

    @abstractmethod
    def execute_feature_selector_processors(self, data, layer):
        """执行特征选择算法。"""
        pass

    @abstractmethod
    def execute_fit_feature_selection(self, data, f_select_idxs):
        """应用特征选择处理步骤。"""
        pass

    @abstractmethod
    def save_f_select_ids(self, f_select_ids, layer):
        """保存选定的特征 ID。"""
        pass

    @abstractmethod
    def execute_feature_and_data_fit_fusion(self, data, layer):
        """执行特征融合并更新数据。"""
        pass

    @abstractmethod
    def split_fit_data_to_local(self, data, layer):
        """对数据进行分割以便本地处理。"""
        pass

    @abstractmethod
    def execute_fit_fusion_features_processors(self, data, layer):
        """执行融合特征处理器。"""
        pass

    @abstractmethod
    def execute_fit_split_features_processors(self, data, layer):
        """执行切分特征处理器。"""
        pass

    @abstractmethod
    def obtain_new_update_builder_configs(self, data, layer):
        """为下一层模型更新构建器配置。"""
        pass

    @abstractmethod
    def execute_category_imbalance(self, data, layer):
        """处理训练过程中的类别不平衡问题。"""
        pass

    @abstractmethod
    def execute_cascade_fit_classifier(self, data, builder_configs, layer):
        """在级联方式下训练分类器。"""
        pass

    @abstractmethod
    def obtain_fit_classifier_instance(self, X_train, y_train, X_val, y_val, builder_configs, classifier_builders,
                                       layer):
        """获取在训练过程中的分类器实例。"""
        pass

    @abstractmethod
    def obtain_relevant_fit_to_data(self, data, classifier_instances, layer):
        """提取相关信息，如特征、预测概率和预测结果。"""
        pass

    @abstractmethod
    def adjust_cascade_classifier(self, classifier_instances, data):
        """在级联过程中调整分类器。"""
        pass

    @abstractmethod
    def save_cascade_classifier(self, classify_instances, layer):
        """在每层训练后保存分类器。"""
        pass

    @abstractmethod
    def obtain_current_metric(self, data, layer):
        """计算当前层的性能指标。"""
        pass

    @abstractmethod
    def execute_post_fit_processor(self, data, layer):
        """在训练后执行后处理器。"""
        pass

    @abstractmethod
    def post_fit_cascade_data_and_infos(self, data, layer):
        """在每层训练后处理数据和特征。"""
        pass

    @abstractmethod
    def execute_after_fit(self, data):
        """训练完成后的操作。"""
        pass

    @abstractmethod
    def predict(self, X):
        """使用训练好的模型进行预测。"""
        pass

    @abstractmethod
    def predict_proba(self, X):
        """预测类别概率。"""
        pass

    @abstractmethod
    def save_relevant_to_predict_data(self, all_finfos, data, layer):
        """保存每层预测的相关数据。"""
        pass

    @abstractmethod
    def execute_before_predict_probs(self, X):
        """在进行预测前准备数据。"""
        pass

    @abstractmethod
    def execute_pre_predict_processor(self, data):
        """在预测前应用预处理器。"""
        pass

    @abstractmethod
    def pre_predict_cascade_data_and_infos(self, data, layer):
        """在每层级联预测前更新数据。"""
        pass

    @abstractmethod
    def obtain_cascade_f_select_ids(self, layer):
        """获取某一层级联的特征选择 ID。"""
        pass

    @abstractmethod
    def execute_predict_feature_selection(self, data, f_select_ids):
        """在预测时应用特征选择。"""
        pass

    @abstractmethod
    def execute_feature_and_data_predict_fusion(self, data, layer):
        """在预测时应用特征融合。"""
        pass

    @abstractmethod
    def split_predict_data_to_local(self, data, layer):
        """在预测时将数据切分为本地处理。"""
        pass

    @abstractmethod
    def execute_predict_fusion_features_processors(self, data, layer):
        """在预测时处理融合特征。"""
        pass

    @abstractmethod
    def execute_predict_split_features_processors(self, data, layer):
        """在预测时处理切分特征。"""
        pass

    @abstractmethod
    def obtain_cascade_predict_classifier_instance(self, layer):
        """在预测时获取分类器实例。"""
        pass

    @abstractmethod
    def obtain_relevant_to_predict_data(self, data, classifier_instances, layer):
        """在预测时提取相关数据，如特征和预测结果。"""
        pass

    @abstractmethod
    def execute_predict_feature_processors(self, features, layer):
        """在预测时应用特征处理器处理特征。"""
        pass

    @abstractmethod
    def post_predict_cascade_data_and_infos(self, data, layer):
        """在每层预测后更新数据。"""
        pass

    @abstractmethod
    def execute_after_predict_probs(self, data):
        """预测完成后的操作。"""
        pass
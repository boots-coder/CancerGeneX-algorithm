from ClassifierBuider.ClassifierBuilder import get_global_classifier_buider, get_local_classifier_buider
from Processor.CategoryImbalance.CategoryBalancer import get_category_balancer
from Processor.FeatureSplitProcessor.FeatureSplitProcessor import get_lobal_feature_split_processor

from Processor.FeaturesProcessor.Standardization.Standardizer import get_standardizer

from Processor.FeatureSelector.Selector.BaseSelector import get_base_selector
from Processor.FeatureSelector.Selector.EnsembleSelector import get_ens_selector
from Processor.FeatureSelector.Selector.RecallAttribute import get_attribute_recall_method
from Processor.FeatureFusion.ConcatenationFusion import get_feature_concatenation_method

from Processor.MetricProcessor.MetricProcessor import get_metric_processor
from Processor.PreProcessor.Standardization.Standardizer import get_pre_standardization_processor


class Dispatcher():

    def __init__(self, Template):
        self.Template = Template

    def obtain_instance(self, config):
        est_name = config.get("Name", None)
        est_type = config.get("Type", None)
        return self._obtain_instance(est_name, est_type, config)

    #确保每个创建的对象都符合指定的模板 Template 类，从而统一对象的接口和功能，避免不符合要求的实例出现在系统中。
    def _obtain_instance(self, name, est_type, config):
        est = self.execute_dispatcher_method(name, est_type, config)
        if isinstance(est, self.Template):
            return est
        else:
            raise est.name + "没有继承 " + self.Template.__name__ + " 类"

    def execute_dispatcher_method(self, name, est_type, configs):
        pass

class FeatFusionDispatcher(Dispatcher):

    def __init__(self):
        from Processor.Common.Template import FeatFusionTemplate
        super(FeatFusionDispatcher, self).__init__(FeatFusionTemplate)

    def execute_dispatcher_method(self, name, est_type, config):
        return get_feature_concatenation_method(name, est_type, config)

class MetricProcessorDispatcher(Dispatcher):

    def __init__(self):
        from Processor.Common.Template import MetricProcessorTemplate
        super(MetricProcessorDispatcher, self).__init__(MetricProcessorTemplate)

    def execute_dispatcher_method(self, name, est_type, config):
        return get_metric_processor(name, est_type, config)

class CategoryImbalanceDispatcher(Dispatcher):

    def __init__(self):
        from Processor.Common.Template import CategoryBalancerTemplate
        super(CategoryImbalanceDispatcher, self).__init__(CategoryBalancerTemplate)

    def execute_dispatcher_method(self, name, est_type, config):
        return get_category_balancer(name, est_type, config)

class FeatureSplitDispatcher(Dispatcher):
    def __init__(self):
        from Processor.Common.Template import FeatureSplitProcessorTemplate
        super(FeatureSplitDispatcher, self).__init__(FeatureSplitProcessorTemplate)

    def execute_dispatcher_method(self, name, est_type, config):
        return get_lobal_feature_split_processor(name, est_type, config)

class ListDispatcher(Dispatcher):

    def obtain_instance(self, configs):
        ests = []
        for name, config in configs.items():
            est_type = config.get("Type", None)
            est = super()._obtain_instance(name, est_type, config)
            ests.append(est)
        return ests

class PreProcessorDispatcher(ListDispatcher):

    def __init__(self):
        from Processor.Common.Template import PreProcessorTemplate
        super(PreProcessorDispatcher, self).__init__(PreProcessorTemplate)

    def execute_dispatcher_method(self, name, est_type, config):
        if est_type == "Standardization":
            return get_pre_standardization_processor(name, est_type, config)
        else:
            raise "暂时不支持这种特征筛选器" #todo ？ 应该没来得及改


class PostProcessorDispatcher(ListDispatcher):

    def __init__(self):
        from Processor.Common.Template import PostProcessorTemplate
        super(PostProcessorDispatcher, self).__init__(PostProcessorTemplate)

    def execute_dispatcher_method(self, name, est_type, config):
        pass

class FeatSelectorDispatcher(ListDispatcher):

    def __init__(self):
        from Processor.Common.Template import FeatureSelectorTemplate
        super(FeatSelectorDispatcher, self).__init__(FeatureSelectorTemplate)

    def execute_dispatcher_method(self, name, est_type, config):
        if est_type == "FeatureSelection":
            return get_base_selector(name, config) or get_ens_selector(name, config)
        elif est_type == "RecallAttribute":
            return get_attribute_recall_method(name, config)
        else:
            raise "暂时不支持这种特征筛选器"
class FusionFeatDispatcher(ListDispatcher):

    def __init__(self):
        from Processor.Common.Template import FusionFeaturesTemplate
        super(FusionFeatDispatcher, self).__init__(FusionFeaturesTemplate)

    def execute_dispatcher_method(self, name, est_type, config):
        pass

class SplitFeatureDispatcher(Dispatcher):
    def __init__(self):
        from Processor.Common.Template import SplitFeatureProcessorTemplate
        super(SplitFeatureDispatcher, self).__init__(SplitFeatureProcessorTemplate)

    def execute_dispatcher_method(self, name, est_type, config):
        pass

class FeaturesProcessorDispatcher(ListDispatcher):

    def __init__(self):
        from Processor.Common.Template import FeaturesProcessorTemplate
        super(FeaturesProcessorDispatcher, self).__init__(FeaturesProcessorTemplate)

    def execute_dispatcher_method(self, name, est_type, config):
        if est_type == "Standardization":
            return get_standardizer(name, est_type, config)
        else:
            raise "暂时不支持这种特征筛选器"

class ClassifierDispatcher(Dispatcher):

    def __init__(self):
        from ClassifierBuider.ClassifierBuilder import BuilderBuiderWrapper
        super(ClassifierDispatcher, self).__init__(BuilderBuiderWrapper)


    def obtain_instance(self, configs):
        global_classifier_builders, local_classifier_builders = [], []
        for name, config in configs.items():
            builder_type = config.get("Builder", None)
            data_types = config.get("DataType", None)
            for data_type in data_types:
                if "Global" == data_type:
                    est = self.execute_global_dispatcher_method(name, builder_type, data_type, config)
                    self.check_classifier_builder(est)
                    global_classifier_builders.append(est)

                elif data_type == "Local":
                    est = self.execute_local_dispatcher_method(name, builder_type, data_type, config)
                    self.check_classifier_builder(est)
                    local_classifier_builders.append(est)

        return global_classifier_builders, local_classifier_builders

    def check_classifier_builder(self, est):
        if not isinstance(est, self.Template):
            raise est.__name__ + "没有继承 " + self.Template.__name__ + " 类"


    def execute_global_dispatcher_method(self, name, builder_type, data_type, config):
        return get_global_classifier_buider(name, builder_type, data_type, config)

    def execute_local_dispatcher_method(self, name, builder_type, data_type, config):
        return get_local_classifier_buider(name, builder_type, data_type, config)


class GlobalAndLocalClassifierDispatcher(Dispatcher):

    def __init__(self):
        from ClassifierBuider.ClassifierBuilder import BuilderBuiderWrapper
        super(GlobalAndLocalClassifierDispatcher, self).__init__(BuilderBuiderWrapper)


    def obtain_instance(self, configs):
        global_classifier_builders, local_classifier_builders = [], []
        for name, config in configs.items():
            builder_type = config.get("Builder", None)
            data_types = config.get("DataType", None)
            for data_type in data_types:
                if "Global" == data_type:
                    est = self.execute_global_dispatcher_method(name, builder_type, data_type, config)
                    self.check_classifier_builder(est)
                    global_classifier_builders.append(est)

                elif data_type == "Local":
                    est = self.execute_local_dispatcher_method(name, builder_type, data_type, config)
                    self.check_classifier_builder(est)
                    local_classifier_builders.append(est)

        return global_classifier_builders, local_classifier_builders

    def check_classifier_builder(self, est):
        if not isinstance(est, self.Template):
            raise est.__name__ + "没有继承 " + self.Template.__name__ + " 类"


    def execute_global_dispatcher_method(self, name, builder_type, data_type, config):
        return get_global_classifier_buider(name, builder_type, data_type, config)

    def execute_local_dispatcher_method(self, name, builder_type, data_type, config):
        return get_local_classifier_buider(name, builder_type, data_type, config)

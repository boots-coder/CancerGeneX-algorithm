
class RootTemplate():

    def __init__(self, name, configs):
        pass

    def executable(self, layer, kwargs):
        if self.need_layers == None or layer in self.need_layers:
            return True
        else:
            return False

    def excecute(self):
        pass

    def fit_executable(self, layer):
        pass


    def fit_execute(self):
        pass

    def predict_executable(self, layer):
        pass

    def predict_execute(self):
        pass

    def obtain_name(self):
        return self.name

    def set_need_layers(self, need_layers):
        if need_layers == None or isinstance(need_layers, list):
            self.need_layers = need_layers
        else:
            raise "设置需要执行层数时必须要是列表形式"


class PreProcessorTemplate(RootTemplate):
    pass

class FeatFusionTemplate(RootTemplate):
    pass

class FeatureSelectorTemplate(RootTemplate):
    pass

class FeaturesProcessorTemplate(RootTemplate):
    pass

class FusionFeaturesTemplate(RootTemplate):
    pass

class CategoryBalancerTemplate(RootTemplate):
    pass

class MetricProcessorTemplate(RootTemplate):
    pass

class PostProcessorTemplate(RootTemplate):
    pass

class FeatureSplitProcessorTemplate(RootTemplate):
    pass

class SplitFeatureProcessorTemplate(RootTemplate):
    pass
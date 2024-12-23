# 定义一个根模板类，用于定义所有子类的基础功能
# Define a root template class that serves as the base for all subclasses
class RootTemplate():

    # 初始化函数，接收名称和配置参数
    # Initialization function, takes name and configs as parameters
    def __init__(self, name, configs):
        pass

    # 检查某一层是否可执行
    # Check if a specific layer is executable
    def executable(self, layer, kwargs):
        if self.need_layers == None or layer in self.need_layers:
            return True
        else:
            return False

    # 执行函数，需具体子类实现
    # Execution function, needs to be implemented in subclasses
    def excecute(self):
        pass

    # 检查某一层是否适合执行
    # Check if a specific layer is fit for execution
    def fit_executable(self, layer):
        pass

    # 执行适合的层
    # Execute the fit layers
    def fit_execute(self):
        pass

    # 检查某一层是否适合预测
    # Check if a specific layer is fit for prediction
    def predict_executable(self, layer):
        pass

    # 执行预测功能
    # Execute prediction
    def predict_execute(self):
        pass

    # 获取对象的名称
    # Obtain the name of the object
    def obtain_name(self):
        return self.name

    # 设置需要的层列表
    # Set the list of required layers
    def set_need_layers(self, need_layers):
        if need_layers == None or isinstance(need_layers, list):
            self.need_layers = need_layers
        else:
            raise "设置需要执行层数时必须要是列表形式"  # Raise an error if the input is not a list

# 定义子类，用于不同功能模块的实现
# Define subclasses for implementing different functional modules

# 数据预处理模板
# Data preprocessing template
class PreProcessorTemplate(RootTemplate):
    pass

# 特征融合模板
# Feature fusion template
class FeatFusionTemplate(RootTemplate):
    pass

# 特征选择模板
# Feature selection template
class FeatureSelectorTemplate(RootTemplate):
    pass

# 特征处理模板
# Feature processing template
class FeaturesProcessorTemplate(RootTemplate):
    pass

# 融合后的特征模板
# Fused features template
class FusionFeaturesTemplate(RootTemplate):
    pass

# 类别平衡模板
# Category balancing template
class CategoryBalancerTemplate(RootTemplate):
    pass

# 评估指标处理模板
# Metric processing template
class MetricProcessorTemplate(RootTemplate):
    pass

# 后处理模板
# Postprocessing template
class PostProcessorTemplate(RootTemplate):
    pass

# 特征分割处理模板
# Feature splitting processing template
class FeatureSplitProcessorTemplate(RootTemplate):
    pass

# 分割特征处理模板
# Splitted feature processing template
class SplitFeatureProcessorTemplate(RootTemplate):
    pass
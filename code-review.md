# Adap-BDCM - 代码整理

This doc aim to review the code part of the paper Adam-BDCM

<!--more-->

![image-20241031133547754](https://markdown-pictures-jhx.oss-cn-beijing.aliyuncs.com/picgo/image-20241031133547754.png)

## 配置文件

这个代码的 config 是一个字典，包含了整个模型训练和处理的配置选项。以下是 config 的详细分解和说明：

```python
config = {
    "ClassNum" : class_num,  # 注意这个参数一定要改
    "Metric" : "acc",
    "DataType" : {"Global"},

    "PreProcessors": {
        "MinMax": {
            "Type": "Standardization",
            "Method": "DecimalScale",
            "BuilderType": ["DL"],
            "FeaturesType": []
        },
    },

    "FeatureSelector" : {
        "GCLasso" : {
            "Type" : "FeatureSelection",
            "Method" : "GCLasso",
            "Parameter" : {},
        },
        "RecallAttribute" : {
            "name" : "RecallAttribute",
            "Type": "RecallAttribute",
            "Method": "RecallAttribute",
            "Parameter": {},
        },
    },

    "FeatureFusion" : {
        "Name" : "FeatureFusion",
        "BuilderType": ["ML", "DL"],
        "Type" : "FeatureConcatenation",
    },

    # "CategoryImbalance": {
    #     "Name" : "SMOTE",
    #     "Type" : "CategoryImbalance",
    #     "Parameter": {},
    # },

    # "CategoryImbalance": {
    #     "Name" : "SMOTE",
    #     "Type" : "CategoryImbalance",
    #     "Parameter": {},
    # },

    "FeatureSplitProcessor" : {
        "Name" : "AverageSplit",
        "Type" : "AverageSplit",
        "SplitNum" : 3,
    },

    "FeatureProcessors": {
        "0": {
            "Type": "Standardization",
            "Method": "DecimalScale",
            "BuilderType": ["DL"],
            "FeaturesType": []
        },
    },

    "MetricsProcessors": {
        "Name": "MetricsProcessor",
        "BuilderType": ["ML", "DL"],
        "Type": "AvgMetricProcessor",
        "ClassifierMethod": "acc",
    },


    "CascadeClassifier": {
        "AdaptiveEnsembleClassifyByNum" : {
            "AdaptiveMethod": "retained_num",
            "CaluateMetric" : "acc",
            "Builder" : "ML",
            "Type" : "AdaptiveEnsembleClassifyByNum",
            "DataType" : ["Global", "Local"],
            "BaseClassifier" : {
                "RandomForestClassifier" : {
                    "Layer" : [2, 3],
                    "Type" : "RandomForestClassifier",
                    "Parameter" : {"n_estimators": 100, "criterion": "gini",
                                   "class_weight": None, "random_state": 0},
                    },
                "ExtraTreesClassifier" : {
                    "Type": "ExtraTreesClassifier",
                    "Parameter": {"n_estimators": 100, "criterion": "gini",
                                  "class_weight": None, "random_state": 0},
                    },
                "GaussianNBClassifier" :{
                    "Type": "GaussianNBClassifier",
                    "Parameter": {}
                    },
                "BernoulliNBClassifier" : {
                    "Type": "BernoulliNBClassifier",
                    "Parameter": {}
                    },
                "KNeighborsClassifier_1" : {
                    "Type": "KNeighborsClassifier",
                    "Parameter": {"n_neighbors": 2}
                    },
                "KNeighborsClassifier_2": {
                    "Type": "KNeighborsClassifier",
                    "Parameter": {"n_neighbors": 3}
                    },
                "KNeighborsClassifier_3": {
                    "Type": "KNeighborsClassifier",
                    "Parameter": {"n_neighbors": 5}
                    },
                "GradientBoostingClassifier": {
                    "Type": "GradientBoostingClassifier",
                    "Parameter": {}
                    },
                "SVCClassifier_1": {
                    "Type": "SVCClassifier",
                    "Parameter": {"kernel": "linear", "probability": True}
                    },
                "SVCClassifier_2": {
                    "Type": "SVCClassifier",
                    "Parameter": {"kernel": "rbf", "probability": True}
                    },
                "SVCClassifier_3": {
                    "Type": "SVCClassifier",
                    "Parameter": {"kernel": "sigmoid", "probability": True}
                    },
                "LogisticRegressionClassifier_1": {
                    "Type": "LogisticRegressionClassifier",
                    "Parameter": {"penalty": 'l2'}
                    },
                "LogisticRegressionClassifier_2": {
                    "Type": "LogisticRegressionClassifier",
                    "Parameter": {"C": 1, "penalty": 'l1', "solver": 'liblinear'}
                    },
                "LogisticRegressionClassifier_3": {
                    "Type": "LogisticRegressionClassifier",
                    "Parameter": {"penalty": None}
                    },

                }
            },

        "DNN" : {
            "Layers" : None,
            "Builder" : "DL",
            "DataType": ["Global", "Local"],
            "Trainer" : {
                "name" : "TrainerWrapper",
                "Parameter" : {}
            },
            "Model" : {
                "name" : "BNN",
                "Parameter" : {"ClassNum" : 2}
            },
            "LossFun" : {
                "name" : "CrossEntropyLoss",
                "Parameter" : {}
            },
            "Optimizer" : {
                "name" : "Adam",
                "Parameter" : {"lr" : 0.001},
            }
        }
    },
}
```

**配置信息详细说明**



​	1.	**ClassNum**：

​	•	表示分类问题的类别数，此处设置为2，表明这是一个二分类问题。

​	2.	**Metric**：

​	•	定义评估指标，这里使用的是 “acc”（即准确率）

​	3.	**DataType**：

​	•	指定数据类型范围，例如 “Global” 表示全局数据，也可能用于区分数据的来源或用途。

​	4.	**PreProcessors**：

​	•	定义数据预处理方法。这里指定了 MinMax 的方法为 “Standardization”，使用 “DecimalScale” 进行标准化，并限制适用于 BuilderType 为 “DL” 的模型。

​	5.	**FeatureSelector**：

​	•	特征选择方法，包括：

​	•	**GCLasso**：一种特征选择方法，使用的是图拉索正则化。

​	•	**RecallAttribute**：用于“RecallAttribute”方法的特征选择，通常在特征回忆任务中有用。

​	6.	**FeatureFusion**：

​	•	特征融合方法，配置为”FeatureConcatenation”（特征拼接）。BuilderType 指定了适用模型类型（包括 “ML” 和 “DL” 模型）。

​	7.	**FeatureSplitProcessor**：

​	•	特征拆分器，名称为 “AverageSplit”，类型为平均拆分，并定义了分割数（SplitNum）为3。

​	8.	**FeatureProcessors**：

​	•	定义特征处理方法。这里是 “Standardization” 标准化，使用 “DecimalScale” 方法。

​	9.	**MetricsProcessors**：

​	•	指标处理器，名称为 “MetricsProcessor”，类型为 “AvgMetricProcessor”，主要用于计算平均指标，适用于 “ML” 和 “DL” 的构建器类型。

​	10.	**CascadeClassifier**：

​	•	级联分类器，包含两类不同的分类器：

​	•	**AdaptiveEnsembleClassifyByNum**：一种自适应分类器，按数量进行自适应分类，使用”acc”作为评估指标。其下包含多种基础分类器（例如 RandomForestClassifier, ExtraTreesClassifier, GaussianNBClassifier 等）。

​	•	**DNN**：深度神经网络（DNN）分类器，包含一些层、训练器、模型、损失函数（如 CrossEntropyLoss）和优化器（如 Adam）。



**config 的读取方式**

在代码中，config 作为字典传递给 UnimodalModel 的初始化函数（即 model = UnimodalModel(config)），其中 config 的各个键值对都包含了模型初始化和训练所需的各种参数信息。随后，这些参数可能会在模型的内部配置流程中被调用，以控制模型的预处理、特征选择、特征融合、特征拆分等一系列操作。

> todo 提问 
>
> 1. "DataType" : ["Global", "Local"] 这里分别表示什么含义？
> 2. DNN 中的 BNN 表示什么含义？





## 构造模型

```python
model = UnimodalModel(config)
```

这里UnimodalModel 类中简单一提：

1 构造器

```python
 def __init__(self, config):

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
```

2 受保护方法（非强制私有方法）

```python
    def _init_components(self, config):
        self._init_pre_processor(config)
        self._init_feature_selectors(config)
        self._init_data_and_feature_fusion(config)
        self._init_fusion_feature_processors(config)
        self._init_split_feature_processors(config)
        self._init_feature_split_processor(config)
        self._init_category_imbalance_processors(config)
        self._init_cascade_classifier_builder(config)
        self._init_cascade_features_processors(config)
        self._init_cascade_metrics_processor(config)
        self._init_post_processor(config)
```

这里也就是说， 在创建模型的同时- 构造了需要的 所有实例

以特征选择器为例：

Step1 调用

```python
self._init_feature_selectors(config)
```

step2 具体实现

```python
def _init_feature_selectors(self, configs):
    feature_selector_configs = configs.get("FeatureSelector", None)
    if feature_selector_configs != None:
        feat_selector_dispatcher = FeatSelectorDispatcher()
        self.feature_selectors = feat_selector_dispatcher.obtain_instance(feature_selector_configs)
    else:
        self.feature_selectors = []
```

step 3 特征选择调度

这里主要为了分配 特征选择/ 召回 对象-- 所以是调度的意思

```python
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
```



step 4 获得最终的 基础特征选择器

```python
# 获取基础选择器
def get_base_selector(name, config):
    method_name = config.get("Method", None)  # 从配置中获取方法名
    if method_name == "GCLasso":
        return GCLasso(name, config)  # 如果方法是"GCLasso"，返回GCLasso选择器实例
    elif method_name == "GCFClassif":
        return GCFClassif(name, config)  # 如果方法是"GCFClassif"，返回GCFClassif选择器实例
    else:
        raise ValueError("Invalid method name")  # 如果方法名不匹配，抛出异常


```



Step 5 自定义 GClasso  的具体信息

```python
class GCLasso(SelectorWrapper):
    def __init__(self, name, kwargs):
        from sklearn.linear_model import Lasso  # 引入Lasso模型
        self.coef = kwargs.get("coef", 0.0001) if kwargs != None else 0.0001  # 获取系数阈值，默认为0.0001
        kwargs = {
            "alpha" : 0.0001, "copy_X" : True, "fit_intercept" : True, "max_iter" : 10000,
            "normalize" : True, "positive" : False, "precompute" : False, "random_state" : None,
            "selection" : 'cyclic', "tol" : 0.0001,  "warm_start" : False
        }  # 设置Lasso模型参数
        super(GCLasso, self).__init__(name, Lasso, kwargs)  # 调用父类构造函数

    def _obtain_selected_index(self, X_train, y_train):
        select_idxs = []  # 选择的特征索引列表
        select_infos = {}  # 选择的特征信息字典


        for ind, coef_ in enumerate(self.est.coef_):  # 遍历模型系数
            if np.abs(coef_) > self.coef:  # 如果系数的绝对值大于阈值
                select_idxs.append(ind)  # 添加索引到选择列表
                select_infos[ind] = coef_  # 添加系数到信息字典
        select_infos["Num"] = len(select_idxs)  # 记录选择特征的数量
        select_infos["Name"] = self.name  # 记录选择器名称
        return select_idxs, select_infos  # 返回选择特征的索引和信息

    def obtain_all_index(self, X=None):
        select_infos = {"inds": [], "metrics": []}  # 选择特征的索引和系数字典
        for ind, coef_ in enumerate(self.est.coef_):  # 遍历模型系数
            select_infos["inds"].append(ind)  # 添加索引到列表
            select_infos["metrics"].append(coef_)  # 添加系数到列表
        return select_infos  # 返回所有特征的索引和系数

```



## 正式训练过程

```python
model.fit(X_train, y_train, X_test, y_test)
```

```python
    def _fit(self, X_train, y_train, X_val, y_val):
        start_time = time.time()
        # 在执行循环前需要先进行一些操作 todo 包括？封装数据，以及必要的调试信息
        data = self.execute_before_fit(X_train, y_train, X_val, y_val)
        # 在循环之前进行一些预处理操作, 比如归一化和 处理缺失值
        data = self.execute_pre_fit_processor(data)
        # 进行循环迭代, 获取
        for layer in range(1, self.max_num_iterations + 1, 1):
            # 在正式进行级联时执行一些操作，可以定义一些自己的特征选择方法
            data = self.pre_fit_cascade_data_and_infos(data, layer)
            # 获得的对应的基因筛选
            fselect_ids, fselect_infos = self.execute_feature_selector_processors(data, layer)
            # 对数据集执行特征筛选
            data = self.execute_fit_feature_selection(data, fselect_ids)
            # 保存筛选出的特征
            self.save_f_select_ids(fselect_ids, layer)
            # 执行特征融合处理 todo 没进来？怎么做到的
            data = self.execute_feature_and_data_fit_fusion(data, layer)
            # 使用数据生成局部数据
            data = self.split_fit_data_to_local(data, layer)
            # 对融合后的数据进行处理, 这个处理应该不涉及到样本数的改变
            data = self.execute_fit_fusion_features_processors(data, layer)
            # 对划分特征进行处理, 这个处理应该不涉及到样本数的改变
            data = self.execute_fit_split_features_processors(data, layer)
            # 对融合的数据执行类别不平衡的算法, 这个处理涉及到样本数的改变
            data = self.execute_category_imbalance(data, layer)
            # 更新构建器的配置信息
            builder_configs = self.obtain_new_update_builder_configs(data, layer)
            # 处理机器学习方法或深度学习方法的模块(使用全局数据) ， 开始进行分类 todo 找到哪里选择出基分类器
            classifier_instances = self.execute_cascade_fit_classifier(data, builder_configs, layer)
            # 保存提取到的特征, 概率特征, 和预测值
            all_finfos = self.obtain_relevant_fit_to_data(data, classifier_instances, layer)
            # 对特征进行处理
            all_finfos = self.execute_fit_feature_processors(all_finfos, layer)
            # 保存当前层提取到的提取到的特征
            self.save_relevant_fit_to_data(all_finfos, data, layer)
            # 可能需要对分类器进行一些调整, 比如排序, 筛除一些不合格的预测器
            # 在这里删除分类器的同时需要将对应的特征删除掉, 否则在预测模型的时候会无法产生对应的特征
            classifier_instances, data = self.adjust_cascade_classifier(classifier_instances, data)
            # 保存分类器
            self.save_cascade_classifier(classifier_instances, layer)
            # 计算当前层的终止指标
            metric = self.obtain_current_metric(data, layer)
            # 每次级联后是
            data = self.execute_post_fit_processor(data, layer)
            # 在进行级联前进行一些数据预处理
            data = self.post_fit_cascade_data_and_infos(data, layer)
            # 级联的层数判断
            if layer == 1:
                count = 0
                best_level, best_metric = layer, metric
                best_metric_processors = self.metrics_processor
            else:
                print("第 " + str(layer) + " 层的精度:", metric)

                if metric >= best_metric:
                    count = 0
                    best_level , best_metric = layer, metric
                    best_metric_processors = self.metrics_processor
                else:
                    count = count + 1

            if count >= self.termination_layer or layer == self.max_num_iterations:
                print("模型的层数 = ", best_level, "最佳的指标 = ", best_metric)
                self.best_level = best_level
                self.best_metric_processors = best_metric_processors
                break

        self.execute_after_fit(data)

        end_time = time.time()
        print("花费的时间:", end_time-start_time)

```



## 特征选择的过程

```python

    # 执行特征选择
    def fit_excecute(self, f_select_ids, f_select_infos, layer):
        X_train, y_train = f_select_infos["X_train"], f_select_infos["y_train"]  # 获取训练数据
        current_input_num = X_train.shape[1]  # 当前输入特征数量

        if self.select_inds is None or (self.original_num is not None and current_input_num != self.original_num) or self.enforcement:
            self.fit(X_train, y_train)  # 进行特征选择
            self.select_inds, self.select_infos = self._obtain_selected_index(X_train, y_train)  # 获取选择的特征索引和信息
            self.original_num = current_input_num  # 更新原始特征数量
            f_select_ids, selected_infos = self.select_inds, self.select_infos  # 更新选择的特征索引和信息
            selected_infos["Dim"] = len(f_select_ids)  # 更新选择的特征维度
        else:
            f_select_ids, selected_infos = self.select_inds, self.select_infos  # 使用已有的特征索引和信息
        return f_select_ids, f_select_infos
```





那么这里的 self 是谁呢？

Answer :  model.getselector();





## 特征召回的过程

```python
def fit_excecute(self, f_select_ids, f_select_infos, layer):
    assert f_select_ids != None, "当前层没有进行特征筛选模块，无法进行属性召回"

    # 总特征数量
    totall_feature_num = f_select_infos.get("Dim", None)

    f_select_num = len(f_select_ids)
    recall_ratio = self._obtain_recall_ratio(layer)
    # 进行特征召回的具体函数
    recall_num = int(recall_ratio * (totall_feature_num - f_select_num))

    all_attribute_ids = set(range(totall_feature_num))
    no_selected_ids = all_attribute_ids - set(f_select_ids)
    assert 0 <= recall_num <= len(no_selected_ids), "召回特征的数量不能超过未选择的特征数量"

    recall_ids = random.sample(list(no_selected_ids), recall_num)
    f_select_ids = recall_ids + f_select_ids

    f_select_infos["RecallNum"] = recall_num

    return f_select_ids, f_select_infos
```





## Dispatcher -调度器



Dispatcher 类及其子类的设计目的是为了**动态地创建和管理各种处理器、选择器、分类器等实例**，并确保这些实例符合统一的接口或模板（通过 Template 类）。它提供了一个通用的、模块化的方式来实例化和使用不同类型的处理器、选择器和分类器，以下是其具体设计目的和工作机制。

**设计目的**

​	1.	**动态创建实例**：Dispatcher 类及其子类通过配置（config）参数，根据不同类型（如 FeatureSelector、FeatureProcessor 等）和名称，动态创建不同的实例对象。

​	2.	**封装实例化逻辑**：不同的子类（如 FeatFusionDispatcher、CategoryImbalanceDispatcher）封装了特定处理器的创建逻辑，利用 Dispatcher 作为基类，使代码结构更清晰、模块化。

​	3.	**确保实例符合模板**：通过 Template 模板类检查创建的实例对象是否符合预期接口和功能。_obtain_instance 方法负责检查返回的对象是否继承了指定的 Template 类，从而确保所有实例都符合接口规范。

​	4.	**扩展性**：可以轻松添加新的处理器或选择器类型，而不需要改变现有代码。只需新增一个具体的 Dispatcher 子类并实现特定的 execute_dispatcher_method 方法。



**核心工作机制**



​	•	obtain_instance **和** _obtain_instance **方法**：

​	•	obtain_instance 是公开的接口，接受一个 config 配置字典，从中提取实例的 name 和 type，然后调用 _obtain_instance 来生成具体实例。

​	•	_obtain_instance 使用 execute_dispatcher_method 根据名称和类型创建实例，并检查其是否是 Template 的子类。

​	•	execute_dispatcher_method **的分派逻辑**：

​	•	每个具体 Dispatcher 子类通过实现 execute_dispatcher_method 来定义不同的创建逻辑。例如，FeatFusionDispatcher 使用 get_feature_concatenation_method 来创建特征融合方法，MetricProcessorDispatcher 使用 get_metric_processor 来创建评估处理器。

​	•	execute_dispatcher_method 是创建实例的核心，基于名称和类型调用适当的工厂函数或方法来生成对应实例。



**各具体子类的作用**



​	1.	FeatFusionDispatcher：用于创建特征融合方法的调度器。

​	2.	MetricProcessorDispatcher：用于创建评估处理器的实例。

​	3.	CategoryImbalanceDispatcher：用于创建类别不平衡处理器。

​	4.	FeatureSplitDispatcher：用于创建特征拆分处理器。

​	5.	ListDispatcher：扩展 Dispatcher 以处理包含多个实例的配置列表（如多种特征选择器）。

​	6.	PreProcessorDispatcher **和** PostProcessorDispatcher：分别用于创建预处理和后处理方法的实例。

​	7.	FeatSelectorDispatcher：用于创建特征选择器调度器，支持 FeatureSelection 和 RecallAttribute 等类型的特征选择器。

​	8.	ClassifierDispatcher **和** GlobalAndLocalClassifierDispatcher：用于创建全局和本地调度器，区分不同数据范围的分类任务（全局和本地）。



**总结**

Dispatcher 类提供了一个通用的、灵活的实例创建框架，并通过继承和多态实现了针对不同类型的处理器、选择器、分类器的动态实例化。各个具体 Dispatcher 子类将具体实例创建的逻辑分开，实现了良好的模块化和扩展性，使得在代码中灵活使用不同的处理器、选择器和分类器成为可能。

**细节补充**

在 Python 中，通过在类定义的括号中声明父类可以明确一个类的继承关系。例如：

```python
class FeatFusionDispatcher(Dispatcher):
    def __init__(self):
        # 初始化逻辑
```

这里的super.__ init  __  （）表示调用父类的构造函数，并且使用FeatureSelectorTemplate 作为参数（模版）进行传入

```python
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
```



## 添加mac的 mps 进行训练



问题： cpu占用率90以上， gpu 几乎为0 ；怀疑是不是代码的结构的问题

训练集大小: 2784 , 维度: 24174



## 一些问题

//todo 

### 1 提问：

1. 为什么没有第一层 - 没有打印出来
2. 为什么有平台

![image-20241031173058415](https://markdown-pictures-jhx.oss-cn-beijing.aliyuncs.com/picgo/image-20241031173058415.png)

没有打印出来而已





### 2 提问：

我理解是，FeatFusionDispatcher得到一个对象， 然后调用了继承父类的obtain_instance方法，进一步调用了_obtain_instance 方法，应该是内部方法吧；然后调用了execute_dispatcher_method方法，但是父类的    def execute_dispatcher_method(self, name, est_type, configs):        pass； 为什么会执行到子类的   def execute_dispatcher_method(self, name, est_type, config):        return get_feature_concatenation_method(name, est_type, config) 呢？？？

这是典型的面向对象编程（OOP）中的**多态**和**方法重写**机制在起作用。即便父类 `Dispatcher` 的 `execute_dispatcher_method` 是一个空实现（`pass`），但子类 `FeatFusionDispatcher` 对该方法进行了重写。当通过 `FeatFusionDispatcher` 的实例调用 `execute_dispatcher_method` 时，会自动调用子类中重写后的版本。

具体原因如下：

1. **类的继承与方法重写**：
    `FeatFusionDispatcher` 类继承自 `Dispatcher` 类。
    父类 `Dispatcher` 中有一个 `execute_dispatcher_method` 方法实现为 `pass`，相当于空实现。
    子类 `FeatFusionDispatcher` 对该方法进行了重写（即重新定义了同名方法），这样子类就有了自己的实现逻辑。

2. **实例方法调用的绑定**：
    当你创建一个 `FeatFusionDispatcher` 的实例，例如：

   ```python
   feat_fusion_dispatcher = FeatFusionDispatcher()
   ```

   此实例是 `FeatFusionDispatcher` 类型的对象。即使 `FeatFusionDispatcher` 没有自己的 `obtain_instance` 方法实现，而是使用了父类 `Dispatcher` 提供的 `obtain_instance` 方法，这个实例仍然是 `FeatFusionDispatcher` 的实例，包含子类对方法的重写。

3. **方法调用时的动态绑定**：
    当你调用：

   ```python
   self.feature_fusions = feat_fusion_dispatcher.obtain_instance(fusion_configs)
   ```

   内部执行到 `_obtain_instance` 方法时（定义在 `Dispatcher` 中），最终会调用：

   ```python
   est = self.execute_dispatcher_method(name, est_type, config)
   ```

   此时的 `self` 是 `FeatFusionDispatcher` 的实例。Python 根据**方法解析顺序（MRO）**和多态特性，会从 `FeatFusionDispatcher` 中去找 `execute_dispatcher_method` 的定义。
    因为 `FeatFusionDispatcher` 提供了自己的 `execute_dispatcher_method`，所以调用的就是子类的实现：

   ```python
   def execute_dispatcher_method(self, name, est_type, config):
       return get_feature_concatenation_method(name, est_type, config)
   ```

   这就是为什么最终会执行到子类的 `execute_dispatcher_method` 而不是父类的空实现。

总结来说：**你实例化的是子类，子类重写了父类方法，因此在运行时调用 `execute_dispatcher_method` 时，会自动使用子类的实现，而不是父类的空实现。**

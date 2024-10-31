import copy
import time
import numpy as np
from sklearn import metrics

from sklearn.model_selection import train_test_split

import warnings

from Processor.ProcessorDispatcher.Dispatcher import FeatFusionDispatcher, FeatSelectorDispatcher, \
    FeaturesProcessorDispatcher, MetricProcessorDispatcher, ClassifierDispatcher, CategoryImbalanceDispatcher, \
    PreProcessorDispatcher, PostProcessorDispatcher, FusionFeatDispatcher, SplitFeatureDispatcher, \
    FeatureSplitDispatcher

warnings.filterwarnings("ignore")

class UnimodalModel():

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

    def _init_pre_processor(self, configs):
        processor_configs = configs.get("PreProcessors", None)
        if processor_configs != None:
            pre_processor_dispatcher = PreProcessorDispatcher()
            self.pre_processors = pre_processor_dispatcher.obtain_instance(processor_configs)
        else:
            self.pre_processors = []

    def _init_feature_selectors(self, configs):
        feature_selector_configs = configs.get("FeatureSelector", None)
        if feature_selector_configs != None:
            feat_selector_dispatcher = FeatSelectorDispatcher()
            self.feature_selectors = feat_selector_dispatcher.obtain_instance(feature_selector_configs)
        else:
            self.feature_selectors = []

    def _init_data_and_feature_fusion(self, configs):
        fusion_configs = configs.get("FeatureFusion", None)
        if fusion_configs != None:
            feat_fusion_dispatcher = FeatFusionDispatcher()
            self.feature_fusions = feat_fusion_dispatcher.obtain_instance(fusion_configs)
        else:
            raise "必须配置特征融合方法"

    def _init_fusion_feature_processors(self, configs):
        fusion_features_processors_configs = configs.get("FusionFeatureProcessors", None)
        if fusion_features_processors_configs != None:
            fusion_feat_dispatcher = FusionFeatDispatcher()
            self.fusion_features_processors = fusion_feat_dispatcher.obtain_instance(fusion_features_processors_configs)
        else:
            self.fusion_features_processors = []

    def _init_feature_split_processor(self, configs):
        feature_split_processor_config = configs.get("FeatureSplitProcessor", None)
        if  feature_split_processor_config != None:
            feature_split_dispatcher = FeatureSplitDispatcher()
            self.feature_split_processor = feature_split_dispatcher.obtain_instance(feature_split_processor_config)
        else:
            self.feature_split_processor = None

    def _init_split_feature_processors(self, configs):
        split_feature_processors_config = configs.get("SplitFeatureProcessors", None)
        if split_feature_processors_config != None:
            split_feature_dispatcher = SplitFeatureDispatcher()
            self.split_feature_processors = split_feature_dispatcher.obtain_instance(split_feature_processors_config)
        else:
            self.split_feature_processors = None

    def _init_category_imbalance_processors(self, config):
        category_imbalance_config = config.get("CategoryImbalance", None)
        if category_imbalance_config != None:
            category_imbalance_dispatcher = CategoryImbalanceDispatcher()
            self.category_imbalance_processor = category_imbalance_dispatcher.obtain_instance(category_imbalance_config)
        else:
            self.category_imbalance_processor = None

    def _init_cascade_classifier_builder(self, configs):
        builder_configs = configs.get("CascadeClassifier", None)
        if builder_configs != None:
            builder_dispatcher = ClassifierDispatcher()
            global_classifier_builders, local_classifier_builders = builder_dispatcher.obtain_instance(builder_configs)
            self.global_classifier_builders = global_classifier_builders
            self.local_classifier_builders = local_classifier_builders
        else:
            raise "分类器不能为空"

    def _init_cascade_features_processors(self, config):
        feature_processors_configs = config.get("FeatureProcessors", None)
        if feature_processors_configs != None:
            features_processor_dispatcher = FeaturesProcessorDispatcher()
            self.feature_processors = features_processor_dispatcher.obtain_instance(feature_processors_configs)
        else:
            self.feature_processors = []

    def _init_cascade_metrics_processor(self, config):
        metrics_processor_configs = config.get("MetricsProcessors", None)
        if metrics_processor_configs != None:
            metric_processor_dispatcher = MetricProcessorDispatcher()
            self.metrics_processor = metric_processor_dispatcher.obtain_instance(metrics_processor_configs)
        else:
            raise "计算每层的指标不能设置为空"

    def _init_post_processor(self, config):
        post_processors_configs = config.get("PostProcessors", None)
        if post_processors_configs != None:
            post_processor_dispatcher = PostProcessorDispatcher()
            self.post_processors = post_processor_dispatcher.obtain_instance(post_processors_configs)
        else:
            self.post_processors = []

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if y_val is None:
            division_ratio = self.config.get("DivisionRatio", 0.8)
            if self.debug:
                print("没有设置验证集， 随机从训练集中进行划分, 划分比例为:", division_ratio)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=division_ratio)
        # 正式进行训练
        self._fit(X_train, y_train, X_val, y_val)

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
            # 特征选择 以及特征选择器的创建
            fselect_ids, fselect_infos = self.execute_feature_selector_processors(data, layer)
            # 根据筛选出的特征 ，对原有数据集进行变换
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

    def save_relevant_fit_to_data(self, all_finfos, data, layer):
        data["Finfos"][layer] = all_finfos

    def execute_before_fit(self, X_train, y_train, X_val, y_val):
        if self.debug:
            print("==================执行循环前的预处理开始==================")

        # 对运行时中的数据部分进行一些操作
        data = dict()
        size, dim = X_train.shape
        data["Original"] = dict(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                                Size=size, Dim=dim)
        data["Finfos"] = dict() #todo ？ feature infomation？

        if self.debug:
            print("训练集大小:", size, ", 维度:", dim)

        return data

    def execute_pre_fit_processor(self, data):
        if self.debug:
            pre_processor_names = []

        # 执行预处理代码
        for pre_processor in self.pre_processors:
            if pre_processor.executable():
                if self.debug:
                   pre_processor_names.append(pre_processor.obtain_name())
                data = pre_processor.fit_excecute(data)

        if self.debug:
            pre_processor_num = len(pre_processor_names)
            print("循环前的预处理器数量:", pre_processor_num)
            if pre_processor_num > 0:
                print("循环前的预处理器的名字:", pre_processor_names)
            print("==================执行循环前的预处理结束==================")

        return data

    def pre_fit_cascade_data_and_infos(self, data, layer):
        print("==================第" + str(layer) + "层开始执行==================")

        # 这些方法都是空实现, 目的是为以后可以在级联的过程中自适应的修改特征筛选器 todo 这里好像就可以实现成自己的特征筛选器，例如在奇数层可以使用GCN特征筛选
        self.change_feature_selectors(data, layer)
        self.change_feature_fusions(data, layer)
        self.change_fusion_features_processors(data, layer)
        self.change_split_features_processors(data, layer)
        self.change_category_imbalance_processor(data, layer)
        self.change_global_classifier_builders(data, layer)
        self.change_feature_processors(data, layer)
        self.change_metric_processor(data, layer)
        self.change_feature_types(data, layer)

        return data

    def change_feature_selectors(self, data, layer):
        pass

    def change_feature_fusions(self, data, layer):
        pass

    def change_fusion_features_processors(self, data, layer):
        pass

    def change_split_features_processors(self, data, layer):
        pass

    def change_category_imbalance_processor(self, data, layer):
        pass

    def change_global_classifier_builders(self, data, layer):
        pass

    def change_feature_processors(self, data, layer):
        pass

    def change_metric_processor(self, data, layer):
        pass

    def change_feature_types(self, data, layer):
        pass

    def execute_feature_selector_processors(self, data, layer):
        if self.debug:
            print("==================特征筛选算法开始执行==================")
            feature_selector_names = []

        # 获得用于训练特征筛选器的数据集
        original_data = data.get("Original")
        X_train, y_train, dim = original_data["X_train"], original_data["y_train"], original_data["Dim"]
        f_select_infos = dict(X_train=X_train, y_train=y_train, Dim=dim)
        # 执行特征筛选
        f_select_idxs = None
        #在本例中 只有一个GCLasso
        for feature_selector in self.feature_selectors:
            if feature_selector.fit_executable(layer):
                if self.debug:
                    feature_selector_names.append(feature_selector.obtain_name())
                f_select_idxs, f_select_infos = feature_selector.fit_excecute(f_select_idxs, f_select_infos, layer)

        if self.debug:
            feature_selector_num = len(feature_selector_names)
            print("使用的特征筛选器的数量为:", feature_selector_num)
            if feature_selector_num > 0:
                print("使用的特征筛选器的名字分别是", feature_selector_names)

                if f_select_infos.get("NewTeatures"):
                    print(feature_selector_names[0] + "重新进行了特征筛选")
                else:
                    print(feature_selector_names[0] + "使用之前筛选好的特征")

                recall_num = f_select_infos.get("RecallNum", None)
                if recall_num is not None:
                    print("属性召回模块召回的数量是:" + str(recall_num))

                print("最终获得的筛选特征数量是: " + str(len(f_select_idxs)))

        return f_select_idxs, f_select_infos

    def execute_fit_feature_selection(self, data, f_select_idxs):
        if f_select_idxs is not None:
            data["Processed"] = dict()
            data["Processed"]["X_train"] = copy.deepcopy(data["Original"]["X_train"][:, f_select_idxs])
            data["Processed"]["y_train"] = copy.deepcopy(data["Original"]["y_train"])
            data["Processed"]["X_val"] = copy.deepcopy(data["Original"]["X_val"][:, f_select_idxs])
            data["Processed"]["y_val"] = copy.deepcopy(data["Original"]["y_val"])
        else:
            data["Processed"] = copy.deepcopy(data["Original"])

        if self.debug:
            print("==================特征筛选算法执行完成==================")

        return data

    def save_f_select_ids(self, f_select_ids, layer):
        self.f_select_ids[layer] = f_select_ids

    def execute_feature_and_data_fit_fusion(self, data, layer):

        if self.feature_fusions.executable(layer):

            if self.debug:
                print("==================特征融合开始执行==================")
                print("特征融合方法: ", self.feature_fusions.obtain_name())

            # 获得原始数据 (经过特征筛选的数据)
            original_data = data.get("Processed")
            original_train, original_val = original_data.get("X_train"), original_data.get("X_val")
            # 获得特征信息
            finfos = data.get("Finfos")
            # 进行特征融合
            fusion_train, fusion_val = self.feature_fusions.fit_excecute(original_train, original_val, finfos, layer)
            # 将融合的数据封装回去
            data["Processed"]["X_train"] = fusion_train
            data["Processed"]["X_val"] = fusion_val

            if self.debug:
                print("==================特征融合执行完成==================")

        return data

    def split_fit_data_to_local(self, data, layer):

        if "Local" not in self.feature_types:
            return data

        if self.feature_split_processor is None:
            raise "当你使用局部特征时，其特征切分器必须要设置"

        if self.feature_split_processor.executable(layer):

            if self.debug:
                print("==================特征划分开始执行==================")
                print("使用的特征划分方式的名字是:", self.feature_split_processor.obtain_name())

            # 获得原始数据 (经过特征筛选的数据)
            processed_data = data.get("Processed")
            X_train, X_val = processed_data.get("X_train"), processed_data.get("X_val")
            y_train, y_val = processed_data.get("y_train"), processed_data.get("y_val")
            # 进行特征融合
            X_split_train, X_split_val, split_finfo = self.feature_split_processor.fit_excecute(X_train, X_val, layer)
            if split_finfo.get("isSuccess"):
                # 将融合的数据封装回去
                data["SplitFeature"] = dict(Xs_train=X_split_train, Xs_val=X_split_val, y_train=y_train, y_val=y_val)
            else:
                print("划分失败, 失败信息" + split_finfo.get("FailureInfo"))

            if self.debug:
                print("==================特征划分执行完成==================")

        return data

    def execute_fit_fusion_features_processors(self, data, layer):
        if self.debug:
            print("==================全局融合特征执行开始==================")
            fusion_features_processor_names = []

        # 获得原始数据 (经过特征筛选的数据)
        processed_data = data.get("Processed")
        X_train, X_val = processed_data.get("X_train"), processed_data.get("X_val")

        for fusion_features_processor in self.fusion_features_processors:
            if fusion_features_processor.executable(layer):
                if self.debug:
                    fusion_features_processor_names.append(fusion_features_processor.obtain_name())
                X_train = fusion_features_processor.excecute(X_train, layer)
                X_val = fusion_features_processor.excecute(X_val, layer)

        # 将融合的数据封装回去
        data["Processed"]["X_train"] = X_train
        data["Processed"]["X_val"] = X_val

        if self.debug:
            fusion_features_processor_num = len(fusion_features_processor_names)
            print("使用的融合特征处理器的数量为:", fusion_features_processor_num)
            if fusion_features_processor_num > 0:
                print("使用的特征融合器的名字分别是", fusion_features_processor_names)
            print("==================全局融合特征执行完成==================")

        return data

    def execute_fit_split_features_processors(self, data, layer):

        if "Local" not in self.feature_types:
            return data

        if self.split_feature_processors is None or len(self.split_feature_processors):
            print("没有设置切分特征处理器, 或者切分特征处理器为空")
            return data

        if self.debug:
            print("==================局部融合特征执行开始==================")
            split_feature_processors_name = []

        # 获得原始数据 (经过特征筛选的数据)
        processed_data = data.get("SplitFeature")
        Xs_train, Xs_val = processed_data.get("Xs_train"), processed_data.get("Xs_val")
        for split_feature_processor in self.split_feature_processors:
            if split_feature_processor.executable(layer):
                if self.debug:
                    split_feature_processors_name.append(split_feature_processor.obtain_name())
                Xs_train, Xs_val = split_feature_processor.excecute(Xs_train, Xs_val, layer)

        data["SplitFeature"]["Xs_train"] = Xs_train
        data["SplitFeature"]["Xs_val"] = Xs_val

        if self.debug:
            split_feature_processors_num = len(split_feature_processors_name)
            print("使用的局部融合特征处理器的数量为:", split_feature_processors_num)
            if split_feature_processors_num > 0:
                print("使用的局部特征融合器的名字分别是", split_feature_processors_name)
            print("==================局部融合特征执行完成==================")

        return data

    def obtain_new_update_builder_configs(self, data, layer):
        new_cfig = dict()
        if "Global" in self.feature_types:
            new_cfig["Global"] = self.obtain_new_update_global_builder_configs(data)
        if "Local" in self.feature_types:
            new_cfig["Local"] = self.obtain_new_update_local_builder_configs(data)
        return new_cfig

    def obtain_new_update_global_builder_configs(self, data):
        new_global_cfig = dict()
        for classifier_builder in self.global_classifier_builders:
            builder_name = classifier_builder.obtain_name()
            builder_type = classifier_builder.obtain_builder_type()
            if builder_type == "DL":
                input_size = data["Processed"]["X_train"].shape[1]
                new_global_cfig[builder_name] = dict(Model=dict(Parameter=dict(InputSize=input_size)))
        return new_global_cfig

    def obtain_new_update_local_builder_configs(self, data):
        new_lobal_cfig = dict()
        for classifier_builder in self.local_classifier_builders:
            builder_name = classifier_builder.obtain_name()
            builder_type = classifier_builder.obtain_builder_type()

            if builder_type == "DL":
                feature_splits = data["SplitFeature"]["Xs_train"]
                input_size = [feat.shape[1] for feat in feature_splits]
                new_lobal_cfig[builder_name] = dict(Model=dict(Parameter=dict(InputSize=input_size)))
        return new_lobal_cfig

    def execute_category_imbalance(self, data, layer):
        if self.category_imbalance_processor is None:
            return data

        if self.debug:
            print("==================类别不平衡器执行开始==================")
            print(self.category_imbalance_processor.obtain_name())

        if self.category_imbalance_processor.fit_executable(layer):
            # 训练数据集的解析
            processed_data = data.get("Processed", None)
            X_train, y_train = processed_data["X_train"], processed_data["y_train"]

            X_train_res, y_train_res = self.category_imbalance_processor.fit_excecute(X_train, y_train, layer)

            # 封装新的数据
            data["Processed"]["X_train_res"] = X_train_res
            data["Processed"]["y_train_res"] = y_train_res

        if self.debug:
            print("==================类别不平衡器执行完成==================")

        return data

    def execute_cascade_fit_classifier(self, data, builder_configs, layer):
        if self.debug:
            print("==================分类器执行开始==================")

        # 获得全局单模态分类器 (机器学习方法和深度学习方法)
        if "Global" in self.feature_types:
            global_data = data.get("Processed")
            if len(self.global_classifier_builders) > 0 :
                # 获取相应的数据信息
                X_train_res = global_data.get("X_train_res", global_data["X_train"])
                y_train_res = global_data.get("y_train_res", global_data["y_train"])
                X_val, y_val = global_data["X_val"], global_data["y_val"]
                global_builder_configs = builder_configs.get("Global", None)
                # 训练全局训练器
                global_classifier_instances = self.obtain_fit_classifier_instance(X_train_res, y_train_res, X_val, y_val,
                                                        global_builder_configs, self.global_classifier_builders, layer)
                global_instances_num = len(global_classifier_instances)
            else:
                global_classifier_instances = None
                global_instances_num = 0
        else:
            # 这是因为局部特征也是由全局特征转化而来的, 如果全局特征为空时, 则局部特征也一定为空
            global_classifier_instances = None
            global_instances_num = 0

        # 获得局部单模态分类器 (机器学习方法和深度学习方法)
        if "Local" in self.feature_types:
            # 获取相应的数据信息
            local_data = data.get("SplitFeature", None)
            X_train_res = local_data.get("Xs_train_res", local_data["Xs_train"])
            y_train_res = local_data.get("y_train_res", local_data["y_train"])
            X_val, y_val = local_data["Xs_val"], local_data["y_val"]
            local_builder_configs = builder_configs.get("Local", None)
            # 训练全局训练器
            local_classifier_instances = self.obtain_fit_classifier_instance(X_train_res, y_train_res, X_val, y_val,
                                                        local_builder_configs, self.local_classifier_builders, layer)
            local_instances_num = len(local_classifier_instances)

        else:
            local_classifier_instances = None
            local_instances_num = 0

        # 检查当前分类器的总个数
        totall_num = global_instances_num + local_instances_num
        if totall_num == 0:
            raise "当前层最终获得的分类器数量为 0, 请重新检查配置信息"

        # 封装全局分类器和局部分类器
        classifier_instances = {"Global" : global_classifier_instances, "Local" : local_classifier_instances,
                     "GlobalNum" : global_instances_num, "LocalNum" : local_instances_num, "TotallNum" : totall_num}

        if self.debug:
            print("训练好的分类器(或特征提取器)总共有" + str(totall_num) + "个, 其中", end=",")
            print("全局分类器(或特征提取器)有" + str(global_instances_num) + "个",  end=",")
            print("局部分类器(或特征提取器)有" + str(local_instances_num) + "个")
            print("==================分类器执行完成==================")

        return classifier_instances

    # def obtain_fit_classifier_instance(self, X_train, y_train, X_val, y_val, builder_configs, classifier_builders, layer):
    #     classifier_instances = {}
    #     for classifier_builder in classifier_builders:
    #
    #         if classifier_builder.fit_executable(layer):
    #
    #             builder_name = classifier_builder.obtain_name()
    #
    #             if self.debug:
    #                 print("==================" + builder_name + " 开始构建分类器 ==================")
    #
    #             builder_config = self.obtain_builder_config(builder_configs, builder_name)
    #             if builder_config != None:
    #                 classifier_builder.update_config(builder_config, layer)
    #
    #             classifier_name, classifier_instance = classifier_builder.obtain_fit_classifier(X_train, y_train, X_val, y_val, layer)
    #             # 获得分类器
    #             classifier_instances[classifier_name] = classifier_instance
    #
    #             if self.debug:
    #                 print("=================="+ builder_name + "分类器构建完成 ==================")
    #
    #     return classifier_instances

    def obtain_fit_classifier_instance(self, X_train, y_train, X_val, y_val, builder_configs, classifier_builders,
                                       layer):
        classifier_instances = {}
        from tqdm import tqdm

        # 创建 tqdm 进度条
        progress_bar = tqdm(classifier_builders, desc="Building classifiers")

        for classifier_builder in progress_bar:

            if classifier_builder.fit_executable(layer):

                builder_name = classifier_builder.obtain_name()

                if self.debug:
                    print("==================" + builder_name + " 开始构建分类器 ==================")

                builder_config = self.obtain_builder_config(builder_configs, builder_name)
                if builder_config is not None:
                    classifier_builder.update_config(builder_config, layer)

                classifier_name, classifier_instance = classifier_builder.obtain_fit_classifier(X_train, y_train, X_val,
                                                                                                y_val, layer)
                # 获得分类器
                classifier_instances[classifier_name] = classifier_instance

                if self.debug:
                    print("==================" + builder_name + "分类器构建完成 ==================")

                # 更新进度条描述
                progress_bar.set_postfix({"Last Built": builder_name})

        return classifier_instances

    def obtain_builder_config(self, builder_configs, builder_name):
        return builder_configs.get(builder_name, None)

    def obtain_relevant_fit_to_data(self, data, classifier_instances, layer):

        if self.debug:
            print("==================开始获取相关信息(包括提取到的特征, 预测的概率值, 预测值等)==================")

        all_finfos = dict()

        global_instances_num = classifier_instances["GlobalNum"]
        # 没有特征实例, 保存空字典就行
        if global_instances_num == 0:
            if self.debug:
                print("没有设置全局分类器或特征提取器")
        else:
            global_finfos = dict()

            global_data = data.get("Processed")
            X_train, y_train = global_data["X_train"], global_data["y_train"]
            X_val, y_val = global_data["X_val"], global_data["y_val"]

            global_classifier_instances = classifier_instances.get("Global")

            for classifier_name, classifier_instance in global_classifier_instances.items():
                finfos = self.obtain_current_layer_fit_features(X_train, y_train, X_val, y_val, classifier_instance, layer)
                global_finfos[classifier_name] = finfos

            all_finfos.update(global_finfos)

            if self.debug:
                global_features_num = len(global_finfos)
                if global_features_num > 0:
                    print("执行全局的特征提取器的过程结束, 最终获得的特征数量为:", global_features_num)
                    print("获得的特征名字有: ", list(global_finfos.keys()))
                    print("每个特征的属性有", list(list(global_finfos.values())[0].keys()))

        local_instances_num = classifier_instances.get("LocalNum")
        # 没有特征实例, 保存空字典就行
        if local_instances_num == 0:
            if self.debug:
                print("没有设置局部分类器或特征提取器")
        else:
            local_finfos = dict()

            split_feature = data.get("SplitFeature", None)
            Xs_train, y_train = split_feature["Xs_train"], split_feature["y_train"]
            Xs_val, y_val = split_feature["Xs_val"], split_feature["y_val"]

            local_classifier_instances = classifier_instances.get("Local")

            for classifier_name, classifier_instances in local_classifier_instances.items():
                for index, classifier_instance in classifier_instances.items():
                    finfos = self.obtain_current_layer_fit_features(Xs_train[index], y_train, Xs_val[index], y_val,
                                                                    classifier_instance, layer)
                    classifier_name_id = classifier_name + "&" + str(index)
                    local_finfos[classifier_name_id] = finfos

            all_finfos.update(local_finfos)

            if self.debug:
                local_features_num = len(local_finfos)
                if local_features_num > 0:
                    print("执行局部的特征结束, 最终获得的特征数量为:", local_features_num)
                    print("获得的特征名字有: ", list(local_finfos.keys()))
                    print("每个特征的属性有", list(list(local_finfos.values())[0].keys()))

        if self.debug:
            print("==================获取相关信息结束(包括提取到的特征, 预测的概率值, 预测值等)==================")

        return all_finfos

    def obtain_current_layer_fit_features(self, X_train, y_train, X_val, y_val, classifier_instance, layer):
        # 真正执行特征提取的地方
        cls_name = classifier_instance.obtain_name()
        builder_type = classifier_instance.obtain_builder_type()
        est_type = classifier_instance.obtain_est_type()
        data_type = classifier_instance.obtain_data_type()

        if classifier_instance.can_obtain_features(layer):
            features_train = classifier_instance.obtain_features(X_train)
            features_val = classifier_instance.obtain_features(X_val)
        else:
            features_train, features_val = None, None

        if classifier_instance.can_obtain_probs(layer):
            probs = classifier_instance.predict_probs(X_val)
            predict = classifier_instance.predict(X_val)
        else:
            probs, predict = None, None

        finfos = dict(ClassifierName=cls_name, EstType=est_type, BuilderType=builder_type, DataType=data_type, Layer=layer,
                      Feature_train=features_train, Feature_val=features_val, Predict=predict, Probs=probs)

        if self.debug:
            print("分类器:" + cls_name + ", 数据类型:" + data_type + "的性能指标:")
            if probs is not None:
                print(metrics.classification_report(y_val, predict, digits=4))

        return finfos

    def adjust_cascade_classifier(self, classifier_instances, data):
        return classifier_instances, data

    def save_cascade_classifier(self, classify_instances, layer):
        self.classifier_instances[layer] = classify_instances

    def execute_fit_feature_processors(self, features, layer):

        if self.debug:
            print("==============开始执行特征处理器===============")
            feature_processors_names = []

        for feature_processor in self.feature_processors:
            if feature_processor.executable(layer):
                if self.debug:
                    feature_processors_names.append(feature_processor.obtain_name())
                features = feature_processor.fit_excecute(features, layer)

        if self.debug:
            feature_processors_num = len(feature_processors_names)
            print("特征提取器的执行数量:", feature_processors_num)
            if feature_processors_num > 0:
                print("特征提取器的执行的名字分别是: ", feature_processors_names)
            print("==============级联后置处理器执行完成===============")

        return features

    def obtain_current_metric(self, data, layer):
        if self.metrics_processor != None:
            if self.debug:
                print("==============开始执行指标计算器===============")
                print("指标计算器的名字", self.metrics_processor.obtain_name())

            metric = self.metrics_processor.fit_excecute(data, layer)

            if self.debug:
                print("==============指标计算器执行完成===============")

            return metric
        else:
            raise "当前层没有设置指标计算器"

    def execute_post_fit_processor(self, data, layer):

        if self.debug:
            print("==============开始执行级联后置处理器===============")
            post_processors_names = []

        for post_processor in self.post_processors:
            if post_processor.fit_executable(layer):
                if self.debug:
                    post_processors_names.append(post_processor.name)
                data = post_processor.fit_excecute(data, layer)

        if self.debug:
            post_processors_num = len(post_processors_names)
            print("级联后置处理器的执行数量:", post_processors_num)
            if post_processors_num > 0:
                print("级联后置处理器的执行的名字分别是: ", post_processors_names)
            print("==============级联后置处理器执行完成===============")

        return data

    def post_fit_cascade_data_and_infos(self, data, layer):
        print("==============第" + str(layer) + "层执行结束===============")
        self.all_feature_split_processors[layer] = copy.deepcopy(self.feature_split_processor)
        self.all_feature_fusions_processors[layer] = copy.deepcopy(self.feature_fusions)
        self.all_split_feature_processors[layer] = copy.deepcopy(self.split_feature_processors)
        self.all_fusion_feature_processors[layer] = copy.deepcopy(self.fusion_features_processors)
        self.all_feature_processors[layer] = copy.deepcopy(self.feature_processors)
        self.all_metrics_processor[layer] = copy.deepcopy(self.metrics_processor)
        self.all_feature_types[layer] = copy.deepcopy(self.feature_types)
        return data

    def execute_after_fit(self, data):
       pass

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        start_time = time.time()
        # 在执行循环前需要先进行一些操作
        data = self.execute_before_predict_probs(X)
        # 执行一些数据预处理步骤
        data = self.execute_pre_predict_processor(data)
        # 进行级联
        for layer in range(1, self.best_level+1, 1):
            # 在每层进行级联时执行一些相关操作
            data = self.pre_predict_cascade_data_and_infos(data, layer)
            # 获得筛选的特征
            f_select_ids = self.obtain_cascade_f_select_ids(layer)
            # 对数据集执行特征筛选
            data = self.execute_predict_feature_selection(data, f_select_ids)
            # 对融合的特征进行处理
            data = self.execute_feature_and_data_predict_fusion(data, layer)
            # 使用数据生成局部数据
            data = self.split_predict_data_to_local(data, layer)
            # 对融合后的数据进行处理, 这个处理应该不涉及到样本数的改变
            data = self.execute_predict_fusion_features_processors(data, layer)
            # 对划分特征进行处理, 这个处理应该不涉及到样本数的改变
            data = self.execute_predict_split_features_processors(data, layer)
            # 处理机器学习方法或深度学习方法的模块
            classifier_instances = self.obtain_cascade_predict_classifier_instance(layer)
            # 提取特征
            all_finfos = self.obtain_relevant_to_predict_data(data, classifier_instances, layer)
            # 对特征进行处理
            all_finfos = self.execute_predict_feature_processors(all_finfos, layer)
            # 保存提取到的特征
            self.save_relevant_to_predict_data(all_finfos, data, layer)

            if layer == self.best_level:
                probs = self.best_metric_processors.predict_execute(data, layer)
                break

            # 在进行级联前进行一些数据预处理
            data = self.post_predict_cascade_data_and_infos(data, layer)

        self.execute_after_predict_probs(data)

        end_time = time.time()
        run_time = end_time - start_time
        print("预测所花费的时间:" , run_time)
        return probs

    def save_relevant_to_predict_data(self, all_finfos, data, layer):
        # 保存当前层提取到的提取到的特征
        data["Finfos"][layer] = all_finfos

    def execute_before_predict_probs(self, X):
        # 对运行时中的数据部分进行一些操作
        data = dict()
        size, dim = X.shape
        data["Original"] = dict(X=X, Size=size, Dim=dim)
        data["Finfos"] = dict()
        # 对运行期间的一些信息进行保存
        if self.debug:
            print("测试集的大小:", size, ", 维度:", dim)
        return data

    def execute_pre_predict_processor(self, data):
        # 执行预处理代码
        for pre_processor in self.pre_processors:
            if pre_processor.executable():
                data = pre_processor.predict_excecute(data)
        return data

    def pre_predict_cascade_data_and_infos(self, data, layer):
        self.feature_split_processor = self.all_feature_split_processors[layer]
        self.feature_fusions = self.all_feature_fusions_processors[layer]
        self.split_feature_processors = self.all_split_feature_processors[layer]
        self.fusion_features_processors = self.all_fusion_feature_processors[layer]
        self.feature_processors = self.all_feature_processors[layer]
        self.metrics_processor = self.all_metrics_processor[layer]
        return data

    def obtain_cascade_f_select_ids(self, layer):
        return self.f_select_ids[layer]

    def execute_predict_feature_selection(self, data, f_select_ids):
        if f_select_ids != None:
            data["Processed"] = dict()
            data["Processed"]["X"] = copy.deepcopy(data["Original"]["X"][:, f_select_ids])
        else:
            data["Processed"] = copy.deepcopy(data["Original"])
        return data

    def execute_feature_and_data_predict_fusion(self, data, layer):

        if self.feature_fusions.executable(layer):
            original_X = data["Processed"]["X"]
            finfos = data.get("Finfos")
            fusion_X = self.feature_fusions.predict_excecute(original_X, finfos, layer)
            data["Processed"]["X"] = fusion_X

        return data

    def split_predict_data_to_local(self, data, layer):

        if self.feature_split_processor is None:
            return data

        if self.feature_split_processor.executable(layer):

            X = data["Processed"]["X"]
            # 进行特征融合
            X_split, split_finfo= self.feature_split_processor.predict_execute(X, layer)
            # 将融合的数据封装回去
            if split_finfo.get("isSuccess"):
                # 将融合的数据封装回去
                data["SplitFeature"] = dict(Xs=X_split)

        return data

    def execute_predict_fusion_features_processors(self, data, layer):

        if self.fusion_features_processors is None and len(self.fusion_features_processors) == 0:
            return data

        # 获得原始数据 (经过特征筛选的数据)
        processed_data = data.get("Processed")
        Xs= processed_data.get("Xs")

        for fusion_features_processor in self.fusion_features_processors:
            if fusion_features_processor.executable(layer):
                Xs = fusion_features_processor.excecute(Xs, layer)

        # 将融合的数据封装回去
        data["Processed"]["Xs"] = Xs

        return data

    def execute_predict_split_features_processors(self, data, layer):

        if self.split_feature_processors is None or len(self.split_feature_processors) == 0:
            return data

        processed_data = data.get("SplitFeature", None)
        if processed_data is None:
            return data

        # 获得原始数据 (经过特征筛选的数据)
        Xs = processed_data.get("Xs")
        for split_feature_processor in self.split_feature_processors:
            if split_feature_processor.executable(layer):
                Xs = split_feature_processor.excecute(Xs, layer)

        data["SplitFeature"]["Xs"] = Xs

        return data

    def obtain_cascade_predict_classifier_instance(self, layer):
        return self.classifier_instances[layer]

    def obtain_relevant_to_predict_data(self, data, classifier_instances, layer):

        all_finfos = dict()

        global_instances_num = classifier_instances["GlobalNum"]
        # 没有特征实例, 保存空字典就行
        if global_instances_num != 0:
            global_finfos = dict()

            X = data["Processed"]["X"]
            global_classifier_instances = classifier_instances["Global"]

            for classifier_name, classifier_instance in global_classifier_instances.items():
                finfos = self.obtain_current_layer_predict_features(X, classifier_instance, layer)
                global_finfos[classifier_name] = finfos

            all_finfos.update(global_finfos)

        local_instances_num = classifier_instances.get("LocalNum")
        # 没有特征实例, 保存空字典就行
        if local_instances_num != 0:
            local_finfos = dict()

            Xs = data["SplitFeature"]["Xs"]
            local_classifier_instances = classifier_instances["Local"]

            for classifier_name, classifier_instances in local_classifier_instances.items():
                for index, classifier_instance in classifier_instances.items():
                    finfos = self.obtain_current_layer_predict_features(Xs[index], classifier_instance, layer)
                    classifier_name_id = classifier_name + "&" + str(index)
                    local_finfos[classifier_name_id] = finfos

            all_finfos.update(local_finfos)



        return all_finfos

    def execute_predict_feature_processors(self, features, layer):

        for feature_processor in self.feature_processors:
            if feature_processor.executable(layer):
                features = feature_processor.predict_excecute(features, layer)

        return features

    def obtain_current_layer_predict_features(self, X, classifier_instance, layer):
        # 真正执行特征提取的地方
        builder_type = classifier_instance.obtain_builder_type()
        est_type = classifier_instance.obtain_est_type()
        data_type = classifier_instance.obtain_data_type()

        features_X = classifier_instance.obtain_features(X)
        predicts = classifier_instance.predict(X)
        probs = classifier_instance.predict_probs(X)

        finfos = {
            "EstType": est_type, "BuilderType": builder_type, "DataType": data_type, "Index": -1,  "Layer": layer,
            "Feature_X": features_X, "Predict": predicts, "Probs": probs
        }

        return finfos

    def post_predict_cascade_data_and_infos(self, data, layer):
        return data

    def execute_after_predict_probs(self, data):
        pass

    def obtain_feature_selectors(self):
        return self.feature_selectors

    def set_feature_selectors(self, new_feature_selectors):
        self.feature_selectors = new_feature_selectors

    def obtain_feature_fusions(self):
        return self.feature_fusions

    def set_feature_fusions(self, new_feature_fusions):
        self.feature_fusions = new_feature_fusions

    def obtain_fusion_features_processors(self):
        return self.fusion_features_processors

    def set_fusion_features_processors(self, new_fusion_features_processors):
        self.fusion_features_processors = new_fusion_features_processors

    def obtain_category_imbalance_processor(self):
        return self.category_imbalance_processor

    def set_category_imbalance_processor(self, new_category_imbalance_processor):
        self.category_imbalance_processor = new_category_imbalance_processor

    def obtain_global_classifier_builders(self):
        return self.global_classifier_builders

    def set_global_classifier_builders(self, new_global_classifier_builders):
        self.global_classifier_builders = new_global_classifier_builders

    def obtain_feature_processors(self):
        return self.feature_processors

    def set_feature_processors(self, new_feature_processors):
        self.feature_processors = new_feature_processors

    def obtain_metric_processor(self):
        return self.metrics_processor

    def set_metric_processor(self, new_feature_processors):
        self.metrics_processor = new_feature_processors

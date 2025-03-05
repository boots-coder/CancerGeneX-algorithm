import copy
import time
import warnings
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from Processor.PreProcessor.clustering.FeatureClusterExecutor import FeatureClusterExecutor
from Processor.ProcessorDispatcher.Dispatcher import (
    FeatFusionDispatcher, FeatSelectorDispatcher, FeaturesProcessorDispatcher,
    MetricProcessorDispatcher, ClassifierDispatcher, CategoryImbalanceDispatcher,
    PreProcessorDispatcher, PostProcessorDispatcher, FusionFeatDispatcher,
    FeatureSplitDispatcher
)

warnings.filterwarnings("ignore")

class UnimodalModel():
    """
    单模态级联模型，支持以下流程：
    1. 数据预处理与初始化
    2. 执行聚类筛选出优秀特征
    3. 多层级联迭代（特征选择、特征融合、类别不平衡处理、分类器训练、指标计算）
    4. 预测流程（根据最优层数预测）
    """

    def __init__(self, config):
        assert config is not None, "单模态级联模型的配置信息不能为空"

        # 基本参数初始化
        self.config = config
        self.max_num_iterations = config.get("MaxNumIterations", 20)
        self.termination_layer = config.get("TerminationLayer", 3)
        self.class_num = config.get("ClassNum", None)
        self.debug = config.get("Debug", True)
        self.feature_types = config.get("DataType", set("Global"))

        # 模型构件初始化
        self.classifier_instances = dict()
        self.f_select_ids = {}
        self._init_components(config)

        # 历史记录，用于在预测阶段回溯构件配置
        self.all_feature_split_processors = dict()
        self.all_feature_fusions_processors = dict()
        self.all_fusion_feature_processors = dict()
        self.all_feature_processors = dict()
        self.all_metrics_processor = dict()
        self.all_feature_types = dict()

    # ------------------------- 初始化相关方法 -------------------------
    def _init_components(self, config):
        self._init_pre_processor(config)
        self._init_feature_selectors(config)
        self._init_data_and_feature_fusion(config)
        self._init_fusion_feature_processors(config)
        self._init_feature_split_processor(config)
        self._init_category_imbalance_processors(config)
        self._init_cascade_classifier_builder(config)
        self._init_cascade_features_processors(config)
        self._init_cascade_metrics_processor(config)
        self._init_post_processor(config)

    def _init_pre_processor(self, configs):
        processor_configs = configs.get("PreProcessors", None)
        if processor_configs is not None:
            pre_processor_dispatcher = PreProcessorDispatcher()
            self.pre_processors = pre_processor_dispatcher.obtain_instance(processor_configs)
        else:
            self.pre_processors = []

    def _init_feature_selectors(self, configs):
        feature_selector_configs = configs.get("FeatureSelector", None)
        if feature_selector_configs is not None:
            feat_selector_dispatcher = FeatSelectorDispatcher()
            self.feature_selectors = feat_selector_dispatcher.obtain_instance(feature_selector_configs)
        else:
            self.feature_selectors = []

    def _init_data_and_feature_fusion(self, configs):
        fusion_configs = configs.get("FeatureFusion", None)
        if fusion_configs is not None:
            feat_fusion_dispatcher = FeatFusionDispatcher()
            self.feature_fusions = feat_fusion_dispatcher.obtain_instance(fusion_configs)
        else:
            raise "必须配置特征融合方法"

    def _init_fusion_feature_processors(self, configs):
        fusion_features_processors_configs = configs.get("FusionFeatureProcessors", None)
        if fusion_features_processors_configs is not None:
            fusion_feat_dispatcher = FusionFeatDispatcher()
            self.fusion_features_processors = fusion_feat_dispatcher.obtain_instance(
                fusion_features_processors_configs)
        else:
            self.fusion_features_processors = []

    def _init_feature_split_processor(self, configs):
        feature_split_processor_config = configs.get("FeatureSplitProcessor", None)
        if feature_split_processor_config is not None:
            feature_split_dispatcher = FeatureSplitDispatcher()
            self.feature_split_processor = feature_split_dispatcher.obtain_instance(feature_split_processor_config)
        else:
            self.feature_split_processor = None

    def _init_category_imbalance_processors(self, config):
        category_imbalance_config = config.get("CategoryImbalance", None)
        if category_imbalance_config is not None:
            category_imbalance_dispatcher = CategoryImbalanceDispatcher()
            self.category_imbalance_processor = category_imbalance_dispatcher.obtain_instance(
                category_imbalance_config)
        else:
            self.category_imbalance_processor = None

    def _init_cascade_classifier_builder(self, configs):
        builder_configs = configs.get("CascadeClassifier", None)
        if builder_configs is not None:
            builder_dispatcher = ClassifierDispatcher()
            global_classifier_builders, local_classifier_builders = builder_dispatcher.obtain_instance(builder_configs)
            self.global_classifier_builders = global_classifier_builders
            self.local_classifier_builders = local_classifier_builders
        else:
            raise "分类器不能为空"

    def _init_cascade_features_processors(self, config):
        feature_processors_configs = config.get("FeatureProcessors", None)
        if feature_processors_configs is not None:
            features_processor_dispatcher = FeaturesProcessorDispatcher()
            self.feature_processors = features_processor_dispatcher.obtain_instance(feature_processors_configs)
        else:
            self.feature_processors = []

    def _init_cascade_metrics_processor(self, config):
        """
        默认使用“AccuracyMetricProcessor”以准确率作为主指标。
        如果 config 中已经显式配置了 MetricsProcessors，就按用户配置执行。
        """
        metrics_processor_configs = config.get("MetricsProcessors", None)
        if metrics_processor_configs is not None:
            metric_processor_dispatcher = MetricProcessorDispatcher()
            self.metrics_processor = metric_processor_dispatcher.obtain_instance(metrics_processor_configs)
        else:
            # 使用默认的 Accuracy 计算器
            from sklearn.metrics import accuracy_score

            class AccuracyMetricProcessor:
                def obtain_name(self):
                    return "AccuracyMetricProcessor"

                def fit_excecute(self, data, layer):
                    """
                    从 data['Finfos'][layer] 中获取各分类器的预测结果(Predict)，
                    做简单投票或平均，再计算准确率
                    """
                    y_true = data['Original']['y_val']
                    latest_layer = max(data['Finfos'].keys())
                    layer_finfos = data['Finfos'][latest_layer]

                    all_preds = []
                    # 遍历每个分类器的 finfo，取其“Predict”字段
                    for info in layer_finfos.values():
                        if info['Predict'] is not None:
                            all_preds.append(info['Predict'])

                    if not all_preds:
                        return 0.0

                    # 若有多个分类器，可以做简单投票:
                    # 先将它们堆叠成 (num_classifiers, num_samples)
                    all_preds = np.array(all_preds)

                    # 对多分类情况，用多数投票
                    # 对二分类情况 (含 0/1)，也是多数投票
                    # 以列为样本维度，所以要对axis=0维度做统计
                    final_pred = []
                    for col in all_preds.T:  # 遍历每个样本在所有分类器下的预测
                        vals, counts = np.unique(col, return_counts=True)
                        # 选出现次数最多的类别
                        majority_class = vals[np.argmax(counts)]
                        final_pred.append(majority_class)
                    final_pred = np.array(final_pred)

                    return accuracy_score(y_true, final_pred)

            self.metrics_processor = AccuracyMetricProcessor()

    def _init_post_processor(self, config):
        post_processors_configs = config.get("PostProcessors", None)
        if post_processors_configs is not None:
            post_processor_dispatcher = PostProcessorDispatcher()
            self.post_processors = post_processor_dispatcher.obtain_instance(post_processors_configs)
        else:
            self.post_processors = []

    # ------------------------- 训练相关主流程方法 -------------------------
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if y_val is None:
            division_ratio = self.config.get("DivisionRatio", 0.8)
            if self.debug:
                print("没有设置验证集， 随机从训练集中进行划分, 划分比例为:", division_ratio)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=division_ratio)
        self._fit(X_train, y_train, X_val, y_val)

    def _fit(self, X_train, y_train, X_val, y_val):
        """
        同时计算并打印：Accuracy(主指标)、AUC(参考指标)。
        早停逻辑使用 Accuracy。
        """
        start_time = time.time()

        # 执行循环前的一些操作（预处理和数据封装）
        data = self.execute_before_fit(X_train, y_train, X_val, y_val)
        data = self.execute_pre_fit_processor(data)

        # 如果想使用聚类选取特征，可启用这行
        # data = self.execute_cluster(data)

        # 开始级联迭代
        best_level, best_metric = 1, 0.0
        best_metric_processors = self.metrics_processor
        count = 0

        for layer in range(1, self.max_num_iterations + 1):
            # 准备第layer层的数据与信息
            data = self.pre_fit_cascade_data_and_infos(data, layer)

            # 特征选择 //todo 改进为投票算法

            fselect_ids, fselect_infos = self.execute_feature_selector_processors(data, layer)
            data = self.execute_fit_feature_selection(data, fselect_ids)
            self.save_f_select_ids(fselect_ids, layer)

            # # 特征融合
            # data = self.execute_feature_and_data_fit_fusion(data, layer)
            #
            # # 特征划分（局部特征）
            # data = self.split_fit_data_to_local(data, layer)
            #
            # # 融合特征处理
            # data = self.execute_fit_fusion_features_processors(data, layer)

            # 类别不平衡处理
            data = self.execute_category_imbalance(data, layer)

            # 训练分类器
            builder_configs = self.obtain_new_update_builder_configs(data, layer)
            classifier_instances = self.execute_cascade_fit_classifier(data, builder_configs, layer)

            # 获取分类器提取的特征与概率
            all_finfos = self.obtain_relevant_fit_to_data(data, classifier_instances, layer)

            # 进行特征处理器处理
            all_finfos = self.execute_fit_feature_processors(all_finfos, layer)

            # 保存信息 -- 保存已经提取的特征
            self.save_relevant_fit_to_data(all_finfos, data, layer)

            # 调整分类器（可选）
            classifier_instances, data = self.adjust_cascade_classifier(classifier_instances, data)
            self.save_cascade_classifier(classifier_instances, layer)

            # 计算当前层的 Accuracy（主指标，用于 early-stop）
            metric = self.obtain_current_metric(data, layer)
            # 计算当前层的 AUC（仅作打印参考）
            auc_value = self.calculate_auc(data)

            # 打印
            print(f"第 {layer} 层 -> Accuracy: {metric:.4f}, AUC: {auc_value:.4f}")

            # 比较是否是最好层
            if metric > best_metric:
                count = 0
                best_level, best_metric = layer, metric
                best_metric_processors = self.metrics_processor
            else:
                count += 1

            # 停止条件
            if count >= self.termination_layer or layer == self.max_num_iterations:
                print(f"最优层 = {best_level}, 最优 Accuracy = {best_metric:.4f}")
                self.best_level = best_level
                self.best_metric_processors = best_metric_processors
                break

            # 后置处理器
            data = self.execute_post_fit_processor(data, layer)
            data = self.post_fit_cascade_data_and_infos(data, layer)

        self.execute_after_fit(data)
        end_time = time.time()
        print("训练总耗时:", end_time - start_time, "秒")

    def calculate_auc(self, data):
        """计算验证集的AUC值（仅作为辅助参考，不用于早停）"""
        try:
            y_true = data['Original']['y_val']
            latest_layer = max(data['Finfos'].keys())
            layer_finfos = data['Finfos'][latest_layer]

            all_probs = []
            for classifier_info in layer_finfos.values():
                if classifier_info['Probs'] is not None:
                    all_probs.append(classifier_info['Probs'])

            if not all_probs:
                return 0.0

            y_pred = np.mean(all_probs, axis=0)
            if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                # 对二分类而言，取正类概率
                y_pred = y_pred[:, 1]
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(y_true, y_pred)

        except Exception as e:
            print(f"计算AUC时出错: {str(e)}")
            return 0.0

    # ------------------------- 训练流程子方法 -------------------------
    def execute_before_fit(self, X_train, y_train, X_val, y_val):
        if self.debug:
            print("==================执行循环前的预处理开始==================")

        data = dict()
        size, dim = X_train.shape
        data["Original"] = dict(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, Size=size, Dim=dim)
        data["Finfos"] = dict()  # 用于存储特征信息

        if self.debug:
            print("训练集大小:", size, ", 维度:", dim)

        return data

    def execute_pre_fit_processor(self, data):
        if self.debug:
            pre_processor_names = []
        for pre_processor in self.pre_processors:
            if pre_processor.executable():
                if self.debug:
                    pre_processor_names.append(pre_processor.obtain_name())
                data = pre_processor.fit_excecute(data)

        if self.debug:
            print("循环前的预处理器数量:", len(pre_processor_names))
            if len(pre_processor_names) > 0:
                print("循环前的预处理器的名字:", pre_processor_names)
            print("==================执行循环前的预处理结束==================")

        return data

    def execute_cluster(self, data):
        # 可选：执行特征聚类，获得代表性特征
        cluster_executor = FeatureClusterExecutor(cluster_threshold=50, cv=5, scorer='accuracy', max_iter=1000)
        data = cluster_executor(data)
        return data

    def pre_fit_cascade_data_and_infos(self, data, layer):
        print("==================第" + str(layer) + "层开始执行==================")
        self.change_feature_selectors(data, layer)
        self.change_feature_fusions(data, layer)
        self.change_fusion_features_processors(data, layer)
        self.change_category_imbalance_processor(data, layer)
        self.change_global_classifier_builders(data, layer)
        self.change_feature_processors(data, layer)
        self.change_metric_processor(data, layer)
        self.change_feature_types(data, layer)
        return data

    def execute_feature_selector_processors(self, data, layer):
        if self.debug:
            print("==================特征筛选算法开始执行==================")
            feature_selector_names = []

        original_data = data.get("Original")
        X_train, y_train = original_data["X_train"], original_data["y_train"]
        dim = original_data["Dim"]
        f_select_infos = dict(X_train=X_train, y_train=y_train, Dim=dim)
        f_select_idxs = None

        for feature_selector in self.feature_selectors:
            if feature_selector.fit_executable(layer):
                if self.debug:
                    feature_selector_names.append(feature_selector.obtain_name())
                f_select_idxs, f_select_infos = feature_selector.fit_excecute(f_select_idxs, f_select_infos, layer)

        if self.debug:
            print("使用的特征筛选器数量为:", len(feature_selector_names))
            if len(feature_selector_names) > 0:
                print("使用的特征筛选器的名字分别是", feature_selector_names)
                if f_select_infos.get("NewFeatures"):
                    print(feature_selector_names[0] + "重新进行了特征筛选")
                else:
                    print(feature_selector_names[0] + "使用之前筛选好的特征")
                recall_num = f_select_infos.get("RecallNum", None)
                if recall_num is not None:
                    print("属性召回模块召回的数量是:" + str(recall_num))
                if f_select_idxs is not None:
                    print("最终获得的筛选特征数量是: " + str(len(f_select_idxs)))

        return f_select_idxs, f_select_infos

    def execute_fit_feature_selection(self, data, f_select_idxs):
        if f_select_idxs is not None:
            data["Processed"] = dict(
                X_train=copy.deepcopy(data["Original"]["X_train"][:, f_select_idxs]),
                y_train=copy.deepcopy(data["Original"]["y_train"]),
                X_val=copy.deepcopy(data["Original"]["X_val"][:, f_select_idxs]),
                y_val=copy.deepcopy(data["Original"]["y_val"])
            )
        else:
            data["Processed"] = copy.deepcopy(data["Original"])

        if self.debug:
            print("==================数据池清洗工作完成==================")

        return data

    def save_f_select_ids(self, f_select_ids, layer):
        self.f_select_ids[layer] = f_select_ids

    def execute_feature_and_data_fit_fusion(self, data, layer):
        if self.feature_fusions.executable(layer):
            if self.debug:
                print("==================特征融合开始执行==================")
                print("特征融合方法: ", self.feature_fusions.obtain_name())

            original_data = data.get("Processed")
            original_train = original_data.get("X_train")
            original_val = original_data.get("X_val")
            finfos = data.get("Finfos")

            fusion_train, fusion_val = self.feature_fusions.fit_excecute(
                data, original_train, original_val, finfos, layer)
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

            processed_data = data.get("Processed")
            X_train, X_val = processed_data.get("X_train"), processed_data.get("X_val")
            y_train, y_val = processed_data.get("y_train"), processed_data.get("y_val")

            X_split_train, X_split_val, split_finfo = self.feature_split_processor.fit_excecute(
                X_train, X_val, layer)
            if split_finfo.get("isSuccess"):
                data["SplitFeature"] = dict(Xs_train=X_split_train, Xs_val=X_split_val,
                                            y_train=y_train, y_val=y_val)
            else:
                print("划分失败, 失败信息" + split_finfo.get("FailureInfo"))

            if self.debug:
                print("==================特征划分执行完成==================")

        return data

    def execute_fit_fusion_features_processors(self, data, layer):
        if self.debug:
            print("==================全局融合特征执行开始==================")
            fusion_features_processor_names = []

        processed_data = data.get("Processed")
        X_train, X_val = processed_data.get("X_train"), processed_data.get("X_val")

        for fusion_features_processor in self.fusion_features_processors:
            if fusion_features_processor.executable(layer):
                if self.debug:
                    fusion_features_processor_names.append(fusion_features_processor.obtain_name())
                X_train = fusion_features_processor.excecute(X_train, layer)
                X_val = fusion_features_processor.excecute(X_val, layer)

        data["Processed"]["X_train"] = X_train
        data["Processed"]["X_val"] = X_val

        if self.debug:
            print("使用的融合特征处理器的数量为:", len(fusion_features_processor_names))
            if len(fusion_features_processor_names) > 0:
                print("使用的特征融合器的名字分别是", fusion_features_processor_names)
            print("==================全局融合特征执行完成==================")

        return data

    def execute_category_imbalance(self, data, layer):
        if self.category_imbalance_processor is None:
            return data

        if self.debug:
            print("==================类别不平衡器执行开始==================")
            print(self.category_imbalance_processor.obtain_name())

        if self.category_imbalance_processor.fit_executable(layer):
            processed_data = data.get("Processed", None)
            X_train, y_train = processed_data["X_train"], processed_data["y_train"]
            X_train_res, y_train_res = self.category_imbalance_processor.fit_excecute(X_train, y_train, layer)
            data["Processed"]["X_train_res"] = X_train_res
            data["Processed"]["y_train_res"] = y_train_res

        if self.debug:
            print("==================类别不平衡器执行完成==================")

        return data

    def execute_cascade_fit_classifier(self, data, builder_configs, layer):
        if self.debug:
            print("==================分类器执行开始==================")

        # 全局分类器训练
        if "Global" in self.feature_types:
            global_data = data.get("Processed")
            X_train_res = global_data.get("X_train_res", global_data["X_train"])
            y_train_res = global_data.get("y_train_res", global_data["y_train"])
            X_val, y_val = global_data["X_val"], global_data["y_val"]

            global_builder_configs = builder_configs.get("Global", None)
            global_classifier_instances = self.obtain_fit_classifier_instance(
                X_train_res, y_train_res, X_val, y_val,
                global_builder_configs, self.global_classifier_builders, layer)
            global_instances_num = len(global_classifier_instances)
        else:
            global_classifier_instances = None
            global_instances_num = 0

        # 局部分类器训练
        if "Local" in self.feature_types:
            local_data = data.get("SplitFeature", None)
            X_train_res = local_data.get("Xs_train_res", local_data["Xs_train"])
            y_train_res = local_data.get("y_train_res", local_data["y_train"])
            X_val, y_val = local_data["Xs_val"], local_data["y_val"]

            local_builder_configs = builder_configs.get("Local", None)
            local_classifier_instances = self.obtain_fit_classifier_instance(
                X_train_res, y_train_res, X_val, y_val,
                local_builder_configs, self.local_classifier_builders, layer)
            local_instances_num = len(local_classifier_instances)
        else:
            local_classifier_instances = None
            local_instances_num = 0

        # 总数判断
        totall_num = global_instances_num + local_instances_num
        if totall_num == 0:
            raise "当前层最终获得的分类器数量为 0, 请重新检查配置信息"

        classifier_instances = {
            "Global": global_classifier_instances,
            "Local": local_classifier_instances,
            "GlobalNum": global_instances_num,
            "LocalNum": local_instances_num,
            "TotallNum": totall_num
        }

        if self.debug:
            print(f"训练好的分类器(或特征提取器)总共有{totall_num}个, 其中", end=",")
            print(f"全局分类器(或特征提取器)有{global_instances_num}个", end=",")
            print(f"局部分类器(或特征提取器)有{local_instances_num}个")
            print("==================分类器执行完成==================")

        return classifier_instances

    def obtain_fit_classifier_instance(self, X_train, y_train, X_val, y_val,
                                       builder_configs, classifier_builders, layer):
        classifier_instances = {}
        progress_bar = tqdm(classifier_builders, desc="Building classifiers")

        for classifier_builder in progress_bar:
            if classifier_builder.fit_executable(layer):
                builder_name = classifier_builder.obtain_name()
                if self.debug:
                    print("==================" + builder_name + " 开始构建分类器 ==================")

                builder_config = self.obtain_builder_config(builder_configs, builder_name)
                if builder_config is not None:
                    classifier_builder.update_config(builder_config, layer)

                classifier_name, classifier_instance = classifier_builder.obtain_fit_classifier(
                    X_train, y_train, X_val, y_val, layer)
                classifier_instances[classifier_name] = classifier_instance

                if self.debug:
                    print("==================" + builder_name + "分类器构建完成 ==================")
                progress_bar.set_postfix({"Last Built": builder_name})

        return classifier_instances

    def obtain_builder_config(self, builder_configs, builder_name):
        if builder_configs is None:
            return None
        return builder_configs.get(builder_name, None)

    def obtain_relevant_fit_to_data(self, data, classifier_instances, layer):
        if self.debug:
            print("==================开始获取相关信息(特征、预测值等)==================")

        all_finfos = dict()

        # 全局特征提取
        global_instances_num = classifier_instances["GlobalNum"]
        if global_instances_num > 0:
            global_finfos = dict()
            global_data = data.get("Processed")
            X_train, y_train = global_data["X_train"], global_data["y_train"]
            X_val, y_val = global_data["X_val"], global_data["y_val"]
            global_cls = classifier_instances.get("Global")

            for cls_name, cls_instance in global_cls.items():
                finfos = self.obtain_current_layer_fit_features(X_train, y_train, X_val, y_val,
                                                                cls_instance, layer)
                global_finfos[cls_name] = finfos

            all_finfos.update(global_finfos)

            if self.debug and len(global_finfos) > 0:
                print("全局特征提取器执行结束, 最终获得的特征数量为:", len(global_finfos))
                print("特征提取器的名字有: ", list(global_finfos.keys()))
                print("每个特征的属性有", list(list(global_finfos.values())[0].keys()))
        else:
            if self.debug:
                print("没有设置全局分类器或特征提取器")

        # 局部特征提取
        local_instances_num = classifier_instances.get("LocalNum")
        if local_instances_num > 0:
            local_finfos = dict()
            split_feature = data.get("SplitFeature", None)
            Xs_train, y_train = split_feature["Xs_train"], split_feature["y_train"]
            Xs_val, y_val = split_feature["Xs_val"], split_feature["y_val"]
            local_cls = classifier_instances.get("Local")

            for cls_name, cls_instance_group in local_cls.items():
                for index, cls_instance in cls_instance_group.items():
                    finfos = self.obtain_current_layer_fit_features(
                        Xs_train[index], y_train, Xs_val[index], y_val, cls_instance, layer)
                    classifier_name_id = cls_name + "&" + str(index)
                    local_finfos[classifier_name_id] = finfos

            all_finfos.update(local_finfos)

            if self.debug and len(local_finfos) > 0:
                print("局部特征提取完成, 获得特征数量:", len(local_finfos))
                print("特征提取器的名字有: ", list(local_finfos.keys()))
                print("每个特征的属性有", list(list(local_finfos.values())[0].keys()))
        else:
            if self.debug:
                print("没有设置局部分类器或特征提取器")

        if self.debug:
            print("==================获取相关信息结束==================")

        return all_finfos

    def obtain_current_layer_fit_features(self, X_train, y_train, X_val, y_val,
                                          classifier_instance, layer):
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

        finfos = dict(
            ClassifierName=cls_name,
            EstType=est_type,
            BuilderType=builder_type,
            DataType=data_type,
            Layer=layer,
            Feature_train=features_train,
            Feature_val=features_val,
            Predict=predict,
            Probs=probs
        )

        if self.debug and probs is not None:
            print("分类器:" + cls_name + ", 数据类型:" + data_type + "的性能指标:")
            print(metrics.classification_report(y_val, predict, digits=4))

        return finfos

    def adjust_cascade_classifier(self, classifier_instances, data):
        # 可在此对分类器进行筛选和调整
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
            print("特征处理器数量:", len(feature_processors_names))
            if len(feature_processors_names) > 0:
                print("特征处理器的名字:", feature_processors_names)
            print("==============特征处理器执行完成===============")

        return features

    def obtain_current_metric(self, data, layer):
        """
        这里返回 metrics_processor.fit_excecute(data, layer)，
        该 processor 默认是 AccuracyMetricProcessor。
        """
        if self.metrics_processor is not None:
            if self.debug:
                print("==============开始执行指标计算器===============")
                print("指标计算器的名字:", self.metrics_processor.obtain_name())

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
            print("级联后置处理器数量:", len(post_processors_names))
            if len(post_processors_names) > 0:
                print("级联后置处理器的名字:", post_processors_names)
            print("==============级联后置处理器执行完成===============")

        return data

    def post_fit_cascade_data_and_infos(self, data, layer):
        print("==============第" + str(layer) + "层执行结束===============")
        # 保存各处理器和特征类型信息，预测时会用到
        self.all_feature_split_processors[layer] = copy.deepcopy(self.feature_split_processor)
        self.all_feature_fusions_processors[layer] = copy.deepcopy(self.feature_fusions)
        self.all_fusion_feature_processors[layer] = copy.deepcopy(self.fusion_features_processors)
        self.all_feature_processors[layer] = copy.deepcopy(self.feature_processors)
        self.all_metrics_processor[layer] = copy.deepcopy(self.metrics_processor)
        self.all_feature_types[layer] = copy.deepcopy(self.feature_types)
        return data

    def execute_after_fit(self, data):
        # 模型训练后的一些收尾操作（可根据需要实现）
        pass

    # ------------------------- 预测相关方法 -------------------------
    def predict(self, X):
        """
        直接返回 argmax(prob) ，若是多分类可这样做，
        若是二分类 (0/1) 则相当于 (prob >= 0.5)
        """
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        start_time = time.time()

        data = self.execute_before_predict_probs(X)
        data = self.execute_pre_predict_processor(data)

        for layer in range(1, self.best_level + 1):
            data = self.pre_predict_cascade_data_and_infos(data, layer)
            f_select_ids = self.obtain_cascade_f_select_ids(layer)
            data = self.execute_predict_feature_selection(data, f_select_ids)
            data = self.execute_feature_and_data_predict_fusion(data, layer)
            data = self.execute_predict_fusion_features_processors(data, layer)

            classifier_instances = self.obtain_cascade_predict_classifier_instance(layer)
            all_finfos = self.obtain_relevant_to_predict_data(data, classifier_instances, layer)
            all_finfos = self.execute_predict_feature_processors(all_finfos, layer)
            self.save_relevant_to_predict_data(all_finfos, data, layer)

            # 使用与训练时同一个 metrics_processor 进行 predict_execute
            # 这里很多 Processor 可能未实现 predict_execute，可按自己需求扩展
            if layer == self.best_level:
                # 你可以在这里写自定义逻辑将 multiple classifier 的 prob 进行融合
                # 简单做个平均
                final_probs = self._average_probs(data, layer)
                probs = final_probs
                break

            data = self.post_predict_cascade_data_and_infos(data, layer)

        self.execute_after_predict_probs(data)
        end_time = time.time()
        print("预测耗时:", end_time - start_time, "秒")
        return probs

    def _average_probs(self, data, layer):
        """
        简单将本层所有分类器的 Prob 求平均，返回 fused_probs
        """
        layer_finfos = data['Finfos'][layer]
        all_probs = []
        for info in layer_finfos.values():
            if info['Probs'] is not None:
                all_probs.append(info['Probs'])
        if not all_probs:
            # 若没有任何概率，返回全0.5或其它默认值
            return np.full((data["Original"]["X"].shape[0], 2), 0.5)
        return np.mean(all_probs, axis=0)

    # ------------------------- 预测流程子方法 -------------------------
    def execute_before_predict_probs(self, X):
        data = dict()
        size, dim = X.shape
        data["Original"] = dict(X=X, Size=size, Dim=dim)
        data["Finfos"] = dict()
        if self.debug:
            print("测试集大小:", size, ", 维度:", dim)
        return data

    def execute_pre_predict_processor(self, data):
        for pre_processor in self.pre_processors:
            if pre_processor.executable():
                data = pre_processor.predict_excecute(data)
        return data

    def pre_predict_cascade_data_and_infos(self, data, layer):
        # 恢复训练时对应layer的组件配置
        self.feature_split_processor = self.all_feature_split_processors[layer]
        self.feature_fusions = self.all_feature_fusions_processors[layer]
        self.fusion_features_processors = self.all_fusion_feature_processors[layer]
        self.feature_processors = self.all_feature_processors[layer]
        self.metrics_processor = self.all_metrics_processor[layer]
        return data

    def obtain_cascade_f_select_ids(self, layer):
        return self.f_select_ids[layer]

    def execute_predict_feature_selection(self, data, f_select_ids):
        if f_select_ids is not None:
            data["Processed"] = dict(X=copy.deepcopy(data["Original"]["X"][:, f_select_ids]))
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

    def execute_predict_fusion_features_processors(self, data, layer):
        if self.fusion_features_processors is None or len(self.fusion_features_processors) == 0:
            return data
        processed_data = data.get("Processed")
        Xs = processed_data.get("X")

        for fusion_features_processor in self.fusion_features_processors:
            if fusion_features_processor.executable(layer):
                Xs = fusion_features_processor.excecute(Xs, layer)

        data["Processed"]["X"] = Xs
        return data

    def obtain_cascade_predict_classifier_instance(self, layer):
        return self.classifier_instances[layer]

    def obtain_relevant_to_predict_data(self, data, classifier_instances, layer):
        all_finfos = dict()

        # 全局
        global_instances_num = classifier_instances["GlobalNum"]
        if global_instances_num != 0:
            global_finfos = dict()
            X = data["Processed"]["X"]
            global_cls = classifier_instances["Global"]
            for cls_name, cls_instance in global_cls.items():
                finfos = self.obtain_current_layer_predict_features(X, cls_instance, layer)
                global_finfos[cls_name] = finfos
            all_finfos.update(global_finfos)

        # 局部
        local_instances_num = classifier_instances.get("LocalNum")
        if local_instances_num != 0:
            local_finfos = dict()
            Xs = data["SplitFeature"]["Xs"]
            local_cls = classifier_instances["Local"]
            for cls_name, cls_instance_group in local_cls.items():
                for index, cls_instance in cls_instance_group.items():
                    finfos = self.obtain_current_layer_predict_features(Xs[index], cls_instance, layer)
                    classifier_name_id = cls_name + "&" + str(index)
                    local_finfos[classifier_name_id] = finfos
            all_finfos.update(local_finfos)

        return all_finfos

    def execute_predict_feature_processors(self, features, layer):
        for feature_processor in self.feature_processors:
            if feature_processor.executable(layer):
                features = feature_processor.predict_excecute(features, layer)
        return features

    def obtain_current_layer_predict_features(self, X, classifier_instance, layer):
        builder_type = classifier_instance.obtain_builder_type()
        est_type = classifier_instance.obtain_est_type()
        data_type = classifier_instance.obtain_data_type()

        features_X = classifier_instance.obtain_features(X)
        predicts = classifier_instance.predict(X)
        probs = classifier_instance.predict_probs(X)

        finfos = {
            "EstType": est_type,
            "BuilderType": builder_type,
            "DataType": data_type,
            "Index": -1,
            "Layer": layer,
            "Feature_X": features_X,
            "Predict": predicts,
            "Probs": probs
        }
        return finfos

    def post_predict_cascade_data_and_infos(self, data, layer):
        return data

    def execute_after_predict_probs(self, data):
        pass

    # ------------------------- 工具和改变方法(空实现) -------------------------
    def change_feature_selectors(self, data, layer):
        pass

    def change_feature_fusions(self, data, layer):
        pass

    def change_fusion_features_processors(self, data, layer):
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

    def save_relevant_fit_to_data(self, all_finfos, data, layer):
        data["Finfos"][layer] = all_finfos

    # ------------------------- Getter/Setter方法 -------------------------
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

    def save_relevant_to_predict_data(self, all_finfos, data, layer):
        data["Finfos"][layer] = all_finfos
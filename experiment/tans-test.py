#!/usr/bin/env python
# coding=utf-8
from sklearn.model_selection import train_test_split

from Model import UnimodalModel

if __name__ == '__main__':
    if __name__ == '__main__':
        import pandas as pd
        import sys

        #  temp = sys.stdout ##保存原来的输出流方式
        # sys.stdout = open('saveit2.txt', 'a')

        class_num = 2
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
                    "Parameter": {0.1},
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


            # "FeatureSplitProcessor" : {
            #     "Name" : "AverageSplit",
            #     "Type" : "AverageSplit",
            #     "SplitNum" : 3,
            # },

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

                "Transformer" : {
                    "Layers" : None,
                    "Builder" : "DL",
                    "DataType": ["Global", "Local"],
                    "Trainer" : {
                        "name" : "TrainerWrapper",
                        "Parameter" : {}
                    },
                    "Model" : {
                        "name" : "Transformer",
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
        # !/usr/bin/env python
        # coding=utf-8
        from sklearn.model_selection import train_test_split
        from Model import UnimodalModel
        from sklearn.metrics import roc_auc_score
        from sklearn.decomposition import PCA
        import scipy.io
        import os
        import numpy as np


        def load_data_from_mat(file_path):
            """
            Load dataset from .mat file, extract features (X) and labels (Y).
            """
            mat_data = scipy.io.loadmat(file_path)

            # Extract features and labels
            X = mat_data['X']  # Assuming the feature matrix is stored under the key 'X'
            Y = mat_data['Y'].flatten()  # Assuming the labels are stored under the key 'Y'

            # Convert labels to 0, 1
            if -1 in Y:  # Convert -1 to 0
                Y[Y == -1] = 0
            elif 2 in Y:  # Convert 2 to 0 for {1, 2} labels
                Y[Y == 2] = 0

            return X, Y


        # 您的数据集列表
        datasets = [
            "ALLAML.mat", "colon.mat", "GLI_85.mat",
            "leukemia.mat", "Prostate_GE.mat", "SMK_CAN_187.mat"
        ]

        # 数据路径
        data_dir = "../data/"  # 根据实际情况修改

        # 重复实验次数
        n_runs = 10

        # 用字典存储每个数据集对应的10次AUC结果列表
        auc_results_runs = {ds: [] for ds in datasets}

        for run_idx in range(n_runs):
            print(f"===== Run {run_idx + 1}/{n_runs} =====")
            for dataset in datasets:
                file_path = os.path.join(data_dir, dataset)
                try:
                    # 加载数据集
                    X, Y = load_data_from_mat(file_path)

                    # 数据划分
                    x_train_val, x_test, y_train_val, y_test = train_test_split(
                        X, Y, test_size=0.3, random_state=42 + run_idx, stratify=Y
                    )

                    x_train, x_val, y_train, y_val = train_test_split(
                        x_train_val, y_train_val, test_size=(50 / 70), random_state=42 + run_idx, stratify=y_train_val
                    )

                    # 初始化并训练模型
                    model = UnimodalModel(config)  # 请确保UnimodalModel和config在您的环境中已定义
                    model.fit(x_train, y_train, x_val, y_val)

                    # 预测测试集概率
                    y_pred_proba = model.predict_proba(x_test)

                    # 确保得到的是正类的概率值
                    if len(y_pred_proba.shape) == 1 or y_pred_proba.shape[1] == 1:
                        y_pred_proba = y_pred_proba.flatten()
                    else:
                        y_pred_proba = y_pred_proba[:, 1]

                    # 计算AUC并存储
                    auc = roc_auc_score(y_test, y_pred_proba)
                    auc_results_runs[dataset].append(auc)

                    print(f"{dataset}: AUC = {auc:.4f}")

                except Exception as e:
                    # 如果出错，存入一个NaN或其他标记值
                    auc_results_runs[dataset].append(np.nan)
                    print(f"Error processing {dataset}: {e}")

        # 计算每个数据集的平均AUC
        print("\nAUC Results Summary (Average over 10 runs):")
        for dataset in datasets:
            dataset_aucs = np.array(auc_results_runs[dataset])
            avg_auc = np.nanmean(dataset_aucs)  # 如果有NaN就跳过它们
            print(f"{dataset}: Mean AUC = {avg_auc:.4f}")
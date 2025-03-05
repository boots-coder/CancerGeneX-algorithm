#!/usr/bin/env python
# coding=utf-8
from sklearn.model_selection import train_test_split
from Model import UnimodalModel
from sklearn.metrics import roc_auc_score


if __name__ == '__main__':
    import pandas as pd
    import sys
    import numpy as np
    import scipy.io

    class_num = 2
    config = {
        "ClassNum": class_num,  # 注意这个参数一定要改
        "Metric": "acc",
        "DataType": {"Global"},

        # 这里取消了标准化的Standardization -- 具体见SelectorWrapper
        "PreProcessors": {
            "MinMax": {
                "Type": "Standardization",
                "Method": "DecimalScale",
                "BuilderType": ["DL"],
                "FeaturesType": []
            },
        },

        "FeatureSelector": {
            # "GCLasso": {
            #     "Type": "FeatureSelection",
            #     "Method": "GCLasso",
            #     "Parameter": {},
            # },
            "VotingSelector": {
                "Type": "FeatureSelection",
                "Method": "VotingSelector",
                "Parameter": {
                    "max_layers": 5,
                    "min_votes_percentage": 0.3,
                    "current_layer": 1,  # 初始层级，会在每一层调用时更新
                    "selectors_config": {
                        "lasso_threshold": 0.0001,
                        "f_test_p_value": 0.05,
                        "variance_threshold": 0.01,
                        "mutual_info_threshold": 0.05,
                        "chi2_p_value": 0.05,
                        "correlation_threshold": 0.1
                    }
                },
            },

            "RecallAttribute": {
                "name": "TwoStageRecall",
                "Type": "RecallAttribute",
                "Method": "TwoStageRecall",
                "Parameter": {
                    "RandomSelectRatio": 0.3,  # 第一阶段随机选择30%的特征
                    "SelectorConfigs": {
                        "MinVotes": 2,  # 最少需要2个选择器选中
                        # 有监督方法参数
                        "MutualInfoPercentile": 10,  # 互信息
                        "VarianceThreshold": 0.1,  # 方差阈值
                        "RFImportancePercentile": 10,  # 随机森林
                        "FScorePercentile": 10,  # F-score
                        "Chi2Percentile": 10,  # 卡方检验
                        "LassoAlpha": 0.01,  # LASSO
                        "GBDTImportancePercentile": 10,  # GBDT

                        # 无监督方法参数
                        "PCAVarianceRatio": 0.9,  # PCA保留方差比例
                        "AEEncodingDim": 64,  # AutoEncoder编码维度
                        "AEEpochs": 50,  # AutoEncoder训练轮数
                        "AEBatchSize": 256,  # AutoEncoder批次大小
                        "AELearningRate": 0.001,  # AutoEncoder学习率
                        "AEReconstructionThreshold": 0.1  # AutoEncoder重构阈值
                    }
                },
            },

        },

        "CategoryImbalance": {
            "Name": "RandomOverSampler",
            "Type": "CategoryImbalance",
            "Parameter": {},
        },

        "FeatureFusion": {
            "Name": "FeatureFusion",
            "BuilderType": ["ML", "DL", "cluster"],
            "Type": "FeatureConcatenation",
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
            # "BuilderType": ["ML", "DL"],
            "BuilderType": ["ML"],
            "Type": "AvgMetricProcessor",
            "ClassifierMethod": "acc",
        },

        "CascadeClassifier": {
            "AdaptiveEnsembleClassifyByNum": {
                "AdaptiveMethod": "retained_num",
                "CaluateMetric": "acc",
                "Builder": "ML",
                "Type": "AdaptiveEnsembleClassifyByNum",
                "DataType": ["Global", "Local"],
                "BaseClassifier": {
                    "RandomForestClassifier": {
                        "Layer": [2, 3],
                        "Type": "RandomForestClassifier",
                        "Parameter": {"n_estimators": 100, "criterion": "gini",
                                      "class_weight": None, "random_state": 0},
                    },
                    "ExtraTreesClassifier": {
                        "Type": "ExtraTreesClassifier",
                        "Parameter": {"n_estimators": 100, "criterion": "gini",
                                      "class_weight": None, "random_state": 0},
                    },
                    "GaussianNBClassifier": {
                        "Type": "GaussianNBClassifier",
                        "Parameter": {}
                    },
                    "BernoulliNBClassifier": {
                        "Type": "BernoulliNBClassifier",
                        "Parameter": {}
                    },
                    "KNeighborsClassifier_1": {
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

            "Transformer": {
                "Layers": None,
                "Builder": "DL",
                "DataType": ["Global", "Local"],
                "Trainer": {
                    "name": "TrainerWrapper",
                    "Parameter": {}
                },
                "Model": {
                    "name": "Transformer",
                    "Parameter": {"ClassNum": 2}
                },
                "LossFun": {
                    "name": "CrossEntropyLoss",
                    "Parameter": {}
                },
                "Optimizer": {
                    "name": "Adam",
                    "Parameter": {"lr": 0.001},
                }
            },
            "BNN": {
                "Layers": None,
                "Builder": "DL",
                "DataType": ["Global", "Local"],
                "Trainer": {
                    "name": "TrainerWrapper",
                    "Parameter": {}
                },
                "Model": {
                    "name": "Transformer",
                    "Parameter": {"ClassNum": 2}
                },
                "LossFun": {
                    "name": "CrossEntropyLoss",
                    "Parameter": {}
                },
                "Optimizer": {
                    "name": "Adam",
                    "Parameter": {"lr": 0.001},
                }
            }
        },
    }
    def load_data(name, file_path, config):
        """
        Load dataset based on the provided name and file path.
        Adjust labels to ensure compatibility with a binary classification task.
        """
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path, sep=",")
            y = np.int64(data['Label'])
            del data['Label']
            X = data.values
        elif file_path.endswith('.mat'):
            mat_data = scipy.io.loadmat(file_path)
            X = mat_data['X']
            y = np.int64(mat_data['Y']).flatten()
        else:
            raise ValueError("Unsupported file format: {}".format(file_path))

        # Process labels
        if name in ['colon', 'leukemia']:
            y[y == -1] = 0  # Convert -1 to 0
        else:
            # For datasets with labels starting at 1
            y = y - 1

        config["ClassNum"] = 2
        return X, y, config

    # 测试仅GLI_85数据集举例
    datasets = {
        "colon": "data/colon.mat"
    }

    for name, file_path in datasets.items():
        try:
            X, Y, config = load_data(name, file_path, config)

            # Split data into train+val (70%) and test (30%)
            x_train_val, x_test, y_train_val, y_test = train_test_split(
                X, Y, test_size=0.3, random_state=42, stratify=Y
            )

            # From train+val (70%), split into train(20%) and val(50%)
            # 注意：这里的 50/70 = 0.714..., 原代码中是(50/70)
            # 如果原代码是写死的 (50/70)，请保持一致
            x_train, x_val, y_train, y_val = train_test_split(
                x_train_val, y_train_val, test_size=(10/70), random_state=42, stratify=y_train_val
            )
            # 使用训练集和验证集训练模型（不使用测试集）
            model = UnimodalModel(config)
            model.fit(x_train, y_train, x_val, y_val)
            # y_pred = model.predict(x_test)
            y_pred_proba = model.predict_proba(x_test)
            # 确保得到的是正类的概率值
            if len(y_pred_proba.shape) == 1 or y_pred_proba.shape[1] == 1:
                y_pred_proba = y_pred_proba.flatten()
            else:
                y_pred_proba = y_pred_proba[:, 1]

            # 将正类(1类)的概率传给AUC计算函数
            auc = roc_auc_score(y_test, y_pred_proba)
            print(f"Dataset: {name}, AUC: {auc:.4f}")
        except Exception as e:
            print(f"Error processing dataset {name}: {e}")

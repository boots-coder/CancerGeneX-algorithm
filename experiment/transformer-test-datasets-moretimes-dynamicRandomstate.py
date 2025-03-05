#!/usr/bin/env python
# coding=utf-8
from sklearn.model_selection import train_test_split
from Model import UnimodalModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import scipy.io
import os

if __name__ == '__main__':
    class_num = 2
    config = {
        "ClassNum": class_num,  # 注意这个参数一定要改
        "Metric": "acc",
        "DataType": {"Global"},

        "PreProcessors": {
            "Zscore": {
                "Type": "Standardization",
                "Method": "Zscore",
                "BuilderType": ["DL"],
                "FeaturesType": []
            },
        },

        "FeatureSelector": {
            "GCLasso": {
                "Type": "FeatureSelection",
                "Method": "GCLasso",
                "Parameter": {},
            },
            "RecallAttribute": {
                "name": "RecallAttribute",
                "Type": "RecallAttribute",
                "Method": "RecallAttribute",
                "Parameter": {0.1},
            },
        },

        "FeatureFusion": {
            "Name": "FeatureFusion",
            "BuilderType": ["ML", "DL"],
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
            "BuilderType": ["ML", "DL"],
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
            }
        },
    }


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

    #
    # # 数据集列表
    # datasets = [
    #     "ALLAML.mat", "colon.mat", "GLI_85.mat",
    #     "leukemia.mat", "Prostate_GE.mat", "SMK_CAN_187.mat"
    # ]
    # 数据集列表
    datasets = [
        "colon.mat"
    ]

    # 数据路径
    data_dir = "../data/"  # 根据实际情况修改

    # 重复实验次数
    n_runs = 10

    # 用字典存储各个指标的结果
    metrics_results = {
        'accuracy': {ds: [] for ds in datasets},
        'precision': {ds: [] for ds in datasets},
        'recall': {ds: [] for ds in datasets},
        'f1': {ds: [] for ds in datasets},
        'auc': {ds: [] for ds in datasets}
    }

    for run_idx in range(n_runs):
        print(f"\n===== Run {run_idx + 1}/{n_runs} =====")
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
                    x_train_val, y_train_val, test_size=(50 / 70),
                    random_state=42 + run_idx, stratify=y_train_val
                )

                # 初始化并训练模型
                model = UnimodalModel(config)
                model.fit(x_train, y_train, x_val, y_val)

                # 获取预测概率和类别
                y_pred_proba = model.predict_proba(x_test)
                if len(y_pred_proba.shape) == 1 or y_pred_proba.shape[1] == 1:
                    y_pred_proba = y_pred_proba.flatten()
                else:
                    y_pred_proba = y_pred_proba[:, 1]

                # 获取预测类别
                y_pred = (y_pred_proba > 0.5).astype(int)

                # 计算各项指标
                metrics_results['accuracy'][dataset].append(accuracy_score(y_test, y_pred))
                metrics_results['precision'][dataset].append(precision_score(y_test, y_pred))
                metrics_results['recall'][dataset].append(recall_score(y_test, y_pred))
                metrics_results['f1'][dataset].append(f1_score(y_test, y_pred))
                metrics_results['auc'][dataset].append(roc_auc_score(y_test, y_pred_proba))

                print(f"{dataset} - Run {run_idx + 1}:")
                print(f"Accuracy: {metrics_results['accuracy'][dataset][-1]:.4f}")
                print(f"Precision: {metrics_results['precision'][dataset][-1]:.4f}")
                print(f"Recall: {metrics_results['recall'][dataset][-1]:.4f}")
                print(f"F1: {metrics_results['f1'][dataset][-1]:.4f}")
                print(f"AUC: {metrics_results['auc'][dataset][-1]:.4f}")

            except Exception as e:
                # 错误处理：将所有指标都记录为NaN
                for metric in metrics_results.keys():
                    metrics_results[metric][dataset].append(np.nan)
                print(f"Error processing {dataset}: {e}")

    # 计算并打印最终结果
    print("\n===== Final Results (Mean ± Std) =====")
    results_list = []

    for dataset in datasets:
        results = {
            'Dataset': dataset,
            'Accuracy': f"{np.nanmean(metrics_results['accuracy'][dataset]):.4f} ± {np.nanstd(metrics_results['accuracy'][dataset]):.4f}",
            'Precision': f"{np.nanmean(metrics_results['precision'][dataset]):.4f} ± {np.nanstd(metrics_results['precision'][dataset]):.4f}",
            'Recall': f"{np.nanmean(metrics_results['recall'][dataset]):.4f} ± {np.nanstd(metrics_results['recall'][dataset]):.4f}",
            'F1': f"{np.nanmean(metrics_results['f1'][dataset]):.4f} ± {np.nanstd(metrics_results['f1'][dataset]):.4f}",
            'AUC': f"{np.nanmean(metrics_results['auc'][dataset]):.4f} ± {np.nanstd(metrics_results['auc'][dataset]):.4f}"
        }
        results_list.append(results)

    # 创建DataFrame
    results_df = pd.DataFrame(results_list)

    # 打印表格形式的结果
    print("\nResults Table:")
    print(results_df.to_string(index=False))

    # 保存结果到CSV文件
    results_df.to_csv('transformer-classification_results.csv', index=False)
    print("\nResults have been saved to 'classification_results.csv'")

    # 打印每个数据集的详细统计信息
    print("\n===== Detailed Statistics =====")
    for dataset in datasets:
        print(f"\nDataset: {dataset}")
        for metric in metrics_results.keys():
            values = np.array(metrics_results[metric][dataset])
            mean = np.nanmean(values)
            std = np.nanstd(values)
            max_val = np.nanmax(values)
            min_val = np.nanmin(values)

            print(f"{metric.capitalize()}:")
            print(f"  Mean ± Std: {mean:.4f} ± {std:.4f}")
            print(f"  Max: {max_val:.4f}")
            print(f"  Min: {min_val:.4f}")

    # 创建数值格式的结果DataFrame
    numeric_results_list = []
    for dataset in datasets:
        numeric_results = {
            'Dataset': dataset,
            'Accuracy_Mean': np.nanmean(metrics_results['accuracy'][dataset]),
            'Accuracy_Std': np.nanstd(metrics_results['accuracy'][dataset]),
            'Precision_Mean': np.nanmean(metrics_results['precision'][dataset]),
            'Precision_Std': np.nanstd(metrics_results['precision'][dataset]),
            'Recall_Mean': np.nanmean(metrics_results['recall'][dataset]),
            'Recall_Std': np.nanstd(metrics_results['recall'][dataset]),
            'F1_Mean': np.nanmean(metrics_results['f1'][dataset]),
            'F1_Std': np.nanstd(metrics_results['f1'][dataset]),
            'AUC_Mean': np.nanmean(metrics_results['auc'][dataset]),
            'AUC_Std': np.nanstd(metrics_results['auc'][dataset])
        }
        numeric_results_list.append(numeric_results)

    numeric_results_df = pd.DataFrame(numeric_results_list)
    numeric_results_df.to_csv('transformer-classification_results_numeric.csv', index=False)
    print("\ntransformer-Numeric results have been saved to 'classification_results_numeric.csv'")
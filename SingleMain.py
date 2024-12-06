#!/usr/bin/env python
# coding=utf-8
from sklearn.model_selection import train_test_split

from Model import UnimodalModel
from sklearn.metrics import roc_auc_score

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

            "CategoryImbalance": {
                "Name" : "SMOTE",
                "Type" : "CategoryImbalance",
                "Parameter": {},
            },


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

                "BNN" : {
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
        #
        # def load_data(config):
        #     import os.path
        #     import pandas as pd
        #     # load data
        #     mat = pd.read_csv(r"data.csv", sep=",")
        #     Y = mat["label"].values
        #     del mat["label"]
        #     mat = mat.set_index("Unnamed: 0")
        #     name = mat.columns.tolist()
        #     X = mat.values
        #
        #     config["ClassNum"] = 6
        #
        #     return X, Y,config
        import numpy as np
        import pandas as pd
        import scipy.io
        from sklearn.model_selection import train_test_split


        def load_data(name, file_path, config):
            """
            Load dataset based on the provided name and file path.
            Adjusts labels to ensure compatibility with a binary classification task.
            """
            if file_path.endswith('.csv'):
                # Load CSV file (e.g., merged_colon_data.csv or Prostate_GE_labels.csv)
                data = pd.read_csv(file_path, sep=",")
                y = np.int64(data['Label'])  # Extract labels
                del data['Label']  # Remove the label column from features
                X = data.values
            elif file_path.endswith('.mat'):
                # Load .mat file
                mat_data = scipy.io.loadmat(file_path)
                X = mat_data['X']  # Extract feature matrix
                y = np.int64(mat_data['Y']).flatten()  # Extract labels
            else:
                raise ValueError("Unsupported file format: {}".format(file_path))

            # Process labels based on dataset name
            if name in ['colon', 'leukemia']:
                y[y == -1] = 0  # Convert -1 to 0 for binary classification
            else:
                # Adjust labels for datasets with labels starting from 1 (e.g., Prostate_GE)
                y = y - 1  # Convert labels to start from 0

            # Update config for 2-class problem
            config["ClassNum"] = 2

            return X, y, config



        # Define datasets
        datasets = {
            # "colon": "merged_colon_data.csv",
            # "leukemia": "leukemia.mat",
            "GLI_85": "data/GLI_85.mat"
        }

        # Iterate over datasets
        for name, file_path in datasets.items():
            try:
                # Load dataset
                X, Y, config = load_data(name, file_path, config)

                # Split data: first into train+validation (70%) and test (30%)
                x_train_val, x_test, y_train_val, y_test = train_test_split(
                    X, Y, test_size=0.3, random_state=42, stratify=Y
                )

                # Further split train+validation (70%) into train (20%) and validation (50%)
                x_train, x_val, y_train, y_val = train_test_split(
                    x_train_val, y_train_val, test_size=(50 / 70), random_state=42, stratify=y_train_val
                )
                # Initialize and train the model
                model = UnimodalModel(config)
                model.fit(x_train, y_train, x_test, y_test)

                # Predict on test set
                y_pred = model.predict(x_test)

                # Evaluate the model (e.g., AUC or accuracy)
                auc = roc_auc_score(y_test, y_pred)
                print(f"Dataset: {name}, AUC: {auc:.4f}")

            except Exception as e:
                print(f"Error processing dataset {name}: {e}")
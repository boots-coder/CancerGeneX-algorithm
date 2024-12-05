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
        def load_data(config):
            import pandas as pd

            # load the merged colon data
            mat = pd.read_csv(r"merged_colon_data.csv", sep=",")

            # Separate the label (Y) from the feature matrix (X)
            Y = mat["Label"].values
            del mat["Label"]
            # 将标签从 -1 和 1 转换为 0 和 1
            Y[Y == -1] = 0

            # No need to set index; extract column names and features directly
            name = mat.columns.tolist()
            X = mat.values

            # Update config for 2-class problem (if applicable, you can adjust this)
            config["ClassNum"] = 2

            return X, Y, config


        # X_train, X_test, y_train, y_test, config = load_KIRP_cnv_data(config)
        X, y, config = load_data(config)
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)


        # print(X_train.shape)

        model = UnimodalModel(config)

        model.fit(X_train, y_train, X_test, y_test)
        y_predit = model.predict(X_test)

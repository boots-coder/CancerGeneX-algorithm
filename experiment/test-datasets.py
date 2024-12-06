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
        # def load_data(config):
        #     import pandas as pd
        #
        #     # load the merged colon data
        #     mat = pd.read_csv(r"merged_colon_data.csv", sep=",")
        #
        #     # Separate the label (Y) from the feature matrix (X)
        #     Y = mat["Label"].values
        #     del mat["Label"]
        #     # 将标签从 -1 和 1 转换为 0 和 1
        #     Y[Y == -1] = 0
        #
        #     # No need to set index; extract column names and features directly
        #     name = mat.columns.tolist()
        #     X = mat.values
        #
        #     # Update config for 2-class problem (if applicable, you can adjust this)
        #     config["ClassNum"] = 2
        #
        #     return X, Y, config
        #
        #
        # # X_train, X_test, y_train, y_test, config = load_KIRP_cnv_data(config)
        # X, y, config = load_data(config)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.4)
        #
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import scipy.io
import os


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


# List of datasets
datasets = [
    "ALLAML.mat", "colon.mat", "GLI_85.mat",
    "leukemia.mat", "Prostate_GE.mat", "SMK_CAN_187.mat"
]

# Directory containing datasets
data_dir = "../data/"  # Update the directory if necessary

# Dictionary to store AUC results
auc_results = {}

# Loop through each dataset
for dataset in datasets:
    file_path = os.path.join(data_dir, dataset)
    try:
        # Load dataset
        X, Y = load_data_from_mat(file_path)

        # Handle high-dimensional data (optional PCA for datasets with large feature sets)
        # if X.shape[1] > 5000:  # If more than 5000 features
        #     print(f"Applying PCA to reduce features for {dataset}")
        #     pca = PCA(n_components=50)  # Retain 50 principal components
        #     X = pca.fit_transform(X)

        # Split data: first into train+validation (70%) and test (30%)
        x_train_val, x_test, y_train_val, y_test = train_test_split(
            X, Y, test_size=0.3, random_state=42, stratify=Y
        )

        # Further split train+validation (70%) into train (20%) and validation (50%)
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_val, y_train_val, test_size=(50 / 70), random_state=42, stratify=y_train_val
        )
        # Initialize and train your model (replace UnimodalModel with your model class)
        model = UnimodalModel(config)  # Replace UnimodalModel with your actual model
        model.fit(x_train, y_train, x_val, y_val)

        # Predict probabilities for the test set
        y_pred_proba = model.predict(x_test)

        # Ensure y_pred_proba is probabilities for the positive class
        if len(y_pred_proba.shape) == 1 or y_pred_proba.shape[1] == 1:
            # Assume binary classification and the model outputs probabilities directly
            y_pred_proba = y_pred_proba.flatten()
        else:
            # Extract the probabilities for class 1 if multi-class probabilities are returned
            y_pred_proba = y_pred_proba[:, 1]

        # Calculate AUC
        auc = roc_auc_score(y_test, y_pred_proba)
        auc_results[dataset] = auc

        print(f"{dataset}: AUC = {auc:.4f}")

    except Exception as e:
        auc_results[dataset] = f"Error: {e}"
        print(f"Error processing {dataset}: {e}")

# Print AUC results summary
print("\nAUC Results Summary:")
for dataset, auc in auc_results.items():
    print(f"{dataset}: {auc}")
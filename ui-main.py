import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QFileDialog, QComboBox, QCheckBox, QGroupBox,
                             QSpinBox, QDoubleSpinBox, QMessageBox)
from PyQt5.QtCore import Qt
import json
import numpy as np
import scipy.io
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 导入自定义模型
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Model import UnimodalModel


def convert_sets_to_lists(obj):
    if isinstance(obj, dict):
        return {key: convert_sets_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_sets_to_lists(item) for item in obj]
    elif isinstance(obj, set):
        return list(obj)
    return obj


class ModelConfigUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.config = self.get_default_config()

        # 设置默认值
        current_dir = os.path.dirname(os.path.abspath(__file__))
        default_path = os.path.join(current_dir, "data", "SMK_CAN_187.mat")
        self.file_path.setText(default_path)

        # 默认选中所有分类器
        for checkbox in self.classifier_checkboxes.values():
            checkbox.setChecked(True)

    def get_default_config(self):
        return {
            "ClassNum": 2,
            "Metric": "acc",
            "DataType": ["Global"],
            "PreProcessors": {
                "MinMax": {
                    "Type": "Standardization",
                    "Method": "DecimalScale",
                    "BuilderType": ["DL"],
                    "FeaturesType": []
                }
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
                    "Parameter": [0.1],
                },
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
                }
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
                    "BaseClassifier": {}
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
                        "name": "BNN",
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
            }
        }

    def init_ui(self):
        self.setWindowTitle('机器学习模型配置界面')
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 文件选择部分
        file_group = QGroupBox("数据文件选择")
        file_layout = QVBoxLayout()

        file_select_layout = QHBoxLayout()
        self.file_path = QLineEdit()
        browse_btn = QPushButton("浏览")
        browse_btn.clicked.connect(self.browse_file)
        file_select_layout.addWidget(QLabel("数据文件:"))
        file_select_layout.addWidget(self.file_path)
        file_select_layout.addWidget(browse_btn)

        feature_label_layout = QHBoxLayout()
        self.feature_name = QLineEdit()
        self.label_name = QLineEdit()
        feature_label_layout.addWidget(QLabel("特征列名:"))
        feature_label_layout.addWidget(self.feature_name)
        feature_label_layout.addWidget(QLabel("标签列名:"))
        feature_label_layout.addWidget(self.label_name)

        file_layout.addLayout(file_select_layout)
        file_layout.addLayout(feature_label_layout)
        file_group.setLayout(file_layout)

        # 模型配置部分
        model_group = QGroupBox("模型配置")
        model_layout = QVBoxLayout()

        class_num_layout = QHBoxLayout()
        self.class_num = QSpinBox()
        self.class_num.setMinimum(2)
        self.class_num.setValue(2)
        class_num_layout.addWidget(QLabel("分类数量:"))
        class_num_layout.addWidget(self.class_num)
        class_num_layout.addStretch()

        split_layout = QHBoxLayout()
        self.train_split = QDoubleSpinBox()
        self.val_split = QDoubleSpinBox()
        self.test_split = QDoubleSpinBox()
        for split in [self.train_split, self.val_split, self.test_split]:
            split.setRange(0, 1)
            split.setSingleStep(0.1)
            split.setDecimals(2)

        self.train_split.setValue(0.2)
        self.val_split.setValue(0.5)
        self.test_split.setValue(0.3)

        split_layout.addWidget(QLabel("训练集比例:"))
        split_layout.addWidget(self.train_split)
        split_layout.addWidget(QLabel("验证集比例:"))
        split_layout.addWidget(self.val_split)
        split_layout.addWidget(QLabel("测试集比例:"))
        split_layout.addWidget(self.test_split)

        model_layout.addLayout(class_num_layout)
        model_layout.addLayout(split_layout)
        model_group.setLayout(model_layout)

        # 基础分类器选择
        classifier_group = QGroupBox("基础分类器选择")
        classifier_layout = QVBoxLayout()

        self.classifier_checkboxes = {}
        classifiers = {
            "RandomForestClassifier": "随机森林",
            "ExtraTreesClassifier": "极端随机树",
            "GaussianNBClassifier": "高斯朴素贝叶斯",
            "BernoulliNBClassifier": "伯努利朴素贝叶斯",
            "KNeighborsClassifier": "K近邻",
            "GradientBoostingClassifier": "梯度提升",
            "SVCClassifier": "支持向量机",
            "LogisticRegressionClassifier": "逻辑回归"
        }

        for key, value in classifiers.items():
            cb = QCheckBox(value)
            self.classifier_checkboxes[key] = cb
            classifier_layout.addWidget(cb)

        classifier_group.setLayout(classifier_layout)

        # 添加所有组件到主布局
        main_layout.addWidget(file_group)
        main_layout.addWidget(model_group)
        main_layout.addWidget(classifier_group)

        # 运行和重置按钮
        button_layout = QHBoxLayout()
        run_btn = QPushButton("运行模型")
        reset_btn = QPushButton("重置")
        run_btn.clicked.connect(self.run_model)
        reset_btn.clicked.connect(self.reset_config)
        button_layout.addWidget(run_btn)
        button_layout.addWidget(reset_btn)
        main_layout.addLayout(button_layout)

    def browse_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "选择数据文件",
            "",
            "MAT files (*.mat);;CSV files (*.csv);;Excel files (*.xlsx *.xls)"
        )
        if file_name:
            self.file_path.setText(file_name)

    def reset_config(self):
        """重置所有配置到默认值"""
        self.config = self.get_default_config()
        self.class_num.setValue(2)
        self.train_split.setValue(0.2)
        self.val_split.setValue(0.5)
        self.test_split.setValue(0.3)
        for checkbox in self.classifier_checkboxes.values():
            checkbox.setChecked(True)
        QMessageBox.information(self, "提示", "配置已重置为默认值")

    def get_selected_classifiers(self):
        base_classifiers = {}
        default_params = {
            "RandomForestClassifier": {
                "Type": "RandomForestClassifier",
                "Parameter": {"n_estimators": 100, "criterion": "gini", "class_weight": None, "random_state": 0}
            },
            "ExtraTreesClassifier": {
                "Type": "ExtraTreesClassifier",
                "Parameter": {"n_estimators": 100, "criterion": "gini", "class_weight": None, "random_state": 0}
            },
            "GaussianNBClassifier": {
                "Type": "GaussianNBClassifier",
                "Parameter": {}
            },
            "BernoulliNBClassifier": {
                "Type": "BernoulliNBClassifier",
                "Parameter": {}
            },
            "KNeighborsClassifier": {
                "Type": "KNeighborsClassifier",
                "Parameter": {"n_neighbors": 5}
            },
            "GradientBoostingClassifier": {
                "Type": "GradientBoostingClassifier",
                "Parameter": {}
            },
            "SVCClassifier": {
                "Type": "SVCClassifier",
                "Parameter": {"kernel": "rbf", "probability": True}
            },
            "LogisticRegressionClassifier": {
                "Type": "LogisticRegressionClassifier",
                "Parameter": {"penalty": 'l2'}
            }
        }

        for name, checkbox in self.classifier_checkboxes.items():
            if checkbox.isChecked():
                base_classifiers[name] = default_params[name]

        return base_classifiers

    def load_data(self, file_path, config):
        """加载数据集并进行预处理"""
        try:
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
                raise ValueError("不支持的文件格式")

            # 处理标签
            unique_labels = np.unique(y)
            if len(unique_labels) > 2:
                raise ValueError("当前只支持二分类问题")

            # 确保标签为0和1
            if -1 in unique_labels:
                y[y == -1] = 0
            elif min(unique_labels) == 1:
                y = y - 1

            config["ClassNum"] = 2
            return X, y, config

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载数据时出错：{str(e)}")
            return None, None, None

    def train_model(self, X, y, splits):
        """训练模型并返回结果"""
        try:
            # 数据集划分
            x_train_val, x_test, y_train_val, y_test = train_test_split(
                X, y,
                test_size=splits["test"],
                random_state=42,
                stratify=y
            )

            x_train, x_val, y_train, y_val = train_test_split(
                x_train_val, y_train_val,
                test_size=splits["val"] / (splits["train"] + splits["val"]),
                random_state=42,
                stratify=y_train_val
            )

            # 初始化和训练模型
            model = UnimodalModel(self.config)
            model.fit(x_train, y_train, x_val, y_val)

            # 预测和评估
            y_pred_proba = model.predict_proba(x_test)
            if len(y_pred_proba.shape) == 1 or y_pred_proba.shape[1] == 1:
                y_pred_proba = y_pred_proba.flatten()
            else:
                y_pred_proba = y_pred_proba[:, 1]

            auc = roc_auc_score(y_test, y_pred_proba)
            return auc, model

        except Exception as e:
            QMessageBox.critical(self, "错误", f"训练模型时出错：{str(e)}")
            return None, None

    def run_model(self):
        # 验证输入
        total_split = self.train_split.value() + self.val_split.value() + self.test_split.value()
        if abs(total_split - 1.0) > 0.001:
            QMessageBox.warning(self, "警告", "数据集划分比例之和必须为1")
            return

        if not self.file_path.text():
            QMessageBox.warning(self, "警告", "请选择数据文件")
            return

        if not any(cb.isChecked() for cb in self.classifier_checkboxes.values()):
            QMessageBox.warning(self, "警告", "请至少选择一个分类器")
            return

        # 更新配置
        self.config["ClassNum"] = self.class_num.value()
        self.config["CascadeClassifier"]["AdaptiveEnsembleClassifyByNum"]["BaseClassifier"] = \
            self.get_selected_classifiers()

        # 加载数据
        X, y, config = self.load_data(self.file_path.text(), self.config)
        if X is None:
            return

        # 获取划分比例
        splits = {
            "train": self.train_split.value(),
            "val": self.val_split.value(),
            "test": self.test_split.value()
        }

        # 训练模型
        auc, model = self.train_model(X, y, splits)
        if auc is not None:
            QMessageBox.information(self, "结果", f"模型训练完成！\nAUC评分：{auc:.4f}")

        # 保存配置和结果
        result = {
            "config": convert_sets_to_lists(self.config),
            "file_path": self.file_path.text(),
            "feature_name": self.feature_name.text(),
            "label_name": self.label_name.text(),
            "splits": splits,
            "performance": {"auc": auc if auc is not None else None}
        }

        # 将配置保存到文件
        try:
            save_dir = os.path.join(os.path.dirname(self.file_path.text()), 'results')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'model_config.json')

            with open(save_path, 'w') as f:
                json.dump(result, f, indent=2)
            QMessageBox.information(self, "保存成功", f"配置已保存到：\n{save_path}")
        except Exception as e:
            QMessageBox.warning(self, "保存警告", f"保存配置时出错：{str(e)}")

        return result, model


def main():
    app = QApplication(sys.argv)
    window = ModelConfigUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
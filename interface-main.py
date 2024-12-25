import sys
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import json
import numpy as np
import scipy.io
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import time
import threading
import logging
from datetime import datetime
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Model import UnimodalModel

app = Flask(__name__)
CORS(app)

# 配置
RECEIVE_DATA_FOLDER = 'receive-data'
ALLOWED_EXTENSIONS = {'mat', 'csv', 'xlsx', 'xls'}
if not os.path.exists(RECEIVE_DATA_FOLDER):
    os.makedirs(RECEIVE_DATA_FOLDER)

# 全局变量存储训练任务
training_tasks = {}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_base_feature_selectors(selected_selectors=None):
    """获取特征选择器配置"""
    all_selectors = {
        "GCLasso": {
            "Type": "FeatureSelection",
            "Method": "GCLasso",
            "Parameter": {},
        },
        "GCFClassif": {
            "Type": "FeatureSelection",
            "Method": "GCFClassif",
            "Parameter": {},
        },
        "GCVariance": {
            "Type": "FeatureSelection",
            "Method": "GCVariance",
            "Parameter": {"threshold": 0.01},
        },
        "GCMutualInfo": {
            "Type": "FeatureSelection",
            "Method": "GCMutualInfo",
            "Parameter": {"threshold": 0.05},
        },
        "GCChiSquare": {
            "Type": "FeatureSelection",
            "Method": "GCChiSquare",
            "Parameter": {"P": 0.05},
        },
        "GCCorrelation": {
            "Type": "FeatureSelection",
            "Method": "GCCorrelation",
            "Parameter": {"threshold": 0.1},
        }
    }

    if selected_selectors is None:
        return {"GCLasso": all_selectors["GCLasso"]}  # 默认使用GCLasso

    # 如果传入的是字符串，转换为列表
    if isinstance(selected_selectors, str):
        selected_selectors = [selected_selectors]

    return {k: all_selectors[k] for k in selected_selectors if k in all_selectors}


def get_base_classifiers(selected_classifiers=None):
    """获取基础分类器配置"""
    all_classifiers = {
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
        }
    }

    if selected_classifiers is None:
        return all_classifiers
    return {k: all_classifiers[k] for k in selected_classifiers if k in all_classifiers}


def build_model_config(class_num=2, selected_classifiers=None, selected_selectors=None):
    """构建模型配置"""
    return {
        "ClassNum": class_num,
        "Metric": "acc",
        "DataType": {"Global"},
        "PreProcessors": {
            "MinMax": {
                "Type": "Standardization",
                "Method": "DecimalScale",
                "BuilderType": ["DL"],
                "FeaturesType": []
            }
        },
        "FeatureSelector": get_base_feature_selectors(selected_selectors),
        "FeatureFusion": {
            "Name": "FeatureFusion",
            "BuilderType": ["ML", "DL", "cluster"],
            "Type": "FeatureConcatenation",
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
                "BaseClassifier": get_base_classifiers(selected_classifiers)
            }
        }
    }


def load_data(file_path):
    """加载数据集"""
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
            raise ValueError("Unsupported file format")

        # 处理标签
        unique_labels = np.unique(y)
        if len(unique_labels) > 2:
            raise ValueError("Currently only supports binary classification")

        if -1 in unique_labels:
            y[y == -1] = 0
        elif min(unique_labels) == 1:
            y = y - 1

        return X, y

    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


@app.route('/api/train', methods=['POST'])
def train():
    """训练接口"""
    try:
        # 检查文件
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({"status": "error", "message": "Invalid file"}), 400

        # 解析配置
        config = json.loads(request.form.get('config', '{}'))
        selected_classifiers = config.get('classifiers', None)
        selected_selectors = config.get('selectors', None)
        splits = config.get('splits', {'train': 0.2, 'val': 0.5, 'test': 0.3})

        # 保存文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(RECEIVE_DATA_FOLDER, timestamp)
        os.makedirs(save_dir)
        file_path = os.path.join(save_dir, secure_filename(file.filename))
        file.save(file_path)

        # 创建任务
        task_id = f"task_{timestamp}"
        training_tasks[task_id] = {
            'status': 'running',
            'progress': 0,
            'start_time': datetime.now().isoformat()
        }

        def training_thread():
            try:
                # 加载数据
                X, y = load_data(file_path)
                model_config = build_model_config(
                    class_num=2,
                    selected_classifiers=selected_classifiers,
                    selected_selectors=selected_selectors
                )

                # 数据集划分
                x_train_val, x_test, y_train_val, y_test = train_test_split(
                    X, y, test_size=splits['test'], random_state=42, stratify=y
                )

                val_size = splits['val'] / (splits['train'] + splits['val'])
                x_train, x_val, y_train, y_val = train_test_split(
                    x_train_val, y_train_val, test_size=val_size,
                    random_state=42, stratify=y_train_val
                )

                # 训练模型
                training_tasks[task_id]['progress'] = 30
                model = UnimodalModel(model_config)
                model.fit(x_train, y_train, x_val, y_val)

                # 评估
                training_tasks[task_id]['progress'] = 80
                y_pred_proba = model.predict_proba(x_test)
                y_pred = model.predict(x_test)

                # 计算指标
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'f1': f1_score(y_test, y_pred),
                    'auc': roc_auc_score(y_test, y_pred_proba[:, 1] if len(y_pred_proba.shape) > 1 else y_pred_proba)
                }

                training_tasks[task_id].update({
                    'status': 'completed',
                    'progress': 100,
                    'metrics': metrics,
                    'end_time': datetime.now().isoformat()
                })

            except Exception as e:
                logger.error(f"Training error: {str(e)}")
                training_tasks[task_id].update({
                    'status': 'failed',
                    'error': str(e)
                })

        # 启动训练线程
        thread = threading.Thread(target=training_thread)
        thread.start()

        return jsonify({
            "status": "success",
            "task_id": task_id,
            "message": "Training started"
        })

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/train/status/<task_id>', methods=['GET'])
def get_status(task_id):
    """获取训练状态"""
    task = training_tasks.get(task_id)
    if not task:
        return jsonify({"status": "error", "message": "Task not found"}), 404

    response = {
        "status": task['status'],
        "progress": task['progress']
    }

    if task['status'] == 'completed':
        response.update({
            "metrics": task['metrics'],
            "training_time": str(datetime.fromisoformat(task['end_time']) -
                                 datetime.fromisoformat(task['start_time']))
        })
    elif task['status'] == 'failed':
        response["error"] = task.get('error', 'Unknown error')

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
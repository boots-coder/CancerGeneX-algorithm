import torch
import numpy as np

from Classification.Common.ClassifierTemplate import ClassifierTemplate
class ModelWrapper(ClassifierTemplate):

    def __init__(self, name, est_class, config):
        super(ModelWrapper, self).__init__(name, config)
        self.est_class = est_class
        self.est_args = config.get("Parameter", None)
        self._init_estimator(est_class, self.est_args)
        # 根据 CUDA 可用性设置 self.cuda
        self.cuda = torch.cuda.is_available()
        print(f"CUDA 可用性: {self.cuda}")

    def _init_estimator(self, est_class, est_args):
        self.est = est_class(**est_args)
        # 将模型移动到相应的设备
        device = torch.device('cuda' if self.cuda else 'cpu')
        print(device)
        device = torch.device('cpu')
        self.est.to(device)

    def predict(self, X):
        return np.argmax(self.predict_probs(X), axis=1)

    def predict_probs(self, X):
        with torch.no_grad():
            X = self.convert_data_to_tensor(X)
            _, outputs = self.est(X)
            outputs = self.convert_data_to_numpy(outputs)
            return outputs

    def convert_data_to_tensor(self, X):
        X = torch.tensor(X).float()
        device = torch.device('cuda' if self.cuda else 'cpu')
        return X.to(device)

    def convert_data_to_numpy(self, X):
        return X.cpu().detach().numpy()

    def obtain_features(self, X):
        with torch.no_grad():
            X = self.convert_data_to_tensor(X)
            features, _ = self.est(X)
            features = self.convert_data_to_numpy(features)
            return features

    def __call__(self, *args, **kwargs):
        return self.est(*args, **kwargs)

    def parameters(self):
        return self.est.parameters()

    def cuda(self):
        if self.cuda:
            self.est = self.est.cuda()
        else:
            print("CUDA 不可用，模型保持在 CPU 上")
        return self

    def obtain_instance(self):
        return self.est
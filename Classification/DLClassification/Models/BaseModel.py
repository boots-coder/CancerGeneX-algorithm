import torch.nn as nn
import torch

from Classification.DLClassification.Models.RootModel import ModelWrapper

def get_dl_model(name, kwargs):
    if name == "DNN":
        return DNNWrapper(name, kwargs)
    if name == "BNN":
        return BNNWrapper(name, kwargs)
    else:
        raise ValueError(f"Unsupported model name: {name}")

class DNNWrapper(ModelWrapper):
    def __init__(self, name, kwargs):
        super(DNNWrapper, self).__init__(name, DNNWrapper.DNN, kwargs)
        # 提取参数
        input_size = kwargs['Parameter']['InputSize']
        class_num = kwargs['Parameter']['ClassNum']

        # 定义模型
        self.model = DNNWrapper.DNN(input_size, class_num)
        self.model.to(self.device)  # 将模型移到设备上

    def to(self, device):
        # 将模型移动到指定设备
        self.model = self.model.to(device)
        self.device = device
        return self

    class DNN(nn.Module):
        def __init__(self, InputSize, ClassNum):
            super(DNNWrapper.DNN, self).__init__()
            self.dense = nn.Linear(InputSize, 32, bias=False)
            self.relu = nn.ReLU()
            self.batchnorm = nn.BatchNorm1d(32)
            self.dense2 = nn.Linear(32, ClassNum)
            self.softmax = nn.Softmax(dim=1)
            self.device = torch .device('cuda' if torch.cuda.is_available() else 'mps')


        def forward(self, x):
            # 确保输入数据在正确的设备上
            x = x.to(self.device)
            x = self.dense(x)
            x = self.batchnorm(x)
            feature = self.relu(x)
            out = self.dense2(feature)
            out = self.softmax(out)
            return feature, out


class BNNWrapper(ModelWrapper):
    def __init__(self, name, kwargs):
        super(BNNWrapper, self).__init__(name, BNNWrapper.BNN, kwargs)
        # 提取参数
        input_size = kwargs['Parameter']['InputSize']
        class_num = kwargs['Parameter']['ClassNum']

        # 定义模型
        self.model = BNNWrapper.BNN(input_size, class_num)
        self.model.to(self.device)  # 将模型移到设备上

    def to(self, device):
        # 将模型移动到指定设备
        self.model = self.model.to(device)
        self.device = device
        return self

    class BNN(nn.Module):
        def __init__(self, InputSize, ClassNum):
            super(BNNWrapper.BNN, self).__init__()
            self.dense_1 = nn.Linear(InputSize, 32, bias=False)
            self.dense_2 = nn.Linear(InputSize, 32, bias=False)
            self.dense_3 = nn.Linear(InputSize, 32, bias=False)
            self.relu_1 = nn.ReLU()
            self.relu_2 = nn.ReLU()
            self.sig = nn.Sigmoid()
            self.batchnorm = nn.BatchNorm1d(32)
            self.dense2 = nn.Linear(32, ClassNum)
            self.softmax = nn.Softmax(dim=1)
            self.device = torch .device('cuda' if torch.cuda.is_available() else 'mps')

        def forward(self, x):
            # 确保输入数据在正确的设备上
            x = x.to(self.device)
            f1 = self.dense_1(x)
            f1 = self.batchnorm(f1)
            f1 = self.relu_1(f1)

            f2 = self.dense_2(x)
            f2 = self.batchnorm(f2)
            f2 = self.relu_2(f2)

            f2_sig = self.dense_3(x)
            f2_sig = self.batchnorm(f2_sig)
            f2_sig = self.sig(f2_sig)

            f2 = torch.mul(f2, f2_sig)

            feature = torch.mul(f1, f2)

            out = self.dense2(feature)
            out = self.softmax(out)
            return feature, out

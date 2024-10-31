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
        self.device = torch.device('cpu')  # 默认设备为 CPU


    def to(self, device):
        # 将模型移动到指定设备
        self.model = self.model.to(device)
        return self

    class DNN(nn.Module):
        def __init__(self, InputSize, ClassNum):
            super(DNNWrapper.DNN, self).__init__()
            self.dense = nn.Linear(InputSize, 32, bias=False)
            self.relu = nn.ReLU()
            self.batchnorm = nn.BatchNorm1d(32)
            self.dense2 = nn.Linear(32, ClassNum)
            self.softmax = nn.Softmax(dim=1)


        def forward(self, x):
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
        self.device = torch.device('cpu')  # 默认设备为 CPU


    def to(self, device):
        # 将模型移动到指定设备
        self.model = self.model.to(device)
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

        def forward(self, x):

            # 检查 x 的形状是否正确
            if x.shape[1] != self.dense_1.in_features:
                # print("输入 x 的第二维度与 dense_1 的输入特征数量不匹配")
                # print("调整 x 的形状")
                x = x.transpose(0, 1)
                # print("调整后的 x 形状:", x.shape)

            f1 = self.dense_1(x)
            # print("经过 dense_1 后的形状:", f1.shape)  # 应该是 [batch_size, 32]

            # # 检查 f1 的形状
            # if f1.shape[1] != 32:
            #     print("f1 的形状不正确，可能需要转置")
            #     f1 = f1.transpose(0, 1)
            #     print("转置后的 f1 形状:", f1.shape)
            #
            # # 检查 f1 是否为内存连续的
            # if not f1.is_contiguous():
            #     f1 = f1.contiguous()
            #
            # # 检查 f1 是否包含异常值
            # if torch.isnan(f1).any() or torch.isinf(f1).any():
            #     print("f1 包含 NaN 或 Inf，无法执行 batchnorm")
            #     return

            # 执行 batchnorm
            f1 = self.batchnorm(f1)
            # print("经过 batchnorm 后的形状（f1）:", f1.shape)

            f1 = self.relu_1(f1)
            # print("经过 ReLU_1 后的形状:", f1.shape)

            f2 = self.dense_2(x)
            # print("经过 dense_2 后的形状:", f2.shape)
            f2 = self.batchnorm(f2)
            # print("经过 batchnorm 后的形状（f2）:", f2.shape)
            f2 = self.relu_2(f2)
            # print("经过 ReLU_2 后的形状:", f2.shape)

            f2_sig = self.dense_3(x)
            # print("经过 dense_3 后的形状:", f2_sig.shape)
            f2_sig = self.batchnorm(f2_sig)
            # print("经过 batchnorm 后的形状（f2_sig）:", f2_sig.shape)
            f2_sig = self.sig(f2_sig)
            # print("经过 sigmoid 后的形状:", f2_sig.shape)

            f2 = torch.mul(f2, f2_sig)
            # print("f2 与 f2_sig 逐元素相乘后的形状:", f2.shape)

            feature = torch.mul(f1, f2)
            # print("f1 与 f2 逐元素相乘后的形状（feature）:", feature.shape)

            out = self.dense2(feature)
            # print("经过 dense2 层后的形状:", out.shape)
            out = self.softmax(out)
            # print("经过 softmax 后的形状:", out.shape)
            # print("BNN forward 执行结束")
            return feature, out
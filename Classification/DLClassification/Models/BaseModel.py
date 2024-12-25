import torch.nn as nn
import torch

from Classification.DLClassification.Models.RootModel import ModelWrapper

def get_dl_model(name, kwargs):
    if name == "DNN":
        return DNNWrapper(name, kwargs)
    if name == "BNN":
        return BNNWrapper(name, kwargs)
    if name == "Transformer":
        return TransformerWrapper(name, kwargs)
    else:
        raise ValueError(f"Unsupported model name: {name}")



"""
DNN: 深度神经网络（Deep Neural Network）
	•	它是一个经典的全连接神经网络（Fully Connected Neural Network）。
"""
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

"""
BNN: 双分支神经网络（Bifurcated Neural Network）
	•	它是一种特殊的神经网络，拥有两个分支用于并行处理输入数据并最终合并。
	•	在代码中，BNNWrapper 类封装了这种双分支结构。其特点包括：
"""
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


import torch.nn as nn
import torch
import math

from Classification.DLClassification.Models.RootModel import ModelWrapper




"""
Transformer: 基于Transformer架构的特征提取器
    • 使用多头自注意力机制处理输入特征
    • 包含位置编码以保持特征的位置信息
    • 使用前馈神经网络进行特征转换
主要参数：
nhead：注意力头数量（默认4）
num_layers：Transformer编码器层数（默认2）
dim_feedforward：前馈网络隐藏层维度（默认128）
model_dim：内部特征维度（固定为64）
"""


class TransformerWrapper(ModelWrapper):
    def __init__(self, name, kwargs):
        super(TransformerWrapper, self).__init__(name, TransformerWrapper.TransformerNet, kwargs)
        # 提取参数
        input_size = kwargs['Parameter']['InputSize']
        class_num = kwargs['Parameter']['ClassNum']

        # 定义模型
        self.model = TransformerWrapper.TransformerNet(input_size, class_num)
        self.model.to(self.device)


    def to(self, device):
        self.model = self.model.to(device)
        self.device = device
        return self

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=5000):
            super(TransformerWrapper.PositionalEncoding, self).__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)

        def forward(self, x):
            return x + self.pe[:, :x.size(1)]

    class TransformerNet(nn.Module):
        def __init__(self, InputSize, ClassNum, nhead=8, num_layers=16, dim_feedforward=256):
            super(TransformerWrapper.TransformerNet, self).__init__()

            self.model_dim = 64  # 将输入转换到这个维度

            # 输入映射层
            self.input_projection = nn.Linear(InputSize, self.model_dim)

            # 位置编码
            self.pos_encoder = TransformerWrapper.PositionalEncoding(self.model_dim)

            # Transformer编码器层
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.model_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            # 输出层
            self.feature_layer = nn.Linear(self.model_dim, 32)
            self.dense = nn.Linear(32, ClassNum)
            self.softmax = nn.Softmax(dim=1)

            self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

        def forward(self, x):
            # 确保输入数据在正确的设备上
            x = x.to(self.device)

            # 将输入reshape成序列形式 [batch_size, sequence_length=1, input_size]
            if len(x.shape) == 2:
                x = x.unsqueeze(1)

            # 投影到模型维度
            x = self.input_projection(x)

            # 添加位置编码
            x = self.pos_encoder(x)

            # Transformer编码器
            transformer_output = self.transformer_encoder(x)

            # 取序列的平均值作为特征
            feature = torch.mean(transformer_output, dim=1)

            # 通过特征层
            feature = self.feature_layer(feature)

            # 分类层
            out = self.dense(feature)
            out = self.softmax(out)

            return feature, out
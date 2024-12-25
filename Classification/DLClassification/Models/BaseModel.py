import torch.nn as nn
import torch
import torch.nn.functional as F
import math
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


class DNNWrapper(ModelWrapper):
    def __init__(self, name, kwargs):
        super(DNNWrapper, self).__init__(name, DNNWrapper.DNN, kwargs)
        # 提取参数
        input_size = kwargs['Parameter']['InputSize']
        ClassNum = kwargs['Parameter']['ClassNum']

        # 定义模型
        self.model = DNNWrapper.DNN(input_size, ClassNum)
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
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

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
        ClassNum = kwargs['Parameter']['ClassNum']

        # 定义模型
        self.model = BNNWrapper.BNN(input_size, ClassNum)
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
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

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


class TransformerWrapper(ModelWrapper):
    def __init__(self, name, kwargs):
        super(TransformerWrapper, self).__init__(name, TransformerWrapper.TransformerNet, kwargs)
        # 提取参数
        input_size = kwargs['Parameter']['InputSize']
        ClassNum = kwargs['Parameter']['ClassNum']

        # 动态调整模型参数
        nhead = max(8, min(16, input_size // 64))
        num_layers = max(8, min(16, input_size // 128))
        dim_feedforward = max(256, min(512, input_size * 2))

        self.model = TransformerWrapper.TransformerNet(InputSize=input_size, ClassNum=ClassNum,
                                                       nhead=nhead, num_layers=num_layers,
                                                       dim_feedforward=dim_feedforward)
        self.model.to(self.device)

    def to(self, device):
        self.model = self.model.to(device)
        self.device = device
        return self

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=5000):
            super(TransformerWrapper.PositionalEncoding, self).__init__()
            self.d_model = d_model
            self.register_buffer('pe', self._create_pe(max_len, d_model))

        def _create_pe(self, max_len, d_model):
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

            # 确保div_term的维度正确
            if d_model % 2 == 0:
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
            else:
                pe[:, 0::2] = torch.sin(position * div_term[:d_model // 2])
                pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
                pe[:, -1] = torch.sin(position * div_term[-1])  # 处理奇数维度

            return pe.unsqueeze(0)

        def forward(self, x):
            # 确保位置编码维度与输入维度匹配
            return x + self.pe[:, :x.size(1), :x.size(2)]

    class AttentionPooling(nn.Module):
        def __init__(self, dim):
            super(TransformerWrapper.AttentionPooling, self).__init__()
            self.attention = nn.Sequential(
                nn.Linear(dim, dim),
                nn.Tanh(),
                nn.Linear(dim, 1)
            )

        def forward(self, x):
            # x shape: [batch_size, seq_len, dim]
            attention_weights = F.softmax(self.attention(x), dim=1)
            attended = torch.sum(attention_weights * x, dim=1)
            return attended

    class TransformerNet(nn.Module):
        def __init__(self, InputSize, ClassNum, nhead=8, num_layers=16, dim_feedforward=256):
            super(TransformerWrapper.TransformerNet, self).__init__()

            # 确保model_dim是head数量的倍数
            self.model_dim = max(64, min(256, (InputSize // 4) // nhead * nhead))

            # 输入映射层
            self.input_projection = nn.Sequential(
                nn.Linear(InputSize, self.model_dim),
                nn.LayerNorm(self.model_dim)
            )

            # 位置编码
            self.pos_encoder = TransformerWrapper.PositionalEncoding(self.model_dim)

            # Transformer编码器层
            encoder_layers = []
            for i in range(num_layers):
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.model_dim,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    batch_first=True,
                    dropout=0.1
                )
                encoder_layers.append(encoder_layer)
            self.transformer_encoder = nn.ModuleList(encoder_layers)

            # 注意力池化
            self.attention_pooling = TransformerWrapper.AttentionPooling(self.model_dim)

            # 特征层
            self.feature_layer = nn.Sequential(
                nn.Linear(self.model_dim, 32),
                nn.LayerNorm(32),
                nn.ReLU()
            )

            # 分类层
            self.dense = nn.Linear(32, ClassNum)
            self.softmax = nn.Softmax(dim=1)

            self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

        def forward(self, x):
            x = x.to(self.device)

            # 调整输入维度
            if len(x.shape) == 2:
                x = x.unsqueeze(1)

            # 输入投影
            x = self.input_projection(x)

            try:
                # 位置编码
                x = self.pos_encoder(x)

                # Transformer编码
                for encoder_layer in self.transformer_encoder:
                    x = encoder_layer(x)

                # 注意力池化
                pooled = self.attention_pooling(x)

                # 特征提取
                feature = self.feature_layer(pooled)

                # 分类
                out = self.dense(feature)
                out = self.softmax(out)

                return feature, out

            except RuntimeError as e:
                print(f"Error in forward pass. Input shape: {x.shape}, Model dim: {self.model_dim}")
                raise e
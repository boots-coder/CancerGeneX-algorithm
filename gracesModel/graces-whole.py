import os  # Added this line to import the os module
import copy
import numpy as np
import torch_geometric.nn as gnn
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from sklearn import svm
import warnings
import scipy.io
import logging
from datetime import datetime

warnings.filterwarnings("ignore")

def get_device(preferred='cuda'):
    if preferred == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print("使用设备: CUDA")
    elif preferred == 'mps' and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
        print("使用设备: MPS (Apple GPU)")
    else:
        device = torch.device('cpu')
        print("使用设备: CPU")
    return device

# 定义GraphConvNet类
class GraphConvNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, alpha, device):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.input = nn.Linear(self.input_size, self.hidden_size[0], bias=False).to(self.device)
        self.alpha = alpha
        self.hiddens = nn.ModuleList([gnn.SAGEConv(self.hidden_size[h], self.hidden_size[h + 1]).to(self.device)
                                      for h in range(len(self.hidden_size) - 1)])
        self.output = nn.Linear(hidden_size[-1], output_size).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        edge_index = self.create_edge_index(x)
        x = self.input(x)
        x = self.relu(x)
        for hidden in self.hiddens:
            x = hidden(x, edge_index)
            x = self.relu(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x

    def create_edge_index(self, x):
        similarity_matrix = torch.abs(F.cosine_similarity(x[..., None, :, :], x[..., :, None, :], dim=-1))
        similarity = torch.sort(similarity_matrix.view(-1))[0]
        eps = torch.quantile(similarity, self.alpha, interpolation='nearest')
        adj_matrix = similarity_matrix >= eps
        row, col = torch.where(adj_matrix)
        edge_index = torch.cat((row.reshape(1, -1), col.reshape(1, -1)), dim=0).to(self.device)
        return edge_index

# 定义GRACES类
class GRACES:
    def __init__(self, n_features, hidden_size=None, q=2, n_dropouts=10, dropout_prob=0.5, batch_size=16,
                 learning_rate=0.001, epochs=50, alpha=0.95, sigma=0, f_correct=0, device='cpu'):
        self.device = device
        self.n_features = n_features
        self.q = q
        if hidden_size is None:
            self.hidden_size = [64, 32]
        else:
            self.hidden_size = hidden_size
        self.n_dropouts = n_dropouts
        self.dropout_prob = dropout_prob
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.alpha = alpha
        self.sigma = sigma
        self.f_correct = f_correct
        self.S = None
        self.new = None
        self.model = None
        self.last_model = None
        self.loss_fn = None
        self.f_scores = None

    @staticmethod
    def bias(x):
        if not all(x[:, 0] == 1):
            ones = torch.ones(x.shape[0], 1, dtype=torch.float32, device=x.device)
            x = torch.cat((ones, x.float()), dim=1)
        return x

    def f_test(self, x, y):
        # 转换为numpy数组以适应SelectKBest
        x_np = x.cpu().numpy()
        y_np = y.cpu().numpy()
        slc = SelectKBest(f_classif, k=x_np.shape[1])
        slc.fit(x_np, y_np)
        return torch.tensor(slc.scores_, dtype=torch.float32, device=self.device)

    def xavier_initialization(self):
        if self.last_model is not None:
            weight = torch.zeros(self.hidden_size[0], len(self.S), device=self.device)
            nn.init.xavier_normal_(weight, gain=nn.init.calculate_gain('relu'))
            old_s = self.S.copy()
            if self.new in old_s:
                old_s.remove(self.new)
            for i in self.S:
                if i != self.new:
                    weight[:, self.S.index(i)] = self.last_model.input.weight.data[:, old_s.index(i)]
            self.model.input.weight.data = weight
            for h in range(len(self.hidden_size) - 1):
                self.model.hiddens[h].lin_l.weight.data = self.last_model.hiddens[h].lin_l.weight.data
                self.model.hiddens[h].lin_r.weight.data = self.last_model.hiddens[h].lin_r.weight.data
            self.model.output.weight.data = self.last_model.output.weight.data

    def train(self, x, y):
        input_size = len(self.S)
        output_size = len(torch.unique(y))
        self.model = GraphConvNet(input_size, output_size, self.hidden_size, self.alpha, self.device).to(self.device)

        self.xavier_initialization()
        x = x[:, self.S].to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        train_set = []
        for i in range(x.shape[0]):
            train_set.append([x[i, :], y[i]])
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        for e in range(self.epochs):
            for data, label in train_loader:
                input_0 = data.view(data.shape[0], -1).to(self.device)
                label = label.to(self.device)
                optimizer.zero_grad()
                output = self.model(input_0.float())
                loss = self.loss_fn(output, label)
                loss.backward()
                optimizer.step()
        self.last_model = copy.deepcopy(self.model)

    def dropout(self):
        model_dp = copy.deepcopy(self.model).to(self.device)
        for h in range(len(self.hidden_size) - 1):
            h_size = self.hidden_size[h]
            dropout_index = np.random.choice(range(h_size), int(h_size * self.dropout_prob), replace=False)
            model_dp.hiddens[h].lin_l.weight.data[:, dropout_index] = torch.zeros(
                model_dp.hiddens[h].lin_l.weight[:, dropout_index].shape, device=self.device)
            model_dp.hiddens[h].lin_r.weight.data[:, dropout_index] = torch.zeros(
                model_dp.hiddens[h].lin_r.weight[:, dropout_index].shape, device=self.device)
        dropout_index = np.random.choice(range(self.hidden_size[-1]), int(self.hidden_size[-1] * self.dropout_prob), replace=False)
        model_dp.output.weight.data[:, dropout_index] = torch.zeros(
            model_dp.output.weight[:, dropout_index].shape, device=self.device)
        return model_dp

    def gradient(self, x, y, model):
        model_gr = GraphConvNet(x.shape[1], len(torch.unique(y)), self.hidden_size, self.alpha, self.device).to(self.device)
        temp = torch.zeros(model_gr.input.weight.shape, device=self.device)
        temp[:, self.S] = model.input.weight
        model_gr.input.weight.data = temp
        for h in range(len(self.hidden_size) - 1):
            model_gr.hiddens[h].lin_l.weight.data = model.hiddens[h].lin_l.weight + self.sigma * torch.randn(
                model.hiddens[h].lin_l.weight.shape, device=self.device)
            model_gr.hiddens[h].lin_r.weight.data = model.hiddens[h].lin_r.weight + self.sigma * torch.randn(
                model.hiddens[h].lin_r.weight.shape, device=self.device)
        model_gr.output.weight.data = model.output.weight.data
        output_gr = model_gr(x.float())
        loss_gr = self.loss_fn(output_gr, y)
        loss_gr.backward()
        input_gradient = model_gr.input.weight.grad
        return input_gradient

    def average(self, x, y, n_average):
        grad_cache = None
        for num in range(n_average):
            model = self.dropout()
            input_grad = self.gradient(x, y, model)
            if grad_cache is None:
                grad_cache = input_grad
            else:
                grad_cache += input_grad
        return grad_cache / n_average

    def find(self, input_gradient):
        gradient_norm = input_gradient.norm(p=self.q, dim=0)
        gradient_norm = gradient_norm / gradient_norm.norm(p=2)
        gradient_norm[1:] = (1 - self.f_correct) * gradient_norm[1:] + self.f_correct * self.f_scores
        gradient_norm[self.S] = 0
        max_index = torch.argmax(gradient_norm)
        return max_index.item()

    def select(self, x, y):
        x = torch.tensor(x, dtype=torch.float32, device=self.device)  # 指定dtype为float32
        y = torch.tensor(y, dtype=torch.long, device=self.device)    # 指定dtype为long
        self.f_scores = self.f_test(x, y)
        self.f_scores[torch.isnan(self.f_scores)] = 0
        self.f_scores = self.f_scores / self.f_scores.norm(p=2)
        x = self.bias(x)
        self.S = [0]
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)
        while len(self.S) < self.n_features + 1:
            self.train(x, y)
            input_gradient = self.average(x, y, self.n_dropouts)
            self.new = self.find(input_gradient)
            self.S.append(self.new)
        selection = self.S
        selection.remove(0)
        selection = [s - 1 for s in selection]
        return selection

# 主函数
def main(name, n_features, n_iters, n_repeats, device):
    np.random.seed(0)
    data = scipy.io.loadmat('data/' + name + '.mat')
    x = data['X'].astype(float).astype(np.float32)  # 确保特征为float32
    if name == 'colon' or name == 'leukemia':
        y = np.int64(data['Y'])
        y[y == -1] = 0
    else:
        y = np.int64(data['Y']) - 1
    y = y.reshape(-1)

    auc_test = np.zeros(n_iters)
    acc_test = np.zeros(n_iters)
    sen_test = np.zeros(n_iters)
    spe_test = np.zeros(n_iters)
    f1_test = np.zeros(n_iters)
    pre_test = np.zeros(n_iters)

    seeds = np.random.choice(range(100), n_iters, replace=False)

    for iter in tqdm(range(n_iters)):
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, train_size=0.8, random_state=seeds[iter], stratify=y)
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, train_size=0.5, random_state=seeds[iter], stratify=y_train)

        auc_grid = np.zeros((len(dropout_prob), len(f_correct)))
        loss_grid = np.zeros((len(dropout_prob), len(f_correct)))

        for i in range(len(dropout_prob)):
            for j in range(len(f_correct)):
                for r in range(n_repeats):
                    slc_g = GRACES(
                        n_features=n_features,
                        dropout_prob=dropout_prob[i],
                        f_correct=f_correct[j],
                        device=device
                    )
                    selection_g = slc_g.select(x_train, y_train)
                    x_train_red_g = x_train[:, selection_g]
                    x_val_red = x_val[:, selection_g]

                    clf_g = svm.SVC(probability=True)
                    clf_g.fit(x_train_red_g, y_train)

                    y_val_pred = clf_g.predict_proba(x_val_red)
                    auc_grid[i, j] += roc_auc_score(y_val, y_val_pred[:, 1])
                    loss_grid[i, j] += -np.sum(y_val * np.log(y_val_pred[:, 1] + 1e-15))  # 防止log(0)

        index_i, index_j = np.where(auc_grid == np.max(auc_grid))
        best_index = np.argmin(loss_grid[index_i, index_j])
        best_prob, best_f_correct = dropout_prob[int(index_i[best_index])], f_correct[int(index_j[best_index])]

        for r in range(1):
            slc = GRACES(
                n_features=n_features,
                dropout_prob=best_prob,
                f_correct=best_f_correct,
                device=device
            )
            selection = slc.select(x_train, y_train)
            x_train_red = x_train[:, selection]
            x_test_red = x_test[:, selection]

            clf = svm.SVC(probability=True)
            clf.fit(x_train_red, y_train)

            y_test_pred_proba = clf.predict_proba(x_test_red)[:, 1]
            y_test_pred_labels = (y_test_pred_proba >= 0.5).astype(int)

            # Compute confusion matrix
            from sklearn.metrics import confusion_matrix
            tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred_labels).ravel()

            print('fn', fn)
            print('tp', tp)
            print('tn', tn)
            print('fp', fp)

            acc = (tp + tn) / (tp + tn + fp + fn)
            sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            pre = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = 2 * (pre * sen) / (pre + sen) if (pre + sen) > 0 else 0.0

            print('acc1', acc)
            print('灵敏度TPR', sen)
            print('特异度', spe)
            print('精确率', pre)
            print('f1值', f1)

            acc_test[iter] += acc
            f1_test[iter] += f1
            sen_test[iter] += sen
            spe_test[iter] += spe
            pre_test[iter] += pre

            auc_test[iter] += roc_auc_score(y_test, y_test_pred_proba)

    return [auc_test, acc_test, f1_test, sen_test, spe_test, pre_test]

if __name__ == "__main__":
    # 定义日志文件夹名称
    logs_dir = "logs"

    # 如果日志文件夹不存在，则创建它
    os.makedirs(logs_dir, exist_ok=True)

    # 设置日志文件的名称，包含当前日期和时间
    log_filename = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = os.path.join(logs_dir, log_filename)

    # 配置日志设置
    logging.basicConfig(
        level=logging.INFO,  # 设置日志级别
        format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
        handlers=[
            logging.FileHandler(log_filepath),  # 将日志写入文件
            logging.StreamHandler()  # 同时输出到控制台（可选）
        ]
    )

    # 实验参数
    name = 'colon'
    max_features = 10  # 根据需要调整
    n_iters = 3
    n_repeats = 3

    # 定义超参数的搜索范围
    dropout_prob = [0.1, 0.5, 0.75]
    f_correct = [0, 0.1, 0.5, 0.9]

    # 获取设备
    device = get_device(preferred='cuda')  # 可选 'cuda', 'mps', 或 'cpu'

    # 初始化结果数组
    auc = np.zeros((max_features, n_iters))
    acc = np.zeros((max_features, n_iters))
    f1 = np.zeros((max_features, n_iters))
    sen = np.zeros((max_features, n_iters))
    spe = np.zeros((max_features, n_iters))
    pre = np.zeros((max_features, n_iters))

    # 记录实验的基本信息
    logging.info(f"开始实验，数据集名称: {name}, n_iters: {n_iters}, n_repeats: {n_repeats}")
    logging.info(f"超参数搜索范围 - dropout_prob: {dropout_prob}, f_correct: {f_correct}")
    logging.info(f"特征选择的最大数量: {max_features}")

    for p in range(max_features):
        n_features = p + 1  # 当前选择的特征数量
        logging.info(f"开始特征数量: {n_features} 的实验")
        results = main(name=name, n_features=n_features, n_iters=n_iters, n_repeats=n_repeats, device=device)
        auc[p, :], acc[p, :], f1[p, :], sen[p, :], spe[p, :], pre[p, :] = results
        logging.info(f"完成特征数量: {n_features} 的实验")

    # 计算平均值
    avg_auc = np.mean(auc, axis=1)
    avg_acc = np.mean(acc, axis=1)
    avg_f1 = np.mean(f1, axis=1)
    avg_sen = np.mean(sen, axis=1)
    avg_spe = np.mean(spe, axis=1)
    avg_pre = np.mean(pre, axis=1)

    # 打印结果到控制台
    for p in range(max_features):
        n_features = p + 1
        print(f'Number of Features: {n_features}')
        print(f'  Average Test AUC: {avg_auc[p]:.4f}')
        print(f'  Average Test Accuracy: {avg_acc[p]:.4f}')
        print(f'  Average Test F1: {avg_f1[p]:.4f}')
        print(f'  Average Test TPR/Sen: {avg_sen[p]:.4f}')
        print(f'  Average Test Spe: {avg_spe[p]:.4f}')
        print(f'  Average Test Pre: {avg_pre[p]:.4f}')
        print('---------------------------------------')

    # 记录结果到日志文件
    logging.info('==================== 实验结果 ====================')
    for p in range(max_features):
        n_features = p + 1
        logging.info(f"特征数量: {n_features}")
        logging.info(f"  平均测试 AUC: {avg_auc[p]:.4f}")
        logging.info(f"  平均测试 Accuracy: {avg_acc[p]:.4f}")
        logging.info(f"  平均测试 F1: {avg_f1[p]:.4f}")
        logging.info(f"  平均测试 TPR/sen: {avg_sen[p]:.4f}")
        logging.info(f"  平均测试 spe: {avg_spe[p]:.4f}")
        logging.info(f"  平均测试 pre: {avg_pre[p]:.4f}")
    logging.info('==================================================')
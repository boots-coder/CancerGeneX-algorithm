from torch.utils.data import Dataset

def get_data_template(name):
    if name == "MyDataset":
        return MyDataset

class MyDataset(Dataset):

    def __init__(self, Xs, y):
        self.Xs = Xs
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        # 返回数据和对应的目标
        return self.Xs[index], self.y[index]

    def obtain_instance(self, kwargs):
        Xs, y = kwargs["Xs"], kwargs["y"]
        return MyDataset(Xs=Xs, y=y)
import torch
from torch.utils.data import DataLoader

from Classification.Common.ClassifierTemplate import ClassifierTemplate

def get_dl_trainer(name, kwargs):
    if name.startswith("TrainerWrapper"):
        return Trainer1(name, kwargs)
    elif name.startswith("Trainer2"):
        return Trainer2(name, kwargs)

    else:
        raise ""

class Trainer1(ClassifierTemplate):

    def __init__(self, name, configs):
        super(Trainer1, self).__init__(name, configs)

        self.max_epoch = configs.get("MaxEpoch", 20)
        self.batch_size = configs.get("BatchSize", 32)
        self.early_stop = configs.get("EarlyStop", True)
        self.early_stop_num = configs.get("EarlyStopNum", 3)
        self.need_shuffle = configs.get("NeedShuffle", True)
        self.customized_train_methods = configs.get("CustomizedTrainMethods", False)
        self.cuda = torch.cuda.is_available()

        self.model = None
        self.optimizer = None
        self.criterion = None

    def set_model(self, model):
        self.model = model

    def obtain_model(self):
        return self.model

    def set_optim(self, optimizer):
        self.optimizer = optimizer

    def obtain_optim(self):
        return self.optimizer

    def set_loss_fun(self, criterion):
        self.criterion = criterion

    def obtain_loss_fun(self):
        return self.criterion

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def obtain_batch_size(self):
        if self.batch_size is None:
            raise "使用小批量进行训练的话, 需要配置批量大小"
        return self.batch_size

    def set_need_shuffle(self, need_shuffle):
        self.need_shuffle = need_shuffle

    def obtain_need_shuffle(self):
        return self.need_shuffle

    def fit(self, X_train, y_train, X_val, y_val):
        # 如果你自定义了训练方法, 则优先考虑你的执行方法 todo 这里不太明白， 什么是自定义训练方法？？什么情况下
        if self.customized_train_methods:
            self._execute_customized_train_methods(X_train, y_train, X_val, y_val)

        # 这些是自定义的执行方法 todo 这里的batch_size 是什么？
        if self.batch_size == None:
            if self.early_stop:
                self._fit_early_stop_without_batch(X_train, y_train, X_val, y_val)
            else:
                self._fit_without_batch(X_train, y_train)
        else:
            if self.early_stop:
                self._fit_early_stop_with_batch(X_train, y_train, X_val, y_val)
            else:
                self._fit_with_batch(X_train, y_train)

    def _execute_customized_train_methods(self, X_train, y_train, X_val, y_val):
        pass

    def _fit_without_batch(self, Xs_train, y_train):
        for epoch in range(self.max_epoch+1):

            if self.need_shuffle:
                Xs_train, y_train = self.obtain_shuffle_data(Xs_train, y_train)

            self.optimizer.zero_grad()
            outputs = self.obtain_ouputs_without_batch(Xs_train)
            loss = self.criterion(outputs, y_train)
            loss.backward()
            self.optimizer.step()

    def obtain_ouputs_without_batch(self, Xs_train):
        if isinstance(Xs_train, torch.Tensor):
            _, outputs = self.model(Xs_train)
        elif isinstance(Xs_train, list):
            _, outputs = self.model(*Xs_train)
        return outputs

    def model2GPU(self):
        with torch.no_grad():
            if self.cuda:
                self.model = self.model.cuda()

    def _fit_with_batch(self, Xs_train, y_train):

        sample_size = y_train.shape[0]
        for epoch in range(self.max_epoch+1):

            if self.need_shuffle:
                Xs_train, y_train = self.obtain_shuffle_data(Xs_train, y_train)

            self.optimizer.zero_grad()
            batch_size = self.obtain_batch_size()

            for start_i in range(0, sample_size, batch_size):
                outputs = self.obtain_outputs_with_batch(Xs_train, start_i, batch_size)
                new_y_train = y_train[start_i: start_i + batch_size]
                loss = self.criterion(outputs, new_y_train)
                loss.backward()
                self.optimizer.step()


    def obtain_shuffle_data(self, Xs, y):
        if Xs.device != y.device:
            y = y.to(Xs.device)

        if not Xs.is_contiguous():
            Xs = Xs.contiguous()

        if torch.isnan(Xs).any() or torch.isinf(Xs).any():
            raise ValueError("Xs contains NaN or Inf, which may cause issues")

        indices = torch.randperm(len(y), device=Xs.device)

        y = y[indices]

        if isinstance(Xs, torch.Tensor):
            Xs = torch.index_select(Xs, 0, indices)
        elif isinstance(Xs, list):
            Xs = [torch.index_select(X_train, 0, indices) for X_train in Xs]
        else:
            raise TypeError("Xs must be a torch.Tensor or a list of tensors")

        return Xs, y
    def obtain_outputs_with_batch(self, Xs,  start_i, batch_size):
        if isinstance(Xs, torch.Tensor):
            new_Xs = Xs[start_i : start_i + batch_size]
            _, outputs = self.model(new_Xs)
        elif isinstance(Xs, list):
            new_Xs = [X[start_i : start_i + batch_size] for X in Xs]
            _, outputs = self.model(*new_Xs)

        return outputs


    def _fit_early_stop_without_batch(self, Xs_train, y_train, Xs_val, y_val):

        if Xs_val is None:
            print("没有设置验证集了, 现在使用训练集作为验证集")
            Xs_train, y_train = Xs_val, y_val

        counter = 0
        best_loss = float('inf')

        for epoch in range(self.max_epoch+1):
            if self.need_shuffle:
                Xs_train, y_train = self.obtain_shuffle_data(Xs_train, y_train)

            self.optimizer.zero_grad()
            train_outputs = self.obtain_ouputs_without_batch(Xs_train)
            loss = self.criterion(train_outputs, y_train)
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():

                val_outputs = self.obtain_ouputs_without_batch(Xs_val)
                val_loss = self.criterion(val_outputs, y_val)

                if val_loss < best_loss:
                    best_loss = val_loss
                    counter = 0
                else:
                    counter = counter + 1

                if counter >= self.early_stop_num:
                    print("在" + str(epoch) + "轮完成早停")
                    break

                if epoch == self.max_epoch:
                    print("达到最大轮数后停止")

    # def _fit_early_stop_with_batch(self, Xs_train, y_train, Xs_val, y_val):
    #     if Xs_val is None:
    #         print("没有设置验证集了, 现在使用训练集作为验证集")
    #         Xs_train, y_train = Xs_val, y_val
    #
    #     train_sample_size = y_train.shape[0]
    #     val_sample_size = y_val.shape[0]
    #
    #     for epoch in range(self.max_epoch+1):
    #         if self.need_shuffle:
    #             Xs_train, y_train = self.obtain_shuffle_data(Xs_train, y_train)
    #
    #         counter = 0
    #         best_loss = float('inf')
    #
    #         self.optimizer.zero_grad()
    #         batch_size = self.obtain_batch_size()
    #
    #         for start_i in range(0, train_sample_size, batch_size):
    #             outputs = self.obtain_outputs_with_batch(Xs_train, start_i, batch_size)
    #             y_train_batch = y_train[start_i: start_i + batch_size]
    #             loss = self.criterion(outputs, y_train_batch)
    #             loss.backward()
    #             self.optimizer.step()
    #
    #         val_loss_sum = 0.0
    #         val_samples = 0
    #
    #         with torch.no_grad():
    #             for start_i in range(0, val_sample_size, batch_size):
    #                 val_outputs = self.obtain_outputs_with_batch(Xs_val, start_i, batch_size)
    #                 y_val_batch = y_val[start_i: start_i + batch_size]
    #                 val_loss = self.criterion(val_outputs, y_val_batch)
    #                 val_loss_sum += val_loss.item() * y_val_batch.size(0)
    #                 val_samples += y_val_batch.size(0)
    #
    #             avg_val_loss = val_loss_sum / val_samples
    #
    #             if avg_val_loss < best_loss:
    #                 best_loss = avg_val_loss
    #                 counter = 0
    #             else:
    #                 counter = counter + 1
    #
    #             if counter >= self.early_stop_num:
    #                 print("在" + str(epoch) + "轮完成早停")
    #                 break
    #
    #             if epoch == self.max_epoch:
    #                 print("达到最大轮数后停止")
    #todo 这里早停对应的是哪一步
    def _fit_early_stop_with_batch(self, Xs_train, y_train, Xs_val, y_val):
        print("开始训练（早停，使用批量）")
        if Xs_val is None:
            print("没有设置验证集，使用训练集作为验证集")
            Xs_val, y_val = Xs_train, y_train

        train_sample_size = y_train.shape[0]
        val_sample_size = y_val.shape[0]
        counter = 0
        best_loss = float('inf')

        for epoch in range(self.max_epoch + 1):


            # print(f"Epoch {epoch + 1}/{self.max_epoch + 1}")
            if self.need_shuffle:
                Xs_train, y_train = self.obtain_shuffle_data(Xs_train, y_train)

            self.optimizer.zero_grad()#清零梯度 防止对当前训练造成影响
            batch_size = self.obtain_batch_size()

            #todo 这一步是在干什么？训练一个什么
            for start_i in range(0, train_sample_size, batch_size):
                # print(f"训练批次起始索引: {start_i}")
                outputs = self.obtain_outputs_with_batch(Xs_train, start_i, batch_size)
                y_train_batch = y_train[start_i: start_i + batch_size].to(self.model.device)
                loss = self.criterion(outputs, y_train_batch)
                loss.backward()
                self.optimizer.step()

            # 验证步骤
            val_loss_sum = 0.0
            val_samples = 0

            with torch.no_grad():
                for start_i in range(0, val_sample_size, batch_size):
                    # print(f"验证批次起始索引: {start_i}")
                    val_outputs = self.obtain_outputs_with_batch(Xs_val, start_i, batch_size)
                    y_val_batch = y_val[start_i: start_i + batch_size].to(self.model.device)
                    val_loss = self.criterion(val_outputs, y_val_batch)
                    val_loss_sum += val_loss.item() * y_val_batch.size(0)
                    val_samples += y_val_batch.size(0)

                avg_val_loss = val_loss_sum / val_samples
                # print(f"验证集平均损失: {avg_val_loss}")

                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    counter = 0
                else:
                    counter += 1

                if counter >= self.early_stop_num:
                    print(f"在第 {epoch + 1} 轮触发早停")
                    break

                if epoch == self.max_epoch:
                    print("达到最大轮数，停止训练")

        print("训练完成")

class Trainer2(Trainer1):

    def __init__(self, name, configs):
        super(Trainer2, self).__init__(name, configs)
        self.data_template_name = configs.get("DataTemplateName", "MyDataset")
        self.data_template = self.obtain_data_template_dispatcher(self.data_template_name)

    def set_data_template(self, data_template):
        self.data_template = data_template

    def obtain_data_template(self):
        return self.data_template

    def obtain_data_template_dispatcher(self, name):
        from Classification.DLClassification.Trainers.MyDataSet import get_data_template
        data_template = get_data_template(name)
        if data_template is None:
            assert "暂时不支持这种数据模板" + self.data_template_name
        return data_template

    def _fit_with_batch(self, X_train, y_train):

        batch_size = self.obtain_batch_size()
        train_loader = self.obtain_data_loader(X_train, y_train, batch_size, shuffle=True)

        for epoch in range(self.max_epoch):
            self.optimizer.zero_grad()

            for X_batch, y_batch in train_loader:
                train_outputs = self.obtain_ouputs_without_batch(X_batch)
                loss = self.criterion(train_outputs, y_batch)
                loss.backward()
                self.optimizer.step()

    def _fit_early_stop_with_batch(self, X_train, y_train, X_val, y_val):

        if X_val is None:
            X_train, y_train = X_val, y_val

        counter = 0
        best_loss = float('inf')

        batch_size = self.obtain_batch_size()

        train_loader = self.obtain_data_loader(X_train, y_train, batch_size, shuffle=True)
        val_loader = self.obtain_data_loader(X_val, y_val, batch_size, shuffle=False)

        for epoch in range(self.max_epoch):
            self.optimizer.zero_grad()

            for X_train_batch, y_train_batch in train_loader:
                train_outputs = self.obtain_ouputs_without_batch(X_train_batch)
                loss = self.criterion(train_outputs, y_train_batch)
                loss.backward()
                self.optimizer.step()

            val_loss_sum = 0.0
            val_samples = 0

            with torch.no_grad():
                for X_val_batch, y_val_batch in val_loader:
                    val_outputs = self.obtain_ouputs_without_batch(X_val_batch)
                    val_loss = self.criterion(val_outputs, y_val_batch)
                    val_loss_sum += val_loss.item() * y_val_batch.size(0)
                    val_samples += y_val_batch.size(0)

                avg_val_loss = val_loss_sum / val_samples

                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    counter = 0
                else:
                    counter = counter + 1

                if counter >= self.early_stop_num:
                    print("在" + str(epoch) + "轮完成早停")
                    break

                if epoch == self.max_epoch:
                    print("达到最大轮数后停止")

    def obtain_data_loader(self, Xs, y, batch_size, shuffle):
        kwargs = dict(Xs=Xs, y=y)
        dataset = self.data_template.obtain_instance(kwargs)
        collate_fn = self.obtain_collate_fn_method(Xs)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
        return loader

    def obtain_collate_fn_method(self, Xs):
        if isinstance(Xs, torch.Tensor):
            collate_fn = None
        elif isinstance(Xs, list):
            def collate_fn(batch):

                data = [item[0] for item in batch]
                data = [torch.stack([sublist[i] for sublist in data]) for i in range(len(data[0]))]
                labels = torch.stack([item[1] for item in batch])

                return data, labels
        return collate_fn

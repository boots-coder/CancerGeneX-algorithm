import torch.nn as nn
import torch.optim as optim
import torch
import copy

from Classification.DLClassification.Models.BaseModel import get_dl_model
from Classification.DLClassification.Trainers.BaseTrainer import get_dl_trainer
from Classification.MLClassification.BaseClassify import get_ml_base_classifier
from Classification.MLClassification.EnsembleClassify import get_ens_classifier


def  get_global_classifier_buider(name, builder_type, data_type, config):
    if builder_type == "ML":
        return MLGlobalClassifierBuider(name, builder_type, data_type, config)
    elif builder_type == "DL":
        return DLGlobalClassifierBuider(name, builder_type, data_type, config)
    else:
        raise "暂时不支持" + name + "特征提取器/分类器的构建器"

def  get_local_classifier_buider(name, builder_type, data_type, config):
    if builder_type == "ML":
        return MLLocalClassifierBuider(name, builder_type, data_type, config)
    elif builder_type == "DL":
        return DLLocalClassifierBuider(name, builder_type, data_type, config)
    else:
        raise "暂时不支持" + name + "特征提取器/分类器的构建器"

class BuilderBuiderWrapper():

    def __init__(self, name, builder_type, data_type):
        self.name = name
        self.builder_type = builder_type
        self.data_type = data_type

    def obtain_fit_classifier(self, X_train, y_train, X_val, y_val, layer=None):
       pass

    def fit_executable(self, layer):
        return True

    def update_config(self, new_cfig, layer):
        pass

    def obtain_name(self):
        return self.name

    def obtain_builder_type(self):
        return self.builder_type

    def obtain_data_type(self):
        return self.data_type


class MLClassifierBuider(BuilderBuiderWrapper):

    def __init__(self, name, builder_type, data_type, config):
        super(MLClassifierBuider, self).__init__(name, builder_type, data_type)
        self.config = copy.deepcopy(config)
        self.config["DataType"] = data_type
        self.debug = config.get("DeBug", True)

    def update_config(self, new_cfig, layer):
        pass

    def obtain_classifier(self, name, config, layer=None):
        est = get_ens_classifier(name, config, layer, default=True) \
              or get_ml_base_classifier(name, config, layer, default=True)
        if est == None:
            raise "暂时不支持" + name + "分类器/特征提取器"
        return est

class MLGlobalClassifierBuider(MLClassifierBuider):

    def obtain_fit_classifier(self, X_train, y_train, X_val, y_val, dim, layer=None):
        #这一步分别训练每一个弱分类器，但是没有进行筛选
        est = self.obtain_classifier(self.name, self.config, layer)
        est.fit(X_train, y_train, X_val, y_val)
        name = est.obtain_name()

        return name, est

class MLLocalClassifierBuider(MLClassifierBuider):

    def obtain_fit_classifier(self, Xs_train, y_train, Xs_val, y_val, layer=None):

        ests = {}

        est_temple = self.obtain_classifier(self.name, self.config, layer)
        name = "Multi" + est_temple.obtain_name()

        for index, (X_train, X_val) in enumerate(zip(Xs_train, Xs_val)):
            if self.debug:
                print("第" + str(index) + "个局部分类器的相关配置:")
            est = copy.deepcopy(est_temple)
            est.fit(X_train, y_train, X_val, y_val)
            ests[index] = est

        return name, ests

class DLClassifierBuider(BuilderBuiderWrapper):

    def __init__(self, name, builder_type, data_type, configs):
        super(DLClassifierBuider, self).__init__(name, builder_type, data_type)

        configs = copy.deepcopy(configs)

        self.cuda = torch.cuda.is_available()
        self.trainer_cfig = configs.get("Trainer", None)
        assert self.trainer_cfig != None, "使用深度学习方法时，必须设置训练器"

        self.model_cfig = configs.get("Model", None)
        assert self.model_cfig != None, "使用深度学习方法时，必须设置模型"
        self.model_cfig["Builder"] = configs.get("Builder", None)
        self.model_cfig["Layers"] = configs.get("Layers", None)
        self.model_cfig["DataType"] = data_type

        self.loss_fun_cfig = configs.get("LossFun", None)
        assert self.loss_fun_cfig != None, "使用深度学习方法时，必须设置损失函数"

        self.optimizer_cfig = configs.get("Optimizer", None)
        assert self.optimizer_cfig != None, "使用深度学习方法时，必须设置优化器"

        self.debug = configs.get("DeBug", True)

    # 没做是否有cuda环境的判断……这里导致了问题 ？
    def move_model_to_cuda(self, model):
        with torch.no_grad():
            if self.cuda:
                return model.cuda()

    def move_model_to_device(self, model):
        with torch.no_grad():
            if torch.cuda.is_available():
                print("CUDA is available. Moving model to GPU.")
                return model.cuda()
            # elif torch.backends.mps.is_available():  # For macOS with MPS support
            #     print("MPS is available. Moving model to MPS.")
            #     model =  model.to(torch.device("mps"))
            #     print("转移到mps上面的模型: ", model)  # 使用逗号分隔，避免字符串拼接报错
            #     return model
            else:
                print("Neither CUDA nor MPS is available. Using CPU.")
                return model.to(torch.device("cpu"))

    # def move_model_to_cuda(self, model):
    #     with torch.no_grad():
    #         if self.cuda:
    #             print("CUDA is available. Moving model to GPU.")
    #             return model.cuda()
    #         else:
    #             print("CUDA is not available. Using CPU.")
    #             return model
    def obtain_new_data(self, X_train, y_train, X_val, y_val):
        new_X_train = self.convert_X_to_tensor(X_train)
        new_y_train = self.convert_y_to_tensor(y_train)
        new_X_val = self.convert_X_to_tensor(X_val)
        new_y_val = self.convert_y_to_tensor(y_val)
        return new_X_train, new_y_train, new_X_val, new_y_val

    def convert_X_to_tensor(self, X):
        X = torch.tensor(X).float()
        if self.cuda:
            X = X.cuda()
        return X

    def convert_y_to_tensor(self, y):
        y = torch.tensor(y).long()
        if self.cuda:
            y = y.cuda()
        return y

    def obtain_trainer(self, trainer_cfig, layer):
        name = trainer_cfig.get("name", None)
        assert name != None, "训练器的名字不能设置为空"
        trainer = get_dl_trainer(name, trainer_cfig)
        if trainer == None:
            raise "暂时不支持" + name + "分类器/特征提取器"
        return trainer

    def obtain_dl_network(self, model_cfig, layer):
        name = model_cfig.get("name", None)
        assert name != None, "特征提取器的名字不能设置为空"
        est = get_dl_model(name, model_cfig)
        if est == None:
            raise "暂时不支持" + name + "分类器/特征提取器"
        return est

    def obtain_loss_fun(self, loss_fun_cfig, name):
        loss_fun_name = loss_fun_cfig.get("name", None)
        assert loss_fun_name != None, "损失函数的名字不能设置为空"

        if loss_fun_name == "CrossEntropyLoss":
            return nn.CrossEntropyLoss()
        elif loss_fun_name == "L1Loss":
            return nn.L1Loss()
        elif loss_fun_name == "KLDivLoss":
            return nn.KLDivLoss()
        elif loss_fun_name == "MultiLabelMarginLoss":
            return nn.MultiMarginLoss()

    def obtain_optimizer(self, optimizer_cfig, layer):
        optim_name = optimizer_cfig.get("name", None)
        assert optim_name != None, "优化器的名字不能设置为空"
        parameters = optimizer_cfig.get("parameters")

        if optim_name == "Adam":
            return optim.Adam(parameters)
        elif optim_name == "SGD":
            return optim.SGD(parameters)
        elif optim_name == "RMSprop":
            return optim.RMSprop(parameters)

    def update_config(self, new_cfig, layer):
        trainer_cfig = new_cfig.get("Trainer", None)
        if trainer_cfig != None :
            self.update_trainer_config(trainer_cfig, layer)

        model_cfig = new_cfig.get("Model", None)
        if model_cfig != None:
            self.update_model_config(model_cfig, layer)

        fun_loss_cfig = new_cfig.get("LossFun", None)
        if fun_loss_cfig != None:
            self.update_fun_loss_config(fun_loss_cfig, layer)

        optimizer_cfig = new_cfig.get("Optimizer", None)
        if optimizer_cfig != None:
            self.update_optimizer_config(optimizer_cfig, layer)

    def update_trainer_config(self,new_cfig, layer):
        pass

    def update_model_config(self, new_cfig, layer):
        self.model_cfig["Parameter"].update({"InputSize": new_cfig["Parameter"]["InputSize"]})

    def update_fun_loss_config(self, new_cfig, layerr):
        pass

    def update_optimizer_config(self, new_cfig, layer):
        pass

class DLGlobalClassifierBuider(DLClassifierBuider):
    def obtain_fit_classifier(self, X_train, y_train, X_val, y_val, dims, layer=None):

        trainer_cfig = self.trainer_cfig
        trainer = self.obtain_trainer(trainer_cfig, layer)

        loss_fun_cfig = self.loss_fun_cfig
        loss_fun = self.obtain_loss_fun(loss_fun_cfig, layer)
        trainer.set_loss_fun(loss_fun)

        model_cfig = self.model_cfig
        model = self.obtain_dl_network(model_cfig, layer)

        # model = self.move_model_to_cuda(model)
        model = self.move_model_to_device(model)
        trainer.set_model(model)

        optimizer_cfig = self.optimizer_cfig
        optimizer_cfig["parameters"] = model.parameters()
        optimizer = self.obtain_optimizer(optimizer_cfig, layer)
        trainer.set_optim(optimizer)

        new_X_train, new_y_train, new_X_val, new_y_val = self.obtain_new_data(X_train, y_train, X_val, y_val)
        trainer.fit(new_X_train, new_y_train, new_X_val, new_y_val)

        name = model.obtain_name()

        if self.debug:
            print("模型的相关配置:")
            print("训练器的配置信息", trainer_cfig)
            print("损失函数的配置信息", loss_fun_cfig)
            print("模型的配置信息", model_cfig)
            print("优化器的配置信息", optimizer_cfig)

        return name, trainer.obtain_model()

class DLLocalClassifierBuider(DLClassifierBuider):

    def obtain_fit_classifier(self, Xs_train, y_train, Xs_val, y_val, dims, layer=None):

        ests = {}

        name = "Multi" + self.model_cfig.get("name")

        for index, (X_train, X_val) in enumerate(zip(Xs_train, Xs_val)):

            trainer_cfig = self.trainer_cfig
            trainer = self.obtain_trainer(trainer_cfig, layer)

            loss_fun_cfig = self.loss_fun_cfig
            loss_fun = self.obtain_loss_fun(loss_fun_cfig, layer)
            trainer.set_loss_fun(loss_fun)

            model_cfig = copy.deepcopy(self.model_cfig)
            model_cfig["Parameter"]["InputSize"] = model_cfig["Parameter"]["InputSize"][index]
            model = self.obtain_dl_network(model_cfig, layer)
            model = self.move_model_to_cuda(model)
            trainer.set_model(model)

            optimizer_cfig = self.optimizer_cfig
            optimizer_cfig["parameters"] = model.parameters()
            optimizer = self.obtain_optimizer(optimizer_cfig, layer)
            trainer.set_optim(optimizer)

            new_X_train, new_y_train, new_X_val, new_y_val = self.obtain_new_data(X_train, y_train, X_val, y_val)
            trainer.fit(new_X_train, new_y_train, new_X_val, new_y_val)

            ests[index] = model

            if self.debug:
                print("第" + str(index) + "个局部分类器的相关配置:")
                print("训练器的配置信息", trainer_cfig)
                print("损失函数的配置信息", loss_fun_cfig)
                print("模型的配置信息", model_cfig)
                print("优化器的配置信息", optimizer_cfig)

        return name, ests

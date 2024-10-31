import numpy as np

from Processor.FeatureSelector.Selector.SelectorWrapper import SelectorWrapper

# 获取基础选择器
def get_base_selector(name, config):
    method_name = config.get("Method", None)  # 从配置中获取方法名
    if method_name == "GCLasso":
        return GCLasso(name, config)  # 如果方法是"GCLasso"，返回GCLasso选择器实例
    elif method_name == "GCFClassif":
        return GCFClassif(name, config)  # 如果方法是"GCFClassif"，返回GCFClassif选择器实例
    else:
        raise ValueError("Invalid method name")  # 如果方法名不匹配，抛出异常


class GCLasso(SelectorWrapper):
    def __init__(self, name, kwargs):
        from sklearn.linear_model import Lasso  # 引入Lasso模型
        self.coef = kwargs.get("coef", 0.0001) if kwargs != None else 0.0001  # 获取系数阈值，默认为0.0001
        kwargs = {
            "alpha" : 0.0001, "copy_X" : True, "fit_intercept" : True, "max_iter" : 10000,
            "normalize" : True, "positive" : False, "precompute" : False, "random_state" : None,
            "selection" : 'cyclic', "tol" : 0.0001,  "warm_start" : False
        }  # 设置Lasso模型参数
        super(GCLasso, self).__init__(name, Lasso, kwargs)  # 调用父类构造函数

    def _obtain_selected_index(self, X_train, y_train):
        select_idxs = []  # 选择的特征索引列表
        select_infos = {}  # 选择的特征信息字典


        for ind, coef_ in enumerate(self.est.coef_):  # 遍历模型系数
            if np.abs(coef_) > self.coef:  # 如果系数的绝对值大于阈值
                select_idxs.append(ind)  # 添加索引到选择列表
                select_infos[ind] = coef_  # 添加系数到信息字典
        select_infos["Num"] = len(select_idxs)  # 记录选择特征的数量
        select_infos["Name"] = self.name  # 记录选择器名称
        return select_idxs, select_infos  # 返回选择特征的索引和信息

    def obtain_all_index(self, X=None):
        select_infos = {"inds": [], "metrics": []}  # 选择特征的索引和系数字典
        for ind, coef_ in enumerate(self.est.coef_):  # 遍历模型系数
            select_infos["inds"].append(ind)  # 添加索引到列表
            select_infos["metrics"].append(coef_)  # 添加系数到列表
        return select_infos  # 返回所有特征的索引和系数


class GCFClassif(SelectorWrapper):

    def __init__(self, name, kwargs):
        from sklearn.feature_selection import SelectKBest, f_classif  # 引入特征选择和F检验方法
        super(GCLasso, self).__init__(name, None, None)  # 调用父类构造函数
        self.P = kwargs.get("P", 0.5) if kwargs != None else 0.5  # 获取P值阈值，默认为0.5
        self.model = SelectKBest(f_classif, k=50)  # 初始化SelectKBest模型，选择前50个特征

    def _obtain_selected_index(self, X_train, y_train):
        select_infos = {"inds": [], "metrics": []}  # 选择特征的索引和P值字典
        for ind, p_ in enumerate(self.est.pvalues_):  # 遍历P值
            if p_ < self.P:  # 如果P值小于阈值
                select_infos["inds"].append(ind)  # 添加索引到列表
                select_infos["metrics"].append(p_)  # 添加P值到列表
        select_infos["Num"] = len(select_infos["inds"])  # 记录选择特征的数量
        return select_infos  # 返回选择特征的索引和P值

    def obtain_all_index(self, X=None):
        select_infos = {"inds": [], "metrics": []}  # 选择特征的索引和P值字典
        for ind, p_ in enumerate(self.est.pvalues_):  # 遍历P值
            select_infos["inds"].append(ind)  # 添加索引到列表
            select_infos["metrics"].append(p_)  # 添加P值到列表
        return select_infos  # 返回所有特征的索引和P值
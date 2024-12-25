import numpy as np

from Processor.FeatureSelector.Selector.SelectorWrapper import SelectorWrapper

# 获取基础选择器
def get_base_selector(name, config):
    method_name = config.get("Method", None)  # 从配置中获取方法名
    if method_name == "GCLasso":
        return GCLasso(name, config)  # 如果方法是"GCLasso"，返回GCLasso选择器实例
    elif method_name == "GCFClassif":
        return GCFClassif(name, config)  # 如果方法是"GCFClassif"，返回GCFClassif选择器实例
    elif method_name == "GCVariance":
        return GCVariance(name, config)  # 如果方法是"GCVariance"，返回GCVariance选择器实例
    elif method_name == "GCMutualInfo":
        return GCMutualInfo(name, config)  # 如果方法是"GCMutualInfo"，返回GCMutualInfo选择器实例
    elif method_name == "GCChiSquare":
        return GCChiSquare(name, config)  # 如果方法是"GCChiSquare"，返回GCChiSquare选择器实例
    elif method_name == "GCCorrelation":
        return GCCorrelation(name, config)  # 如果方法是"GCCorrelation"，返回GCCorrelation选择器实例
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

"""
GCFClassif 是一个特征选择算法，
它基于统计学中的 F 检验 (ANOVA F-value)
"""
class GCFClassif(SelectorWrapper):
    def __init__(self, name, kwargs):
        from sklearn.feature_selection import SelectKBest, f_classif  # 引入特征选择和F检验方法
        self.P = kwargs.get("P", 0.5) if kwargs != None else 0.5  # 获取P值阈值，默认为0.5
        kwargs = {
            "score_func": f_classif,  # 设置评分函数为f_classif
            "k": 'all'  # 选择所有特征以便后续根据P值筛选
        }
        super(GCFClassif, self).__init__(name, SelectKBest, kwargs)  # 修正: 正确调用父类构造函数

    def _obtain_selected_index(self, X_train, y_train):
        select_idxs = []  # 选择的特征索引列表
        select_infos = {}  # 选择的特征信息字典

        pvalues = self.est.pvalues_  # 获取F检验的P值
        for ind, p_ in enumerate(pvalues):  # 遍历P值
            if p_ < self.P:  # 如果P值小于阈值
                select_idxs.append(ind)  # 添加索引到选择列表
                select_infos[ind] = p_  # 添加P值到信息字典

        select_infos["Num"] = len(select_idxs)  # 记录选择特征的数量
        select_infos["Name"] = self.name  # 记录选择器名称
        return select_idxs, select_infos  # 返回选择特征的索引和信息

    def obtain_all_index(self, X=None):
        select_infos = {"inds": [], "metrics": []}  # 选择特征的索引和P值字典
        for ind, p_ in enumerate(self.est.pvalues_):  # 遍历P值
            select_infos["inds"].append(ind)  # 添加索引到列表
            select_infos["metrics"].append(p_)  # 添加P值到列表
        return select_infos  # 返回所有特征的索引和P值



class GCVariance(SelectorWrapper):
    def __init__(self, name, kwargs):
        from sklearn.feature_selection import VarianceThreshold  # 引入方差阈值选择器
        self.threshold = kwargs.get("threshold", 0.01) if kwargs != None else 0.01  # 获取方差阈值，默认为0.01
        kwargs = {
            "threshold": self.threshold  # 设置方差阈值参数
        }
        super(GCVariance, self).__init__(name, VarianceThreshold, kwargs)  # 调用父类构造函数

    def _obtain_selected_index(self, X_train, y_train):
        select_idxs = []  # 选择的特征索引列表
        select_infos = {}  # 选择的特征信息字典

        variances = self.est.variances_  # 获取每个特征的方差
        for ind, var in enumerate(variances):  # 遍历所有特征的方差
            if var > self.threshold:  # 如果方差大于阈值
                select_idxs.append(ind)  # 添加索引到选择列表
                select_infos[ind] = var  # 添加方差值到信息字典

        select_infos["Num"] = len(select_idxs)  # 记录选择特征的数量
        select_infos["Name"] = self.name  # 记录选择器名称
        return select_idxs, select_infos  # 返回选择特征的索引和信息

    def obtain_all_index(self, X=None):
        select_infos = {"inds": [], "metrics": []}  # 选择特征的索引和方差值字典
        for ind, var in enumerate(self.est.variances_):  # 遍历所有特征的方差
            select_infos["inds"].append(ind)  # 添加索引到列表
            select_infos["metrics"].append(var)  # 添加方差值到列表
        return select_infos  # 返回所有特征的索引和方差值


class GCMutualInfo(SelectorWrapper):
    def __init__(self, name, kwargs):
        from sklearn.feature_selection import SelectKBest, mutual_info_classif  # 引入互信息特征选择方法
        self.threshold = kwargs.get("threshold", 0.05) if kwargs != None else 0.05  # 获取互信息阈值，默认为0.05
        self.n_neighbors = kwargs.get("n_neighbors", 3) if kwargs != None else 3  # 获取近邻数，默认为3
        self.random_state = kwargs.get("random_state", 42) if kwargs != None else 42  # 获取随机种子，默认为42

        def mutual_info_wrapper(X, y):
            return mutual_info_classif(X, y, n_neighbors=self.n_neighbors, random_state=self.random_state)

        kwargs = {
            "score_func": mutual_info_wrapper,  # 设置评分函数
            "k": 'all'  # 选择所有特征以便后续根据阈值筛选
        }
        super(GCMutualInfo, self).__init__(name, SelectKBest, kwargs)  # 调用父类构造函数

    def _obtain_selected_index(self, X_train, y_train):
        select_idxs = []  # 选择的特征索引列表
        select_infos = {}  # 选择的特征信息字典

        scores = self.est.scores_  # 获取互信息分数
        for ind, score in enumerate(scores):  # 遍历所有特征的互信息分数
            if score > self.threshold:  # 如果互信息分数大于阈值
                select_idxs.append(ind)  # 添加索引到选择列表
                select_infos[ind] = score  # 添加分数到信息字典

        select_infos["Num"] = len(select_idxs)  # 记录选择特征的数量
        select_infos["Name"] = self.name  # 记录选择器名称
        return select_idxs, select_infos  # 返回选择特征的索引和信息

    def obtain_all_index(self, X=None):
        select_infos = {"inds": [], "metrics": []}  # 选择特征的索引和互信息分数字典
        for ind, score in enumerate(self.est.scores_):  # 遍历所有特征的互信息分数
            select_infos["inds"].append(ind)  # 添加索引到列表
            select_infos["metrics"].append(score)  # 添加互信息分数到列表
        return select_infos  # 返回所有特征的索引和互信息分数


class GCChiSquare(SelectorWrapper):
    """
    基于卡方检验的特征选择
    卡方检验用于评估离散特征与离散目标变量之间的关系
    """

    def __init__(self, name, kwargs):
        from sklearn.feature_selection import SelectKBest, chi2  # 引入卡方检验特征选择方法
        self.P = kwargs.get("P", 0.05) if kwargs != None else 0.05  # 获取P值阈值，默认为0.05

        kwargs = {
            "score_func": chi2,  # 设置评分函数为卡方检验
            "k": 'all'  # 选择所有特征以便后续根据P值筛选
        }
        super(GCChiSquare, self).__init__(name, SelectKBest, kwargs)  # 调用父类构造函数

    def _obtain_selected_index(self, X_train, y_train):
        select_idxs = []  # 选择的特征索引列表
        select_infos = {}  # 选择的特征信息字典

        # 确保所有特征值非负（卡方检验的要求）
        X_train = np.abs(X_train)

        pvalues = self.est.pvalues_  # 获取卡方检验的P值
        for ind, p_ in enumerate(pvalues):  # 遍历P值
            if p_ < self.P:  # 如果P值小于阈值
                select_idxs.append(ind)  # 添加索引到选择列表
                select_infos[ind] = p_  # 添加P值到信息字典

        select_infos["Num"] = len(select_idxs)  # 记录选择特征的数量
        select_infos["Name"] = self.name  # 记录选择器名称
        return select_idxs, select_infos  # 返回选择特征的索引和信息

    def obtain_all_index(self, X=None):
        select_infos = {"inds": [], "metrics": []}  # 选择特征的索引和P值字典
        for ind, p_ in enumerate(self.est.pvalues_):  # 遍历P值
            select_infos["inds"].append(ind)  # 添加索引到列表
            select_infos["metrics"].append(p_)  # 添加P值到列表
        return select_infos  # 返回所有特征的索引和P值


class GCCorrelation(SelectorWrapper):
    """
    基于相关系数的特征选择
    选择与目标变量相关性较强的特征
    """

    def __init__(self, name, kwargs):
        from sklearn.feature_selection import SelectKBest  # 引入特征选择基类
        self.threshold = kwargs.get("threshold", 0.1) if kwargs != None else 0.1  # 获取相关系数阈值，默认为0.1

        def correlation_score(X, y):
            correlations = np.array([np.abs(np.corrcoef(X[:, i], y)[0, 1]) for i in range(X.shape[1])])
            return correlations, np.zeros_like(correlations)  # 返回相关系数和虚拟P值

        kwargs = {
            "score_func": correlation_score,  # 设置评分函数为相关系数计算
            "k": 'all'  # 选择所有特征以便后续根据阈值筛选
        }
        super(GCCorrelation, self).__init__(name, SelectKBest, kwargs)  # 调用父类构造函数

    def _obtain_selected_index(self, X_train, y_train):
        select_idxs = []  # 选择的特征索引列表
        select_infos = {}  # 选择的特征信息字典

        correlations = self.est.scores_  # 获取相关系数
        for ind, corr in enumerate(correlations):  # 遍历相关系数
            if np.abs(corr) > self.threshold:  # 如果相关系数绝对值大于阈值
                select_idxs.append(ind)  # 添加索引到选择列表
                select_infos[ind] = corr  # 添加相关系数到信息字典

        select_infos["Num"] = len(select_idxs)  # 记录选择特征的数量
        select_infos["Name"] = self.name  # 记录选择器名称
        return select_idxs, select_infos  # 返回选择特征的索引和信息

    def obtain_all_index(self, X=None):
        select_infos = {"inds": [], "metrics": []}  # 选择特征的索引和相关系数字典
        for ind, corr in enumerate(self.est.scores_):  # 遍历相关系数
            select_infos["inds"].append(ind)  # 添加索引到列表
            select_infos["metrics"].append(corr)  # 添加相关系数到列表
        return select_infos  # 返回所有特征的索引和相关系数
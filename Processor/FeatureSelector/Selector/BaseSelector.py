import numpy as np

from Processor.FeatureSelector.Selector.SelectorWrapper import SelectorWrapper

# 获取基础选择器
def get_base_selector(name, config):
    method_name = config.get("Method", None)
    if method_name == "GCLasso":
        return GCLasso(name, config)
    elif method_name == "GCFClassif":
        return GCFClassif(name, config)
    elif method_name == "GCVariance":
        return GCVariance(name, config)
    elif method_name == "GCMutualInfo":
        return GCMutualInfo(name, config)
    elif method_name == "GCChiSquare":
        return GCChiSquare(name, config)
    elif method_name == "GCCorrelation":
        return GCCorrelation(name, config)
    # elif method_name == "VotingSelector":
    #     return VotingFeatureSelector(name, config)
    # get_base_selector 内部改一下:
    elif method_name == "VotingSelector":
        # 在这里把对应子配置再取出来
        param_dict = config.get("Parameter", {})
        return VotingFeatureSelector(name, param_dict)
    else:
        raise ValueError("Invalid method name")


class VotingFeatureSelector(SelectorWrapper):
    """
    基于多种特征选择器投票的特征选择方法
    根据网络层级动态调整投票阈值，层级越深，投票阈值越低
    """

    def __init__(self, name, kwargs):
        from sklearn.base import BaseEstimator, TransformerMixin

        # 确保安全获取参数，处理嵌套字典结构
        params = kwargs.get("Parameter", {}) if isinstance(kwargs, dict) else {}

        # 获取配置参数
        self.max_layers = params.get("max_layers", 5)
        self.min_votes_percentage = params.get("min_votes_percentage", 0.3)
        self.selectors_config = params.get("selectors_config", {})
        self.current_layer = params.get("current_layer", 1)

        # 创建一个基本估计器类
        class DummyEstimator(BaseEstimator, TransformerMixin):
            def fit(self, X, y):
                return self

            def transform(self, X):
                return X

            def fit_transform(self, X, y=None):
                return self.fit(X, y).transform(X)

        # 初始化基本估计器
        self.dummy_estimator = DummyEstimator()

        # 初始化父类
        super(VotingFeatureSelector, self).__init__(name, DummyEstimator, {})

        # 必须明确设置 est 属性
        self.est = self.dummy_estimator

        # 初始化所有子选择器
        self.selectors = self._initialize_selectors()

    def _initialize_selectors(self):
        """初始化所有特征选择器"""
        selectors = []

        try:
            # 创建Lasso选择器
            lasso_config = {"Method": "GCLasso", "coef": self.selectors_config.get("lasso_threshold", 0.0001)}
            selectors.append(GCLasso("Lasso", lasso_config))

            # 创建F检验选择器
            f_classif_config = {"Method": "GCFClassif", "P": self.selectors_config.get("f_test_p_value", 0.05)}
            selectors.append(GCFClassif("F-test", f_classif_config))

            # 创建方差选择器
            variance_config = {"Method": "GCVariance",
                               "threshold": self.selectors_config.get("variance_threshold", 0.01)}
            selectors.append(GCVariance("Variance", variance_config))

            # 创建互信息选择器
            mutual_info_config = {"Method": "GCMutualInfo",
                                  "threshold": self.selectors_config.get("mutual_info_threshold", 0.05)}
            selectors.append(GCMutualInfo("MutualInfo", mutual_info_config))

            # 创建相关性选择器
            corr_config = {"Method": "GCCorrelation",
                           "threshold": self.selectors_config.get("correlation_threshold", 0.1)}
            selectors.append(GCCorrelation("Correlation", corr_config))
        except Exception as e:
            print(f"Error initializing selectors: {e}")
            # 确保返回至少一个选择器，避免空列表
            if not selectors:
                try:
                    variance_config = {"Method": "GCVariance", "threshold": 0.01}
                    selectors.append(GCVariance("Variance", variance_config))
                except:
                    pass

        return selectors

    def fit(self, X_train, y_train):
        """训练所有选择器"""
        # 确保 est 属性已设置
        if not hasattr(self, 'est') or self.est is None:
            from sklearn.base import BaseEstimator, TransformerMixin
            class DummyEstimator(BaseEstimator, TransformerMixin):
                def fit(self, X, y): return self

                def transform(self, X): return X

            self.est = DummyEstimator()

        # 拟合基本估计器
        self.est.fit(X_train, y_train)

        # 拟合所有子选择器
        working_selectors = []
        for selector in self.selectors:
            try:
                selector.fit(X_train, y_train)
                working_selectors.append(selector)
            except Exception as e:
                print(f"Warning: Selector {selector.name} failed to fit: {e}")

        # 更新选择器列表，只保留成功拟合的选择器
        self.selectors = working_selectors if working_selectors else self.selectors

        return self

    def transform(self, X):
        """根据投票结果选择特征"""
        selected_indices, _ = self._obtain_selected_index(X, None)
        return X[:, selected_indices] if len(selected_indices) > 0 else X

    def _obtain_selected_index(self, X_train, y_train=None):
        """根据投票机制获取选中的特征索引"""
        n_features = X_train.shape[1]
        vote_counts = np.zeros(n_features)

        # 获取每个选择器选择的特征
        selector_results = {}
        for selector in self.selectors:
            try:
                if hasattr(selector, '_obtain_selected_index'):
                    indices, info = selector._obtain_selected_index(X_train, y_train)
                    selector_results[selector.name] = indices
                    for idx in indices:
                        if 0 <= idx < n_features:  # 确保索引有效
                            vote_counts[idx] += 1
            except Exception as e:
                print(f"Warning: Selector {selector.name} failed in feature selection: {e}")

        # 计算当前层的投票阈值
        max_votes = max(1, len(self.selectors))  # 防止除以零
        layer_ratio = (self.max_layers - self.current_layer + 1) / self.max_layers
        vote_threshold = max(1, int(max_votes * layer_ratio),
                             int(max_votes * self.min_votes_percentage))

        # 选择投票数超过阈值的特征
        select_idxs = []
        select_infos = {}

        for idx, votes in enumerate(vote_counts):
            if votes >= vote_threshold:
                select_idxs.append(idx)
                select_infos[idx] = votes / max_votes  # 归一化的投票得分

        # 如果没有特征被选中，至少选择投票最多的前5%特征
        if len(select_idxs) == 0:
            min_features = max(int(n_features * 0.05), 1)
            top_indices = np.argsort(-vote_counts)[:min_features]
            select_idxs = top_indices.tolist()
            for idx in select_idxs:
                select_infos[idx] = vote_counts[idx] / max_votes

        select_infos["Num"] = len(select_idxs)
        select_infos["Name"] = self.name
        select_infos["Layer"] = self.current_layer
        select_infos["VoteThreshold"] = vote_threshold
        select_infos["SelectorResults"] = selector_results

        return select_idxs, select_infos

    def obtain_all_index(self, X=None):
        """获取所有特征的投票情况"""
        if X is None:
            return {"inds": [], "metrics": []}

        n_features = X.shape[1]
        vote_counts = np.zeros(n_features)

        for selector in self.selectors:
            try:
                if hasattr(selector, 'obtain_all_index'):
                    all_info = selector.obtain_all_index(X)
                    for i, idx in enumerate(all_info["inds"]):
                        if 0 <= idx < n_features:  # 确保索引有效
                            vote_counts[idx] += 1
            except Exception as e:
                print(f"Warning: Selector {selector.name} failed in obtain_all_index: {e}")

        select_infos = {"inds": [], "metrics": []}
        for idx in range(n_features):
            select_infos["inds"].append(idx)
            select_infos["metrics"].append(vote_counts[idx] / max(1, len(self.selectors)))

        return select_infos
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
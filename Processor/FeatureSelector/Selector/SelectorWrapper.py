# from sklearn.preprocessing import StandardScaler
#
# from Processor.Common.Template import FeatureSelectorTemplate
# from sklearn.pipeline import make_pipeline
#
# # 定义SelectorWrapper类，继承自FeatureSelectorTemplate
# class SelectorWrapper(FeatureSelectorTemplate):
#
#     def __init__(self, name, est_class, est_args):
#         self.name = name  # 选择器名称
#         self.est_class = est_class  # 模型类，例如Lasso
#         self.est_args = est_args  # 模型参数
#         self.est = None  # 模型实例
#
#         self.select_inds = None  # 选择的特征索引
#         self.original_num = None  # 原始特征数量
#         self.enforcement = False  # 是否强制重新选择特征
#
#         self.need_info_saved = False  # 是否需要保存信息
#
#     # 拟合模型
#     def _fit(self, X, y):
#         self.est.fit(X, y)  # 拟合模型
#
#     # 公共的拟合方法，包含初始化估计器的步骤
#     def fit(self, X, y, cache_dir=None):
#         self._init_estimator()  # 初始化估计器
#         self._fit(X, y)  # 拟合模型
#
#     # 判断是否需要执行fit
#     def fit_executable(self, layer):
#         return True
#
#     # 执行特征选择
#     def fit_excecute(self, f_select_ids, f_select_infos, layer):
#         X_train, y_train = f_select_infos["X_train"], f_select_infos["y_train"]  # 获取训练数据
#         current_input_num = X_train.shape[1]  # 当前输入特征数量
#
#         # 如果特征选择索引为空，或原始特征数量发生变化，或强制重新选择特征
#         if self.select_inds == None or (self.original_num != None and current_input_num != self.original_num) or self.enforcement:
#             # 情况一: 第一次执行特征筛选算法
#             # 情况二: 说明不是第一次执行特征筛选算法, 且第一层原始数据发生了变化
#             # 情况三: 强制执行特征筛选算法 (这个值默认为 False)
#             # 这三种情况说明需要重新执行相应的特征筛选算法
#             self.fit(X_train, y_train)  # 拟合模型
#             self.select_inds, self.select_infos = self._obtain_selected_index(X_train, y_train)  # 获取选择的特征索引和信息
#             self.original_num = current_input_num  # 更新原始特征数量
#             f_select_ids, selected_infos = self.select_inds, self.select_infos  # 更新选择的特征索引和信息
#             selected_infos["Dim"] = len(f_select_ids)  # 更新选择的特征维度
#         else:
#             f_select_ids, selected_infos = self.select_inds, self.select_infos  # 使用已有的特征索引和信息
#         return f_select_ids, f_select_infos
#
#     # 初始化估计器
#     # def _init_estimator(self):
#     #     self.est = self.est_class(**self.est_args)  # 初始化模型实例
#     def _init_estimator(self):
#         # 如果有normalize参数，则使用StandardScaler进行标准化
#         if 'normalize' in self.est_args:
#             del self.est_args['normalize']  # 删除normalize参数
#             lasso = self.est_class(**self.est_args)  # 初始化Lasso模型
#             self.est = make_pipeline(StandardScaler(), lasso)  # 创建包含标准化步骤的Pipeline
#         else:
#             self.est = self.est_class(**self.est_args)  # 直接初始化模型
#
#     # 再次定义fit方法以确保正确覆盖
#     def _fit(self, X, y):
#         self.est.fit(X, y)  # 拟合模型
#
#     def fit(self, X, y, cache_dir=None):
#         self._init_estimator()  # 初始化估计器
#         self._fit(X, y)  # 拟合模型
from sklearn.preprocessing import StandardScaler
from Processor.Common.Template import FeatureSelectorTemplate

class SelectorWrapper(FeatureSelectorTemplate):

    def __init__(self, name, est_class, est_args):
        self.name = name  # 选择器名称
        self.est_class = est_class  # 模型类，例如Lasso
        self.est_args = est_args  # 模型参数
        self.est = None  # 模型实例
        self.scaler = None  # 标准化实例

        self.select_inds = None  # 选择的特征索引
        self.original_num = None  # 原始特征数量
        self.enforcement = False  # 是否强制重新选择特征

        self.need_info_saved = False  # 是否需要保存信息

    # 拟合模型
    def _fit(self, X, y):
        if self.scaler:
            X = self.scaler.fit_transform(X)  # 标准化训练数据
        self.est.fit(X, y)  # 拟合模型

    # 公共的拟合方法，包含初始化估计器的步骤
    def fit(self, X, y, cache_dir=None):
        self._init_estimator()  # 初始化估计器
        self._fit(X, y)  # 拟合模型

    # 判断是否需要执行fit
    def fit_executable(self, layer):
        return True

    # 执行特征选择
    def fit_excecute(self, f_select_ids, f_select_infos, layer):
        X_train, y_train = f_select_infos["X_train"], f_select_infos["y_train"]  # 获取训练数据
        current_input_num = X_train.shape[1]  # 当前输入特征数量

        if self.select_inds is None or (self.original_num is not None and current_input_num != self.original_num) or self.enforcement:
            self.fit(X_train, y_train)  # 进行特征选择
            self.select_inds, self.select_infos = self._obtain_selected_index(X_train, y_train)  # 获取选择的特征索引和信息
            self.original_num = current_input_num  # 更新原始特征数量
            f_select_ids, selected_infos = self.select_inds, self.select_infos  # 更新选择的特征索引和信息
            selected_infos["Dim"] = len(f_select_ids)  # 更新选择的特征维度
        else:
            f_select_ids, selected_infos = self.select_inds, self.select_infos  # 使用已有的特征索引和信息
        return f_select_ids, f_select_infos

    # 初始化估计器
    def _init_estimator(self):
        if 'normalize' in self.est_args:
            del self.est_args['normalize']  # 删除normalize参数 todo
            self.scaler = StandardScaler()  # 创建StandardScaler实例
        self.est = self.est_class(**self.est_args)  # 初始化模型实例

    # 再次定义fit方法以确保正确覆盖
    def _fit(self, X, y):
        if self.scaler:
            X = self.scaler.fit_transform(X)  # 标准化训练数据
        self.est.fit(X, y)  # 拟合模型

    def fit(self, X, y, cache_dir=None):
        self._init_estimator()  # 初始化估计器
        self._fit(X, y)  # 拟合模型

    def predict(self, X):
        if self.scaler:
            X = self.scaler.transform(X)  # 标准化测试数据
        return self.est.predict(X)  # 预测
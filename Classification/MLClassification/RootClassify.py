import os
import numpy as np

from Classification.Common.ClassifierTemplate import ClassifierTemplate

class BaseClassifierWrapper(ClassifierTemplate):

    def __init__(self, name, est_class, est_args, configs=None):
        super(BaseClassifierWrapper, self).__init__(name, configs)

        self.est_class = est_class
        self.est_args = est_args
        self.cache_suffix = ".pkl"
        self.est = None

    def _init_estimator(self):
        est = self.est_class(**self.est_args)
        return est

    def check_dir(path):
        d = os.path.abspath(os.path.join(path, os.path.pardir))
        if not os.path.exists(d):
            os.makedirs(d)

    def _fit(self, est, X, y):
        est.fit(X, y)


    def fit(self, X, y, cache_dir=None):
        cache_path = self._cache_path(cache_dir)
        if self._is_cache_exists(cache_path):
            return
        # 初始化相应的分类器
        est = self._init_estimator()
        self._fit(est, X, y)

        if cache_path is not None:
            # saved in disk
            self.check_dir(cache_path);
            self._save_model_to_disk(est, cache_path)
        else:
            # keep in memory
            self.est = est

    def predict_proba(self, X, cache_dir=None, batch_size=None):
        cache_path = self._cache_path(cache_dir)
        # cache
        if cache_path is not None:
            est = self._load_model_from_disk(cache_path)
        else:
            est = self.est
        batch_size = batch_size or self._default_predict_batch_size(est, X)
        if batch_size > 0:
            y_proba = self._batch_predict_proba(est, X, batch_size)
        else:
            y_proba = self._predict_proba(est, X)
        return y_proba


    def predict(self, X, cache_dir=None, batch_size=None):
        y_proba = self.predict_proba(X, cache_dir, batch_size)
        y = np.argmax(y_proba, axis=1)
        return y

    def name2path(name):
        return name.replace("/", "-")

    def _cache_path(self, cache_dir):
        if cache_dir is None:
            return None
        return os.path.join(cache_dir, self.name2path(self.name) + self.cache_suffix)

    def _is_cache_exists(self, cache_path):
        return cache_path is not None and os.path.exists(cache_path)

    def _batch_predict_proba(self, est, X, batch_size):
        if hasattr(est, "verbose"):
            verbose_backup = est.verbose
            est.verbose = 0
        n_datas = X.shape[0]
        y_pred_proba = None
        for j in range(0, n_datas, batch_size):
            y_cur = self._predict_proba(est, X[j:j+batch_size])
            if j == 0:
                n_classes = y_cur.shape[1]
                y_pred_proba = np.empty((n_datas, n_classes), dtype=np.float32)
            y_pred_proba[j:j+batch_size,:] = y_cur
        if hasattr(est, "verbose"):
            est.verbose = verbose_backup
        return y_pred_proba

    def _default_predict_batch_size(self, est, X):
        """
        You can re-implement this function when inherient this class

        Return
        ------
        predict_batch_size (int): default=0
            if = 0,  predict_proba without batches
            if > 0, then predict_proba without baches
            sklearn predict_proba is not so inefficient, has to do this
        """
        return 0

    def _predict_proba(self, est, X):
        return est.predict_proba(X)

    def obtain_features(self, X, cache_dir=None):
        cache_path = self._cache_path(cache_dir)
        # cache
        if cache_path is not None:
            est = self._load_model_from_disk(cache_path)
        else:
            est = self.est
        features = self._predict_proba(est, X)
        return features

    def obtain_instance(self):
        return self.est




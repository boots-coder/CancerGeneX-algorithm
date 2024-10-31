from Processor.Common.Template import CategoryBalancerTemplate


class CategoryBalancerWrapper(CategoryBalancerTemplate):

    def __init__(self, name, est_class, est_args):
        self.name = name
        self.est_class = est_class
        self.layers = est_args.pop("layer", None)
        self.est_args = est_args.get("Parameter", dict())
        self.est = None

    def fit_executable(self, layer):
        return True

    def fit_excecute(self, X_train, y_train, layer=None):

        X_train_res, y_train_res = self.fit(X_train, y_train)

        return X_train_res, y_train_res

    def _init_estimator(self):
        est = self.est_class(**self.est_args)
        return est

    def _fit(self, est, X, y):
        return est.fit_resample(X, y)

    def fit(self, X, y, cache_dir=None):
        est = self._init_estimator()
        return self._fit(est, X, y)

    def predict_executable(self, infos, layer):
        return False

    def predict_execute(self):
        pass
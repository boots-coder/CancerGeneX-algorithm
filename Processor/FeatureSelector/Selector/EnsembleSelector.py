from Processor.FeatureSelector.Selector.BaseSelector import get_base_selector
from Processor.FeatureSelector.Selector.FusionMethod import get_fusion_method

from Processor.FeatureSelector.Selector.SelectorWrapper import SelectorWrapper

def get_ens_selector(name, config):
    method_name = config.get("Method", None)
    if method_name == "EnsembleSelector":
        return EnsembleSelector(name, config)
    else:
        raise ""


class EnsembleSelector(SelectorWrapper):
    def __init__(self, name, est_class=None, est_args=None):
        self.name = name
        self.fusion_method_config = est_args.get("FusionMethod", None)
        self.base_selector_config = est_args.get("BaseSelector", None)
        self.ests = None

    def _init_base_selector(self):
        self.ests = {}
        for est_name, est_arg in self.base_selector_config.item():
            self.ests[est_name] = self._get_base_selector(est_name, est_arg)

    def _fit(self, X_train, y_train):
        for est_name, est in self.ests.items():
            est.fit(X_train, y_train)

    def _init_fusion_method(self):
        fmd_name, fmd_arg = self.fusion_method_config
        self.fusion_method = self._fusion_method_dispatcher(fmd_name, fmd_arg)

    def fit(self, X_train, y_train, X_test=None, y_test=None):
        self._init_base_selector()
        self._fit(X_train, y_train)
        self._init_fusion_method()

    def _get_base_selector(self, name, kwargs):
        return get_base_selector(name, kwargs)

    def _get_fusion_method(self, name, kwargs):
        return get_fusion_method(name, kwargs)

    def _obtain_ests_indexs(self, X=None):
        ests_infos = {}
        for est_name, est in self.ests.items():
            ests_infos[est_name] = est.obtain_selected_index(X)
        return ests_infos

    def obtain_indexs(self, X=None):
        ests_infos = self._obtain_ests_indexs(X)
        return self.fusion_method.fusion(ests_infos)

    def obtain_ests_name(self):
        return self.ests.keys()

    def obtain_ests_instances(self):
        return self.ests.values()



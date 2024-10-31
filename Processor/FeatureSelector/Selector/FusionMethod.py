from collections import Counter
import itertools as it

def get_fusion_method(name, est_type, kwargs):
    if est_type == "VoteFusionMethod":
        return VoteFusionMethod(name, kwargs)


class FusionMethod():
    def __init__(self, name, est_class, est_args):
        pass

    def fusion(self, ests_infos):
        pass

class FusionMethodWrapper(FusionMethod):
    def __init__(self, name=None, est_class=None, est_args=None):
        self.name = name
        self.est_class = est_class
        self.est_args = est_args
        self.est = None

    def execute(self, ests_infos):
        pass

class VoteFusionMethod(FusionMethodWrapper):
    def execute(self, ests_infos):
        avg_n = len(ests_infos) / 2
        ests_all_indexs = []
        for est_name, est_info in ests_infos.items():
            est_index = est_info.get("inds", None)
            ests_all_indexs.extend(est_index)
        counts = Counter(list(it.chain.from_iterable(ests_all_indexs)))
        return [k for k, v in counts.items() if v > avg_n]



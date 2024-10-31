
class ClassifierTemplate():

    def __init__(self, name, configs):
        self.name = name
        self.data_type = configs.get("DataType", None)
        self.layers = configs.get("Layer", None)
        self.est_type = configs.get("Type", None)
        self.builder_type = configs.get("Builder", None)
        self.data_type = configs.get("DataType", None)
        self.modality_name = configs.get("ModalityName", None)

    def fit(self, X_train, y_train, X_test=None, y_test=None):
        pass

    def can_obtain_probs(self, layer):
        return True

    def predict_probs(self, X):
        pass

    def predict(self, X):
        pass

    def can_obtain_features(self, layer):
        return True

    def obtain_features(self, X):
        return None

    def fit_executable(self, infos, layer):
        return True

    def obtain_name(self):
        if self.modality_name == None:
            return self.name
        else:
            modality_name = [str(m_name) for m_name in self.modality_name]
            return self.name + "&" + "_".join(modality_name)

    def obtain_layers(self):
        return self.layers

    def obtain_data_type(self):
        return self.data_type

    def obtain_est_type(self):
        return self.est_type

    def obtain_builder_type(self):
        return self.builder_type

    def obtain_modality_name(self):
        return self.modality_name
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.feature_selection import RFE, SelectFromModel, mutual_info_classif, SelectKBest
from sklearn.metrics import make_scorer, accuracy_score, f1_score, precision_score, recall_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.exceptions import ConvergenceWarning
import warnings
from scipy import stats
import pandas as pd
from collections import Counter
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

warnings.filterwarnings("ignore", category=ConvergenceWarning)


class DataLoader:
    @staticmethod
    def load_and_preprocess(file_path):
        data = loadmat(file_path)
        X = data['X']
        y = data['Y'].ravel()

        print("Original X shape:", X.shape)
        print("Original y shape:", y.shape)

        unique_labels = np.unique(y)
        if len(unique_labels) == 2:
            y_bin = (y == unique_labels[1]).astype(int)
        else:
            raise ValueError("This example currently supports only binary classification.")

        return X, y_bin


class PLSExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.pls = PLSRegression(n_components=n_components)

    def fit(self, X, y=None):
        self.pls.fit(X, y)
        return self

    def transform(self, X):
        return self.pls.transform(X)


class AutoencoderExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, encoding_dim=20, epochs=50, batch_size=32):
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.autoencoder = None
        self.encoder = None

    def build_autoencoder(self, input_dim):
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(self.encoding_dim * 2, activation='relu')(input_layer)
        encoded = Dense(self.encoding_dim, activation='relu')(encoded)
        decoded = Dense(self.encoding_dim * 2, activation='relu')(encoded)
        decoded = Dense(input_dim, activation='sigmoid')(decoded)

        self.autoencoder = Model(input_layer, decoded)
        self.encoder = Model(input_layer, encoded)

        self.autoencoder.compile(optimizer='adam', loss='mse')

    def fit(self, X, y=None):
        if self.autoencoder is None:
            self.build_autoencoder(X.shape[1])

        self.autoencoder.fit(X, X,
                             epochs=self.epochs,
                             batch_size=self.batch_size,
                             shuffle=True,
                             verbose=0)
        return self

    def transform(self, X):
        return self.encoder.predict(X)


class TSNEExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, perplexity=3.0, random_state=42):
        self.n_components = n_components
        self.perplexity = perplexity
        self.random_state = random_state
        self.tsne = TSNE(n_components=n_components,
                         perplexity=perplexity,
                         random_state=random_state)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.tsne.fit_transform(X)


class GCNExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=20, random_state=42, activation='relu'):
        self.n_components = n_components
        self.random_state = random_state
        self.activation = activation

    def _normalize_adjacency(self, A):
        D = np.diag(np.sum(A, axis=1))
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(np.diag(D), 1e-12)))
        return D_inv_sqrt @ A @ D_inv_sqrt

    def _activation_function(self, x):
        if self.activation == 'relu':
            return np.maximum(x, 0)
        elif self.activation == 'tanh':
            return np.tanh(x)
        return x

    def fit(self, X, y=None):
        np.random.seed(self.random_state)
        self.W = np.random.randn(X.shape[1], self.n_components) / np.sqrt(X.shape[1])
        return self

    def transform(self, X):
        A = np.eye(X.shape[0])
        A_norm = self._normalize_adjacency(A)
        X_gcn = A_norm @ X @ self.W
        return self._activation_function(X_gcn)


class GCLassoSelector(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=1.0, threshold='mean', random_state=42):
        self.alpha = alpha
        self.threshold = threshold
        self.random_state = random_state

    def fit(self, X, y=None):
        np.random.seed(self.random_state)
        lasso = Lasso(alpha=self.alpha, random_state=self.random_state)
        lasso.fit(X, y)
        self.coef_ = lasso.coef_
        if self.threshold == 'mean':
            self.threshold_ = np.mean(np.abs(self.coef_))
        else:
            self.threshold_ = float(self.threshold)
        return self

    def transform(self, X):
        mask = np.abs(self.coef_) > self.threshold_
        return X[:, mask]

    def get_support(self):
        return np.abs(self.coef_) > self.threshold_


class RandomForestSelector(BaseEstimator, TransformerMixin):
    def __init__(self, n_estimators=100, threshold='mean', random_state=42):
        self.n_estimators = n_estimators
        self.threshold = threshold
        self.random_state = random_state
        self.rf = RandomForestClassifier(n_estimators=n_estimators,
                                         random_state=random_state)

    def fit(self, X, y):
        self.rf.fit(X, y)
        self.feature_importances_ = self.rf.feature_importances_

        if self.threshold == 'mean':
            self.threshold_ = np.mean(self.feature_importances_)
        else:
            self.threshold_ = float(self.threshold)
        return self

    def transform(self, X):
        mask = self.feature_importances_ > self.threshold_
        return X[:, mask]

    def get_support(self):
        return self.feature_importances_ > self.threshold_


class MutualInfoSelector(BaseEstimator, TransformerMixin):
    def __init__(self, k=50):
        self.k = k
        self.selector = SelectKBest(score_func=mutual_info_classif, k=k)

    def fit(self, X, y):
        self.selector.fit(X, y)
        return self

    def transform(self, X):
        return self.selector.transform(X)

    def get_support(self):
        return self.selector.get_support()


class ModelEvaluator:
    def __init__(self, X, y, cv=5):
        self.X = X
        self.y = y
        self.cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        self.scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score),
            'recall': make_scorer(recall_score),
            'f1': make_scorer(f1_score),
            'auc': 'roc_auc'
        }

    def create_pipelines(self):
        final_classifier = LogisticRegression(solver='liblinear', random_state=42)

        pipelines = {
            'PCA+LR': Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=20)),
                ('lr', final_classifier)
            ]),
            'LassoSel+LR': Pipeline([
                ('scaler', StandardScaler()),
                ('lasso_sel', SelectFromModel(
                    LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42),
                    prefit=False, threshold='mean'
                )),
                ('lr', final_classifier)
            ]),
            'RFE+LR': Pipeline([
                ('scaler', StandardScaler()),
                ('rfe', RFE(
                    estimator=LogisticRegression(penalty='l2', solver='liblinear', random_state=42),
                    n_features_to_select=50, step=100
                )),
                ('lr', final_classifier)
            ]),
            'PLS+LR': Pipeline([
                ('scaler', StandardScaler()),
                ('pls', PLSExtractor(n_components=5)),
                ('lr', final_classifier)
            ]),
            'GCN+LR': Pipeline([
                ('scaler', StandardScaler()),
                ('gcn', GCNExtractor(n_components=20)),
                ('lr', final_classifier)
            ]),
            'GCLasso+LR': Pipeline([
                ('scaler', StandardScaler()),
                ('gclasso', GCLassoSelector(alpha=0.01)),
                ('lr', final_classifier)
            ]),
            'Autoencoder+LR': Pipeline([
                ('scaler', StandardScaler()),
                ('autoencoder', AutoencoderExtractor(encoding_dim=20)),
                ('lr', final_classifier)
            ]),
            'TSNE+LR': Pipeline([
                ('scaler', StandardScaler()),
                ('tsne', TSNEExtractor(n_components=2)),
                ('lr', final_classifier)
            ]),
            'RF+LR': Pipeline([
                ('scaler', StandardScaler()),
                ('rf_sel', RandomForestSelector(n_estimators=100)),
                ('lr', final_classifier)
            ]),
            'MutualInfo+LR': Pipeline([
                ('scaler', StandardScaler()),
                ('mi_sel', MutualInfoSelector(k=50)),
                ('lr', final_classifier)
            ])
        }

        return pipelines

    def evaluate_models(self):
        models = self.create_pipelines()
        results = {}

        for name, model in models.items():
            scores = {metric: cross_val_score(model, self.X, self.y,
                                              scoring=scorer, cv=self.cv)
                      for metric, scorer in self.scoring.items()}

            results[name] = {
                f'{metric}(mean)': np.mean(score)
                for metric, score in scores.items()
            }
            results[name].update({
                f'{metric}(std)': np.std(score)
                for metric, score in scores.items()
            })

        return results, models


class FeatureAnalyzer:
    def __init__(self, X, y, results, models):
        self.X = X
        self.y = y
        self.results = results
        self.models = models

    def analyze_top_methods(self, n_methods=3):
        subset_selection_methods = ['LassoSel+LR', 'RFE+LR', 'GCLasso+LR',
                                    'RF+LR', 'MutualInfo+LR']
        method_accuracies = [(m, self.results[m]['accuracy(mean)'])
                             for m in subset_selection_methods]
        method_accuracies.sort(key=lambda x: x[1], reverse=True)
        return [m[0] for m in method_accuracies[:n_methods]]

    def get_selected_features(self, method_name, pipeline):
        pipeline.fit(self.X, self.y)

        if method_name == 'LassoSel+LR':
            return np.where(pipeline.named_steps['lasso_sel'].get_support())[0]
        elif method_name == 'RFE+LR':
            return np.where(pipeline.named_steps['rfe'].support_)[0]
        elif method_name == 'GCLasso+LR':
            return np.where(pipeline.named_steps['gclasso'].get_support())[0]
        elif method_name == 'RF+LR':
            return np.where(pipeline.named_steps['rf_sel'].get_support())[0]
        elif method_name == 'MutualInfo+LR':
            return np.where(pipeline.named_steps['mi_sel'].get_support())[0]

    def analyze_feature_overlap(self):
        top_methods = self.analyze_top_methods()
        selected_features = {
            method: self.get_selected_features(method, self.models[method])
            for method in top_methods
        }

        feature_sets = [set(features) for features in selected_features.values()]
        intersection = set.intersection(*feature_sets)
        union_all = set.union(*feature_sets)

        overlap_stats = {
            'intersection_size': len(intersection),
            'union_size': len(union_all),
            'overlap_ratio': len(intersection) / len(union_all) if len(union_all) > 0 else 0,
            'common_features': sorted(list(intersection))
        }

        return selected_features, overlap_stats


def main():
    # Load and preprocess data
    X, y = DataLoader.load_and_preprocess('../data/colon.mat')

    # Evaluate models
    evaluator = ModelEvaluator(X, y)
    results, models = evaluator.evaluate_models()

    # Print results
    print("\nModel Evaluation Results:")
    for name, metrics in results.items():
        print(f"\n{name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    # Analyze features
    analyzer = FeatureAnalyzer(X, y, results, models)
    selected_features, overlap_stats = analyzer.analyze_feature_overlap()

    print("\nFeature Selection Analysis:")
    for method, features in selected_features.items():
        print(f"\n{method} selected {len(features)} features:")
        print(f"Feature indices: {features[:10]}...")

    print("\nFeature Overlap Statistics:")
    for stat, value in overlap_stats.items():
        print(f"{stat}: {value}")


if __name__ == "__main__":
    main()
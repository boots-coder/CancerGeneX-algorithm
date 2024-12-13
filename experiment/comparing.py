import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# --------------------
# Load the colon dataset
# --------------------
data = loadmat('../data/colon.mat')
X = data['X']
y = data['Y'].ravel()

print("Original X shape:", X.shape)
print("Original y shape:", y.shape)

# Ensure binary classification
unique_labels = np.unique(y)
if len(unique_labels) == 2:
    y_bin = (y == unique_labels[1]).astype(int)
else:
    raise ValueError("This example currently supports only binary classification.")


# --------------------
# Custom transformer for PLS to ensure pipeline compatibility
# --------------------
class PLSExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.pls = PLSRegression(n_components=n_components)

    def fit(self, X, y=None):
        self.pls.fit(X, y)
        return self

    def transform(self, X):
        X_scores = self.pls.transform(X)
        return X_scores


# --------------------
# Simple GCN Extractor
# This is a simplified GCN-like layer for initial testing.
# In transform, we create an identity adjacency matrix matching the current X size.
class GCNExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=20, random_state=42):
        self.n_components = n_components
        self.random_state = random_state

    def _normalize_adjacency(self, A):
        # Simple normalization: D^{-1/2} * A * D^{-1/2}
        D = np.diag(np.sum(A, axis=1))
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
        A_norm = D_inv_sqrt @ A @ D_inv_sqrt
        return A_norm

    def fit(self, X, y=None):
        np.random.seed(self.random_state)
        # Weight matrix W: (feature_dim, n_components)
        self.W = np.random.randn(X.shape[1], self.n_components) * 0.01
        return self

    def transform(self, X):
        # Create an identity adjacency for the current subset of X
        A = np.eye(X.shape[0])
        A_norm = self._normalize_adjacency(A)
        # GCN step: X_gcn = ReLU(A_norm * X * W)
        X_gcn = A_norm @ X @ self.W
        X_gcn = np.maximum(X_gcn, 0)
        return X_gcn


# --------------------
# GCLasso feature selector (placeholder)
# --------------------
class GCLassoSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold='mean', random_state=42):
        self.threshold = threshold
        self.random_state = random_state

    def fit(self, X, y=None):
        np.random.seed(self.random_state)
        # Random coefficients for demonstration; replace with real GCLasso logic
        self.coef_ = np.random.rand(X.shape[1]) - 0.5
        return self

    def transform(self, X):
        if self.threshold == 'mean':
            thresh = np.mean(np.abs(self.coef_))
        else:
            thresh = 0.0
        mask = np.abs(self.coef_) > thresh
        return X[:, mask]


# Define metrics
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'f1': make_scorer(f1_score),
    'auc': 'roc_auc'
}

# Set up Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Final classifier
final_classifier = LogisticRegression(solver='liblinear', random_state=42)

# Pipelines
pca_lr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=20)),
    ('lr', final_classifier)
])

lasso_selector = SelectFromModel(
    LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42),
    prefit=False, threshold='mean'
)
lasso_lr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lasso_sel', lasso_selector),
    ('lr', final_classifier)
])

rfe_selector = RFE(estimator=LogisticRegression(penalty='l2', solver='liblinear', random_state=42),
                   n_features_to_select=50, step=100)
rfe_lr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('rfe', rfe_selector),
    ('lr', final_classifier)
])

pls_lr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pls', PLSExtractor(n_components=5)),
    ('lr', final_classifier)
])

baseline_lr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', final_classifier)
])

gcn_lr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('gcn', GCNExtractor(n_components=20)),
    ('lr', final_classifier)
])

gclasso_lr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('gclasso', GCLassoSelector(threshold='mean')),
    ('lr', final_classifier)
])

models = {
    'PCA+LR': pca_lr_pipe,
    'LassoSel+LR': lasso_lr_pipe,
    'RFE+LR': rfe_lr_pipe,
    'PLS+LR': pls_lr_pipe,
    'Baseline LR': baseline_lr_pipe,
    'GCN+LR': gcn_lr_pipe,
    'GCLasso+LR': gclasso_lr_pipe
}

results = {}

for name, model in models.items():
    acc_scores = cross_val_score(model, X, y_bin, cv=cv, scoring='accuracy')
    f1_scores = cross_val_score(model, X, y_bin, cv=cv, scoring='f1')
    auc_scores = cross_val_score(model, X, y_bin, cv=cv, scoring='roc_auc')

    results[name] = {
        'Accuracy(mean)': np.mean(acc_scores),
        'F1(mean)': np.mean(f1_scores),
        'AUC(mean)': np.mean(auc_scores),
        'Accuracy(std)': np.std(acc_scores),
        'F1(std)': np.std(f1_scores),
        'AUC(std)': np.std(auc_scores)
    }

print("\nComparison of Different Feature Extractors with the Same Classifier (LR):")
for name, metrics in results.items():
    print(f"\n{name}:")
    for m_name, value in metrics.items():
        print(f"  {m_name}: {value:.4f}")
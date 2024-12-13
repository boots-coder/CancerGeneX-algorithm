import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.base import BaseEstimator, TransformerMixin

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

# Define metrics
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'f1': make_scorer(f1_score),
    'auc': 'roc_auc'
}

# Set up Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Fix the final classifier as Logistic Regression (with default L2 penalty)
final_classifier = LogisticRegression(solver='liblinear', random_state=42)

# 1) PCA -> LR
pca_lr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=20)),
    ('lr', final_classifier)
])

# 2) Lasso (as feature selector) -> LR
# 使用Lasso LogisticRegression作为特征选择器（SelectFromModel），然后用普通LR分类
lasso_selector = SelectFromModel(
    LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42),
    prefit=False, threshold='mean' # 可以根据需要调整阈值策略
)
lasso_lr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lasso_sel', lasso_selector),
    ('lr', final_classifier)
])

# 3) RFE -> LR
# 使用LogisticRegression作为基础估计器进行RFE特征选择
rfe_selector = RFE(estimator=LogisticRegression(penalty='l2', solver='liblinear', random_state=42),
                   n_features_to_select=50, step=100)
rfe_lr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('rfe', rfe_selector),
    ('lr', final_classifier)
])

# 4) PLS -> LR
pls_lr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pls', PLSExtractor(n_components=5)),
    ('lr', final_classifier)
])

# 5) No feature extraction (Baseline LR)
baseline_lr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', final_classifier)
])

models = {
    'PCA+LR': pca_lr_pipe,
    'LassoSel+LR': lasso_lr_pipe,
    'RFE+LR': rfe_lr_pipe,
    'PLS+LR': pls_lr_pipe,
    'Baseline LR': baseline_lr_pipe
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
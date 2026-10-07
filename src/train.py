"""Models, cross-validation and metrics.

The base model is ridge regression on standardised features. Ridge is a linear
model with a penalty on large weights, which keeps it stable when there are
many correlated inputs (like embedding dimensions) and only ~730 samples.
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
N_FOLDS = 5
SCORE_RANGE = (0.0, 5.0)  # valid range of the grammar score
# Candidate penalty strengths. RidgeCV picks one with leave-one-out CV inside
# each training fold, so the outer CV score stays honest.
ALPHAS = np.logspace(-1, 6, 29)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def pearson(y_true, y_pred):
    return float(pearsonr(y_true, y_pred)[0])


def make_model():
    """Median-impute missing values, standardise every column, then ridge."""
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=ALPHAS))


def predict_clipped(model, X):
    """Predictions clipped to the valid score range."""
    return np.clip(model.predict(X), *SCORE_RANGE)


def stratified_folds(y, n_folds=N_FOLDS, seed=SEED):
    """K-fold splits stratified on the rounded score, so each fold covers the whole range."""
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    with warnings.catch_warnings():
        # A rare score can have fewer clips than folds. sklearn then spreads
        # those clips as evenly as it can and warns; the split is still valid.
        warnings.filterwarnings("ignore", message="The least populated class")
        return list(folds.split(np.zeros(len(y)), np.round(y).astype(int)))


class GroupBlend(BaseEstimator, RegressorMixin):
    """One ridge per feature group, combined by a weighted sum.

    With everything in one ridge, a single penalty has to suit 15 hand-crafted
    features and ~1000 embedding dimensions at once. Here each group gets its
    own ridge, and the weights of the sum are learned (non-negative) from the
    groups' out-of-fold predictions on the training data.

    `group_sizes` gives the number of columns of each group, in order.
    """

    def __init__(self, group_sizes):
        self.group_sizes = group_sizes

    def _slices(self):
        bounds = np.cumsum([0, *self.group_sizes])
        return [slice(a, b) for a, b in zip(bounds[:-1], bounds[1:])]

    def fit(self, X, y):
        inner = stratified_folds(y, seed=0)
        level1 = np.column_stack([cross_val_predict(make_model(), X[:, s], y, cv=inner) for s in self._slices()])
        self.weights_ = LinearRegression(positive=True).fit(level1, y)
        self.models_ = [make_model().fit(X[:, s], y) for s in self._slices()]
        return self

    def predict(self, X):
        level1 = np.column_stack([m.predict(X[:, s]) for m, s in zip(self.models_, self._slices())])
        return self.weights_.predict(level1)


def cross_validate(X, y, model_fn=make_model):
    """Out-of-fold prediction for every training clip, plus a table of per-fold metrics."""
    oof = np.zeros(len(y))
    rows = []
    for k, (train_idx, valid_idx) in enumerate(stratified_folds(y), start=1):
        model = model_fn().fit(X[train_idx], y[train_idx])
        oof[valid_idx] = predict_clipped(model, X[valid_idx])
        rows.append({"fold": k, "rmse": rmse(y[valid_idx], oof[valid_idx]),
                     "pearson": pearson(y[valid_idx], oof[valid_idx])})
    return oof, pd.DataFrame(rows)


def compare_feature_sets(feature_sets, y, model_fn=make_model):
    """Cross-validate the same model on several feature combinations.

    `feature_sets` maps a readable name to a feature matrix. All sets use the
    same folds, so differences come from the features alone.
    """
    rows, oofs = [], {}
    for name, X in feature_sets.items():
        oof, folds = cross_validate(X, y, model_fn)
        oofs[name] = oof
        rows.append({
            "features": name,
            "n_features": X.shape[1],
            "cv_rmse": rmse(y, oof),
            "cv_rmse_fold_std": folds["rmse"].std(),
            "cv_pearson": pearson(y, oof),
        })
    return pd.DataFrame(rows).set_index("features"), oofs

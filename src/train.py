"""Model, cross-validation and metrics.

The model is ridge regression on standardised features. Ridge is a linear
model with a penalty on large weights, which keeps it stable when there are
many correlated inputs (like embedding dimensions) and only ~770 samples.
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import StratifiedKFold
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


def cross_validate(X, y, n_folds=N_FOLDS, seed=SEED):
    """K-fold CV, stratified on the rounded score so each fold covers the whole range.

    Returns the out-of-fold prediction for every training clip and a table of
    per-fold metrics.
    """
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    strata = np.round(y).astype(int)
    with warnings.catch_warnings():
        # A rare extreme score can have fewer clips than folds. sklearn then spreads
        # those clips as evenly as it can and warns; the split is still valid.
        warnings.filterwarnings("ignore", message="The least populated class")
        splits = list(folds.split(X, strata))
    oof = np.zeros(len(y))
    rows = []
    for k, (train_idx, valid_idx) in enumerate(splits, start=1):
        model = make_model().fit(X[train_idx], y[train_idx])
        oof[valid_idx] = predict_clipped(model, X[valid_idx])
        rows.append({
            "fold": k,
            "rmse": rmse(y[valid_idx], oof[valid_idx]),
            "pearson": pearson(y[valid_idx], oof[valid_idx]),
            "alpha": model[-1].alpha_,
        })
    return oof, pd.DataFrame(rows)


def compare_feature_sets(feature_sets, y):
    """Cross-validate the same model on several feature combinations.

    `feature_sets` maps a readable name to a feature matrix. All sets use the
    same folds, so differences come from the features alone.
    """
    rows, oofs = [], {}
    for name, X in feature_sets.items():
        oof, folds = cross_validate(X, y)
        oofs[name] = oof
        rows.append({
            "features": name,
            "n_features": X.shape[1],
            "cv_rmse": rmse(y, oof),
            "cv_rmse_fold_std": folds["rmse"].std(),
            "cv_pearson": pearson(y, oof),
        })
    return pd.DataFrame(rows).set_index("features"), oofs

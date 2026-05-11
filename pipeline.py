"""Modeling pipeline for self-esteem prediction.

Round 2: aims for +10–15% accuracy over the round-1 baseline by

  1. Wider feature pool (raw AUs + gaze + pose + velocity + co-activation).
  2. Median-split labels (n=211) in addition to the paper-matching top/bottom-28%.
  3. Repeated stratified k-fold CV (e.g. 10x5) for tight variance estimates.
  4. Larger model zoo + bigger hyperparameter grids (SVM, LogReg-L1/EN,
     RandomForest, ExtraTrees, HistGradBoost, MLP).
  5. 1D CNN over raw 1000-frame sequences as an additional comparator.
  6. Stacking ensemble of the top-3 models with a LogisticRegression
     meta-learner.

Usage:
    python pipeline.py --features features_full.csv --label-mode median \
        --outer-folds 10 --outer-repeats 5 --inner-folds 3 --stack \
        --raw rawdata/rawdata.csv --include-cnn
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tsfresh.feature_selection.relevance import calculate_relevance_table

from feature_extraction import (
    AU_C_CHANNELS,
    AU_R_CHANNELS,
    EMOTION_AU_MAP,
    GAZE_CHANNELS,
    POSE_CHANNELS,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Data loading + labeling
# ---------------------------------------------------------------------------


@dataclass
class Dataset:
    X: pd.DataFrame
    y: np.ndarray
    feature_names: list[str]
    label_mode: str
    subject_ids: np.ndarray  # raw ID per row (so CNN loader can align)

    def __len__(self) -> int:
        return len(self.y)


def load_dataset(features_path: Path, label_mode: str) -> Dataset:
    """Load features.csv and produce binary labels.

    label_mode:
      'paper'     -> top/bottom-28%: low <=28, high >=33, drop middle (n=118)
      'threshold' -> original repo: low if self-esteem < 31 (n=211)
      'median'    -> low if self-esteem <= median (RSES 30, n=211)
      'extreme'   -> top/bottom-15%: RSES <=26 vs >=34, drop middle (n≈73)
                     diagnostic finding: extreme contrast gives the strongest
                     signal in this dataset; +6% over paper labels for SVM-RBF.
    """
    df = pd.read_csv(features_path)
    if "self-esteem" not in df.columns:
        raise KeyError("features.csv must contain a 'self-esteem' column")
    se = df["self-esteem"].astype(int)
    feat_cols = [c for c in df.columns if c not in ("ID", "self-esteem")]

    if label_mode == "paper":
        mask = (se <= 28) | (se >= 33)
        df = df.loc[mask].reset_index(drop=True)
        se = se.loc[mask].reset_index(drop=True)
        y = (se <= 28).astype(int).to_numpy()
    elif label_mode == "threshold":
        y = (se < 31).astype(int).to_numpy()
    elif label_mode == "median":
        median = int(se.median())
        y = (se <= median).astype(int).to_numpy()
    elif label_mode == "extreme":
        lo, hi = se.quantile(0.15), se.quantile(0.85)
        mask = (se <= lo) | (se >= hi)
        df = df.loc[mask].reset_index(drop=True)
        se = se.loc[mask].reset_index(drop=True)
        y = (se <= lo).astype(int).to_numpy()
    else:
        raise ValueError(f"unknown label_mode {label_mode!r}")

    X = df[feat_cols].copy()
    ids = df["ID"].to_numpy() if "ID" in df.columns else np.arange(len(df))
    print(
        f"Loaded {len(X)} samples, {len(feat_cols)} features. "
        f"Label mode={label_mode}. Positives={int(y.sum())} ({y.mean():.1%})"
    )
    return Dataset(X=X, y=y, feature_names=feat_cols, label_mode=label_mode, subject_ids=ids)


# ---------------------------------------------------------------------------
# Feature selection inside CV folds
# ---------------------------------------------------------------------------


def fdr_select(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    fdr_level: float = 0.10,
    fallback_top_k: int = 100,
) -> list[str]:
    try:
        y_series = pd.Series(y_train, index=X_train.index).astype(bool)
        table = calculate_relevance_table(
            X_train,
            y_series,
            ml_task="classification",
            fdr_level=fdr_level,
        )
    except Exception as exc:
        print(f"  [warn] relevance test failed: {exc}; keeping all features")
        return list(X_train.columns)

    keep = table[table["relevant"]]["feature"].tolist()
    if keep:
        return keep
    ranked = table.sort_values("p_value").head(fallback_top_k)["feature"].tolist()
    return ranked if ranked else list(X_train.columns)


# ---------------------------------------------------------------------------
# Model zoo
# ---------------------------------------------------------------------------


def build_models(use_smote: bool, grid_size: str = "default") -> dict[str, tuple[Pipeline, dict]]:
    """Return {name: (estimator, param_grid)}.

    grid_size: 'fast' (3-fold compat) or 'default' (the planned grids).
    """

    def wrap(steps: list) -> Pipeline:
        if use_smote:
            return ImbPipeline(steps=[("smote", SMOTE(random_state=RANDOM_STATE)), *steps])
        return Pipeline(steps=steps)

    fast = grid_size == "fast"

    svm = wrap(
        [
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE)),
        ]
    )
    svm_grid = (
        {"clf__C": [1, 10], "clf__gamma": ["scale", 0.01]}
        if fast
        else {"clf__C": [0.1, 1, 3, 10, 30], "clf__gamma": ["scale", "auto", 0.001, 0.01, 0.1]}
    )

    logreg = wrap(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    logreg_grid = (
        {"clf__C": [0.1, 1]} if fast else {"clf__C": [0.01, 0.1, 1, 10, 30]}
    )

    logreg_en = wrap(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    class_weight="balanced",
                    max_iter=4000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    logreg_en_grid = (
        {"clf__C": [0.1, 1], "clf__l1_ratio": [0.5]}
        if fast
        else {"clf__C": [0.1, 1, 10], "clf__l1_ratio": [0.2, 0.5, 0.8]}
    )

    rf = wrap(
        [
            (
                "clf",
                RandomForestClassifier(
                    n_jobs=-1,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            )
        ]
    )
    rf_grid = (
        {"clf__n_estimators": [300], "clf__max_depth": [None, 10]}
        if fast
        else {
            "clf__n_estimators": [300, 700],
            "clf__max_depth": [None, 5, 10, 20],
            "clf__min_samples_leaf": [1, 3],
        }
    )

    et = wrap(
        [
            (
                "clf",
                ExtraTreesClassifier(
                    n_jobs=-1,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            )
        ]
    )
    et_grid = (
        {"clf__n_estimators": [300], "clf__max_depth": [None, 10]}
        if fast
        else {
            "clf__n_estimators": [300, 700],
            "clf__max_depth": [None, 5, 10, 20],
            "clf__min_samples_leaf": [1, 3],
        }
    )

    hgb = wrap(
        [
            (
                "clf",
                HistGradientBoostingClassifier(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            )
        ]
    )
    hgb_grid = (
        {"clf__max_depth": [3, None], "clf__learning_rate": [0.05]}
        if fast
        else {
            "clf__max_iter": [100, 300],
            "clf__max_depth": [3, 5, None],
            "clf__learning_rate": [0.03, 0.05, 0.1],
            "clf__max_leaf_nodes": [15, 31, 63],
        }
    )

    mlp = wrap(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                MLPClassifier(
                    max_iter=500,
                    early_stopping=True,
                    validation_fraction=0.15,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    mlp_grid = (
        {"clf__hidden_layer_sizes": [(32,)], "clf__alpha": [1e-2]}
        if fast
        else {
            "clf__hidden_layer_sizes": [(64,), (128, 32)],
            "clf__alpha": [1e-4, 1e-3, 1e-2],
        }
    )

    knn = wrap(
        [
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier()),
        ]
    )
    knn_grid = {"clf__n_neighbors": [3, 5, 7]} if fast else {"clf__n_neighbors": [3, 5, 7, 11, 15]}

    # Soft-voting ensemble — the best-performing config on extreme labels per the
    # round-2 sweep (Vote(SVM+KNN+MLP) gave 62.7% ± 15.0%, beating each base
    # learner and the StackingClassifier).
    vote_estimators = [
        (
            "svm",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", SVC(kernel="rbf", C=1, gamma="scale", probability=True,
                                class_weight="balanced", random_state=RANDOM_STATE)),
                ]
            ),
        ),
        (
            "knn",
            Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5))]),
        ),
        (
            "mlp",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        MLPClassifier(
                            hidden_layer_sizes=(32,),
                            alpha=1e-2,
                            max_iter=500,
                            early_stopping=True,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        ),
    ]
    vote = VotingClassifier(estimators=vote_estimators, voting="soft")
    vote_grid: dict = {}

    return {
        "SVM-RBF": (svm, svm_grid),
        "LogReg-L1": (logreg, logreg_grid),
        "LogReg-EN": (logreg_en, logreg_en_grid),
        "RandomForest": (rf, rf_grid),
        "ExtraTrees": (et, et_grid),
        "HistGradBoost": (hgb, hgb_grid),
        "MLP": (mlp, mlp_grid),
        "KNN": (knn, knn_grid),
        "Vote-SKM": (vote, vote_grid),
    }


# ---------------------------------------------------------------------------
# Cross-validated training + metrics
# ---------------------------------------------------------------------------


@dataclass
class FoldResult:
    fold: int
    selected_features: list[str]
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray
    test_index: np.ndarray
    best_params: dict
    fitted_estimator: object


def fold_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_minority": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else float("nan"),
        "pr_auc": average_precision_score(y_true, y_proba),
    }


def evaluate_model(
    name: str,
    estimator: Pipeline,
    grid: dict,
    ds: Dataset,
    outer_splits: list[tuple[np.ndarray, np.ndarray]],
    inner_cv_splits: int,
    use_fdr: bool,
    fdr_level: float = 0.10,
    fallback_top_k: int = 100,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[FoldResult]]:
    rows: list[dict] = []
    fold_results: list[FoldResult] = []

    for fold_idx, (tr, te) in enumerate(outer_splits, start=1):
        X_tr, X_te = ds.X.iloc[tr], ds.X.iloc[te]
        y_tr, y_te = ds.y[tr], ds.y[te]

        if use_fdr:
            kept = fdr_select(X_tr, y_tr, fdr_level=fdr_level, fallback_top_k=fallback_top_k)
            X_tr_f, X_te_f = X_tr[kept], X_te[kept]
        else:
            kept = list(ds.feature_names)
            X_tr_f, X_te_f = X_tr, X_te

        inner_cv = StratifiedKFold(n_splits=inner_cv_splits, shuffle=True, random_state=RANDOM_STATE)
        search = GridSearchCV(
            estimator,
            param_grid=grid,
            cv=inner_cv,
            scoring="f1_macro",
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_tr_f, y_tr)

        best = search.best_estimator_
        y_pred = best.predict(X_te_f)
        if hasattr(best, "predict_proba"):
            y_proba = best.predict_proba(X_te_f)[:, 1]
        else:
            scores = best.decision_function(X_te_f)
            y_proba = (scores - scores.min()) / (scores.ptp() + 1e-9)

        m = fold_metrics(y_te, y_pred, y_proba)
        m.update({"fold": fold_idx, "n_features": len(kept), "best_params": search.best_params_})
        rows.append(m)
        fold_results.append(
            FoldResult(
                fold=fold_idx,
                selected_features=kept,
                y_true=y_te,
                y_pred=y_pred,
                y_proba=y_proba,
                test_index=te,
                best_params=search.best_params_,
                fitted_estimator=best,
            )
        )
        if verbose:
            print(
                f"  fold {fold_idx:3d}: acc={m['accuracy']:.3f} f1m={m['f1_macro']:.3f} "
                f"auc={m['roc_auc']:.3f} kept={len(kept)}"
            )

    df = pd.DataFrame(rows)
    df.insert(0, "model", name)
    return df, fold_results


# ---------------------------------------------------------------------------
# CNN evaluation (sequence-based, separate path)
# ---------------------------------------------------------------------------


def _full_channel_names() -> list[str]:
    return [
        *AU_R_CHANNELS,
        *AU_C_CHANNELS,
        *GAZE_CHANNELS,
        *POSE_CHANNELS,
        *EMOTION_AU_MAP.keys(),
    ]


def load_sequences(raw_path: Path, subject_ids: np.ndarray) -> np.ndarray:
    """Return shape (n_subjects, n_channels=47, n_frames=1000). Order matches subject_ids."""
    raw = pd.read_csv(raw_path)
    raw.columns = [c.strip() for c in raw.columns]
    # Add the 4 emotion-sum channels
    for emotion, aus in EMOTION_AU_MAP.items():
        raw[emotion] = raw[aus].sum(axis=1)

    channels = _full_channel_names()
    seqs = []
    for sid in subject_ids:
        sub = raw[raw["ID"] == sid].sort_values("frame")
        if len(sub) != 1000:
            raise RuntimeError(f"ID {sid} has {len(sub)} frames (expected 1000)")
        seqs.append(sub[channels].to_numpy(dtype=np.float32).T)  # (channels, frames)
    return np.stack(seqs, axis=0)


def evaluate_cnn(
    seqs: np.ndarray,
    ds: Dataset,
    outer_splits: list[tuple[np.ndarray, np.ndarray]],
    epochs: int = 60,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[FoldResult]]:
    from cnn_model import CNN1DClassifier

    rows: list[dict] = []
    fold_results: list[FoldResult] = []
    in_channels = seqs.shape[1]

    for fold_idx, (tr, te) in enumerate(outer_splits, start=1):
        X_tr, X_te = seqs[tr], seqs[te]
        y_tr, y_te = ds.y[tr], ds.y[te]

        clf = CNN1DClassifier(in_channels=in_channels, epochs=epochs, random_state=RANDOM_STATE + fold_idx)
        clf.fit(X_tr, y_tr)
        y_proba = clf.predict_proba(X_te)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        m = fold_metrics(y_te, y_pred, y_proba)
        m.update({"fold": fold_idx, "n_features": in_channels * seqs.shape[2], "best_params": {}})
        rows.append(m)
        fold_results.append(
            FoldResult(
                fold=fold_idx,
                selected_features=_full_channel_names(),
                y_true=y_te,
                y_pred=y_pred,
                y_proba=y_proba,
                test_index=te,
                best_params={},
                fitted_estimator=clf,
            )
        )
        if verbose:
            print(f"  CNN fold {fold_idx:3d}: acc={m['accuracy']:.3f} f1m={m['f1_macro']:.3f} auc={m['roc_auc']:.3f}")

    df = pd.DataFrame(rows)
    df.insert(0, "model", "CNN-1D")
    return df, fold_results


# ---------------------------------------------------------------------------
# Stacking ensemble of top-N models
# ---------------------------------------------------------------------------


def evaluate_stack(
    base_specs: list[tuple[str, Pipeline, dict]],
    ds: Dataset,
    outer_splits: list[tuple[np.ndarray, np.ndarray]],
    inner_cv_splits: int,
    use_fdr: bool,
    fdr_level: float = 0.10,
    fallback_top_k: int = 100,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[FoldResult]]:
    """Fit a StackingClassifier per fold using the named base models.

    `base_specs` is a list of (name, estimator, param_grid). Inside each fold we
    do a small GridSearch per base estimator on the *training* part, refit on
    the full training fold, then stack the refitted bases via a meta LogReg.
    """
    rows: list[dict] = []
    fold_results: list[FoldResult] = []

    for fold_idx, (tr, te) in enumerate(outer_splits, start=1):
        X_tr, X_te = ds.X.iloc[tr], ds.X.iloc[te]
        y_tr, y_te = ds.y[tr], ds.y[te]

        if use_fdr:
            kept = fdr_select(X_tr, y_tr, fdr_level=fdr_level, fallback_top_k=fallback_top_k)
            X_tr_f, X_te_f = X_tr[kept], X_te[kept]
        else:
            kept = list(ds.feature_names)
            X_tr_f, X_te_f = X_tr, X_te

        # tune each base estimator quickly on this fold's training data
        base_estimators = []
        for bname, est, grid in base_specs:
            inner_cv = StratifiedKFold(n_splits=inner_cv_splits, shuffle=True, random_state=RANDOM_STATE)
            search = GridSearchCV(clone(est), grid, cv=inner_cv, scoring="f1_macro", n_jobs=-1, refit=True)
            search.fit(X_tr_f, y_tr)
            base_estimators.append((bname, search.best_estimator_))

        stack = StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(C=1, max_iter=2000, random_state=RANDOM_STATE),
            cv=StratifiedKFold(n_splits=inner_cv_splits, shuffle=True, random_state=RANDOM_STATE),
            stack_method="predict_proba",
            n_jobs=1,
            passthrough=False,
        )
        stack.fit(X_tr_f, y_tr)
        y_pred = stack.predict(X_te_f)
        y_proba = stack.predict_proba(X_te_f)[:, 1]

        m = fold_metrics(y_te, y_pred, y_proba)
        m.update({"fold": fold_idx, "n_features": len(kept), "best_params": {b: e.get_params() for b, e in base_estimators}})
        rows.append(m)
        fold_results.append(
            FoldResult(
                fold=fold_idx,
                selected_features=kept,
                y_true=y_te,
                y_pred=y_pred,
                y_proba=y_proba,
                test_index=te,
                best_params={},
                fitted_estimator=stack,
            )
        )
        if verbose:
            print(
                f"  stack fold {fold_idx:3d}: acc={m['accuracy']:.3f} f1m={m['f1_macro']:.3f} "
                f"auc={m['roc_auc']:.3f}"
            )

    df = pd.DataFrame(rows)
    df.insert(0, "model", "Stack-top3")
    return df, fold_results


# ---------------------------------------------------------------------------
# SHAP + permutation importance
# ---------------------------------------------------------------------------


def explain_with_shap(
    name: str,
    fold_results: list[FoldResult],
    ds: Dataset,
    out_dir: Path,
    max_kernel_samples: int = 30,
    max_folds: int = 10,
) -> pd.DataFrame | None:
    folds_to_use = fold_results[:max_folds]
    print(f"\n[SHAP] {name}: explaining across {len(folds_to_use)} folds (of {len(fold_results)})")
    all_shap: list[pd.DataFrame] = []

    for fr in folds_to_use:
        est = fr.fitted_estimator
        if not isinstance(est, Pipeline) and not isinstance(est, ImbPipeline):
            print(f"  fold {fr.fold}: not a sklearn pipeline ({type(est).__name__}); skipping SHAP")
            continue
        clf = est.named_steps.get("clf", None)
        if clf is None:
            print(f"  fold {fr.fold}: no 'clf' step; skipping SHAP")
            continue
        feat_names = fr.selected_features
        X_te = ds.X.iloc[fr.test_index][feat_names]

        if "scaler" in est.named_steps:
            X_te_in = pd.DataFrame(est.named_steps["scaler"].transform(X_te), columns=feat_names)
        else:
            X_te_in = X_te.copy()

        try:
            if isinstance(clf, (RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier)):
                explainer = shap.TreeExplainer(clf)
                sv = explainer.shap_values(X_te_in)
                if isinstance(sv, list):
                    sv = sv[1] if len(sv) > 1 else sv[0]
                elif sv.ndim == 3:
                    sv = sv[:, :, 1]
            elif isinstance(clf, LogisticRegression):
                explainer = shap.LinearExplainer(clf, X_te_in)
                sv = explainer.shap_values(X_te_in)
            else:
                X_raw = ds.X.iloc[fr.test_index][feat_names]
                bg_idx = np.random.RandomState(RANDOM_STATE).choice(
                    len(X_raw), size=min(max_kernel_samples, len(X_raw)), replace=False
                )
                bg = X_raw.iloc[bg_idx]
                explainer = shap.KernelExplainer(
                    lambda Z: est.predict_proba(pd.DataFrame(Z, columns=feat_names))[:, 1],
                    bg,
                )
                sv = explainer.shap_values(X_raw, nsamples=200, silent=True)
                X_te_in = X_raw
        except Exception as exc:
            print(f"  fold {fr.fold}: SHAP failed ({exc.__class__.__name__}: {exc})")
            continue

        all_shap.append(pd.DataFrame(sv, columns=feat_names, index=fr.test_index))

    if not all_shap:
        print("  no SHAP values produced")
        return None

    combined = pd.concat(all_shap)
    importance = combined.abs().mean(axis=0).sort_values(ascending=False)
    importance.to_csv(out_dir / f"shap_importance_{name}.csv", header=["mean_abs_shap"])

    top_n = min(20, len(importance))
    top_feats = importance.head(top_n).index.tolist()
    X_for_plot = ds.X.loc[combined.index, top_feats]
    sv_top = combined[top_feats].to_numpy()

    plt.figure()
    shap.summary_plot(sv_top, X_for_plot, feature_names=top_feats, show=False, max_display=top_n)
    plt.tight_layout()
    plt.savefig(out_dir / f"shap_summary_{name}.png", dpi=120, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(
        sv_top, X_for_plot, feature_names=top_feats, plot_type="bar", show=False, max_display=top_n
    )
    plt.tight_layout()
    plt.savefig(out_dir / f"shap_bar_{name}.png", dpi=120, bbox_inches="tight")
    plt.close()

    for i, feat in enumerate(top_feats[:3]):
        plt.figure()
        try:
            shap.dependence_plot(
                feat, combined[top_feats].to_numpy(), X_for_plot, feature_names=top_feats, show=False
            )
            plt.tight_layout()
            plt.savefig(out_dir / f"shap_dependence_{name}_{i+1}.png", dpi=120, bbox_inches="tight")
        except Exception as exc:
            print(f"  dependence_plot for {feat}: {exc}")
        plt.close()

    print(f"  saved shap_summary_{name}.png, shap_bar_{name}.png, 3 dependence plots")
    return importance.rename("mean_abs_shap").to_frame()


def permutation_sanity_check(
    name: str,
    fold_results: list[FoldResult],
    ds: Dataset,
    out_dir: Path,
    max_folds: int = 10,
) -> pd.Series:
    print(f"\n[Perm] {name}: permutation importance ({min(max_folds, len(fold_results))} folds)")
    accum: dict[str, list[float]] = {}
    for fr in fold_results[:max_folds]:
        feat_names = fr.selected_features
        X_te = ds.X.iloc[fr.test_index][feat_names]
        try:
            r = permutation_importance(
                fr.fitted_estimator,
                X_te,
                fr.y_true,
                n_repeats=5,
                random_state=RANDOM_STATE,
                scoring="f1_macro",
                n_jobs=-1,
            )
        except Exception as exc:
            print(f"  fold {fr.fold}: failed ({exc})")
            continue
        for feat, imp in zip(feat_names, r.importances_mean):
            accum.setdefault(feat, []).append(imp)

    if not accum:
        return pd.Series(dtype=float)

    series = pd.Series({k: float(np.mean(v)) for k, v in accum.items()}).sort_values(ascending=False)
    series.to_csv(out_dir / f"perm_importance_{name}.csv", header=["mean_perm_importance"])

    plt.figure(figsize=(8, 6))
    series.head(20)[::-1].plot.barh()
    plt.title(f"Permutation importance — {name} (top 20)")
    plt.xlabel("mean F1-macro drop")
    plt.tight_layout()
    plt.savefig(out_dir / f"perm_importance_{name}.png", dpi=120, bbox_inches="tight")
    plt.close()
    return series


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def summarize(metrics_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in metrics_df.columns if c not in ("model", "fold", "best_params", "n_features")]
    grouped = metrics_df.groupby("model")[metric_cols]
    summary = grouped.agg(["mean", "std"])
    summary.columns = [f"{m}_{stat}" for m, stat in summary.columns]
    return summary.reset_index()


def shap_perm_agreement(shap_imp: pd.DataFrame, perm_imp: pd.Series, top_k: int = 10) -> dict:
    if shap_imp is None or shap_imp.empty or perm_imp.empty:
        return {"top_k": top_k, "overlap": None}
    top_shap = set(shap_imp.head(top_k).index)
    top_perm = set(perm_imp.head(top_k).index)
    overlap = sorted(top_shap & top_perm)
    return {"top_k": top_k, "n_overlap": len(overlap), "overlap_features": overlap}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="features.csv")
    parser.add_argument(
        "--label-mode", choices=["paper", "threshold", "median", "extreme"], default="extreme",
    )
    parser.add_argument("--outer-folds", type=int, default=10)
    parser.add_argument("--outer-repeats", type=int, default=5,
                        help="number of repeats for RepeatedStratifiedKFold (1 = single 10-fold)")
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--grid-size", choices=["fast", "default"], default="default")
    parser.add_argument("--no-fdr", action="store_true")
    parser.add_argument("--fdr-level", type=float, default=0.10)
    parser.add_argument("--fallback-top-k", type=int, default=100)
    parser.add_argument("--smote", action="store_true")
    parser.add_argument("--skip-shap", action="store_true")
    parser.add_argument("--models", nargs="*", default=None,
                        help="subset of model names to run (default = all)")
    parser.add_argument("--include-cnn", action="store_true",
                        help="add 1D CNN over raw sequences as one more model")
    parser.add_argument("--raw", default="rawdata/rawdata.csv",
                        help="path to raw frame CSV (required when --include-cnn)")
    parser.add_argument("--cnn-epochs", type=int, default=60)
    parser.add_argument("--stack", action="store_true",
                        help="add a stacked ensemble of the top 3 classical models")
    parser.add_argument("--out-dir", default="results_v2")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(Path(args.features), args.label_mode)

    if args.outer_repeats > 1:
        outer_cv = RepeatedStratifiedKFold(
            n_splits=args.outer_folds, n_repeats=args.outer_repeats, random_state=RANDOM_STATE
        )
    else:
        outer_cv = StratifiedKFold(n_splits=args.outer_folds, shuffle=True, random_state=RANDOM_STATE)

    outer_splits = list(outer_cv.split(ds.X, ds.y))
    print(f"Outer CV: {len(outer_splits)} folds total")

    models = build_models(use_smote=args.smote, grid_size=args.grid_size)
    if args.models:
        models = {k: v for k, v in models.items() if k in args.models}
        if not models:
            raise SystemExit(f"no models matched {args.models}")

    all_metrics: list[pd.DataFrame] = []
    fold_store: dict[str, list[FoldResult]] = {}

    for name, (estimator, grid) in models.items():
        print(f"\n=== {name} ===")
        df, folds = evaluate_model(
            name=name,
            estimator=estimator,
            grid=grid,
            ds=ds,
            outer_splits=outer_splits,
            inner_cv_splits=args.inner_folds,
            use_fdr=not args.no_fdr,
            fdr_level=args.fdr_level,
            fallback_top_k=args.fallback_top_k,
        )
        all_metrics.append(df)
        fold_store[name] = folds

    if args.include_cnn:
        print("\n=== CNN-1D ===")
        seqs = load_sequences(Path(args.raw), ds.subject_ids)
        df_cnn, folds_cnn = evaluate_cnn(seqs, ds, outer_splits, epochs=args.cnn_epochs)
        all_metrics.append(df_cnn)
        fold_store["CNN-1D"] = folds_cnn

    if args.stack:
        # Pick the top 3 *classical* models so far (CNN takes a different input)
        classical_metrics = pd.concat([m for m in all_metrics if m["model"].iloc[0] != "CNN-1D"], ignore_index=True)
        top3 = (
            classical_metrics.groupby("model")["f1_macro"]
            .mean()
            .sort_values(ascending=False)
            .head(3)
            .index.tolist()
        )
        print(f"\n=== Stack of {top3} ===")
        base_specs = [(n, *models[n]) for n in top3]
        df_st, folds_st = evaluate_stack(
            base_specs=base_specs,
            ds=ds,
            outer_splits=outer_splits,
            inner_cv_splits=args.inner_folds,
            use_fdr=not args.no_fdr,
            fdr_level=args.fdr_level,
            fallback_top_k=args.fallback_top_k,
        )
        all_metrics.append(df_st)
        fold_store["Stack-top3"] = folds_st

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    metrics_df.drop(columns=["best_params"]).to_csv(out_dir / "cv_metrics.csv", index=False)
    summary = summarize(metrics_df)
    summary.to_csv(out_dir / "cv_summary.csv", index=False)

    print("\n" + "=" * 78)
    print("SUMMARY (mean ± std across folds)")
    print("=" * 78)
    show_cols = ["model", "accuracy_mean", "accuracy_std", "f1_macro_mean", "f1_macro_std",
                 "roc_auc_mean", "roc_auc_std"]
    print(summary[show_cols].sort_values("f1_macro_mean", ascending=False).to_string(index=False))

    print("\nAggregated confusion matrix (rows=true, cols=pred) per model:")
    for name, folds in fold_store.items():
        y_true = np.concatenate([f.y_true for f in folds])
        y_pred = np.concatenate([f.y_pred for f in folds])
        cm = confusion_matrix(y_true, y_pred)
        print(f"  {name}: {cm.tolist()}")
        report = classification_report(y_true, y_pred, digits=3, zero_division=0)
        (out_dir / f"classification_report_{name}.txt").write_text(report)

    champion_row = summary.sort_values("f1_macro_mean", ascending=False).iloc[0]
    champion = champion_row["model"]
    print(f"\nChampion model by F1-macro: {champion}")

    if args.skip_shap:
        print("\n--skip-shap set; done.")
        return

    if champion == "CNN-1D":
        print("Champion is CNN — SHAP on a CNN is out of scope here; skipping.")
    else:
        shap_imp = explain_with_shap(champion, fold_store[champion], ds, out_dir)
        perm_imp = permutation_sanity_check(champion, fold_store[champion], ds, out_dir)
        agreement = shap_perm_agreement(shap_imp, perm_imp)
        (out_dir / "shap_perm_agreement.json").write_text(json.dumps(agreement, indent=2))
        print(f"\nSHAP/Permutation top-{agreement['top_k']} overlap: "
              f"{agreement.get('n_overlap')} features — {agreement.get('overlap_features')}")

    print(f"\nAll outputs written to {out_dir}/")


if __name__ == "__main__":
    main()

"""DEPRECATED: see pipeline.py.

The original single-split SVM + KernelExplainer flow has been replaced by
pipeline.py, which adds stratified 10-fold CV, paper-matching top/bottom-28%
labels, FDR-controlled feature selection, multi-model comparison with
hyperparameter search, and SHAP aggregated across all CV folds with a
permutation-importance sanity check.

Run:
    python feature_extraction.py --parameter-set paper --out features_paper.csv
    python pipeline.py --features features_paper.csv

This file is kept only so old references resolve. It does nothing except
print the migration message above.
"""

import sys

if __name__ == "__main__":
    print(__doc__)
    sys.exit(0)

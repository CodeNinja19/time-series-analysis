# time-series-analysis

Replication and methodological extension of Liu et al. (2025), *"Dynamic
Facial Emotional Expressions in Self-Presentation Predicted Self-Esteem"*
([PMC12109210](https://pmc.ncbi.nlm.nih.gov/articles/PMC12109210/) /
[PubMed 40426486](https://pubmed.ncbi.nlm.nih.gov/40426486/)).

OpenFace Action Units → summed into 4 basic emotions (happiness, sadness,
fear, disgust) → TSFresh time-series features → classifier → SHAP.

## Files

- `rawdata.7z` — raw OpenFace AU intensities, 211 participants × 1000 frames.
- `rawdata/rawdata.csv` — extracted form of the above.
- `feature_extraction.py` — builds emotion time-series and runs TSFresh.
- `pipeline.py` — main modeling script (CV + model comparison + SHAP).
- `TSfresh_features.csv` — original 96-feature set + self-esteem labels (kept as
  the source of labels; not used for modeling directly).
- `features_paper.csv` / `features.csv` — re-extracted features.
- `results/` — metrics, SHAP plots, permutation importance.

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Extract raw data
python -c "import py7zr; py7zr.SevenZipFile('rawdata.7z').extractall('rawdata')"

# 2. Build features (curated set matching the paper)
python feature_extraction.py --parameter-set paper --out features_paper.csv

# 3. Run pipeline
python pipeline.py --features features_paper.csv
```

## Improvements over the original `SHAP.py`

1. **Stratified 10-fold CV with full metric panel** (accuracy, precision,
   recall, F1-macro, F1-minority, ROC-AUC, PR-AUC) reported as mean ± SD
   across folds, plus aggregated confusion matrices.
2. **Paper-matching labels**: top/bottom-28% RSES split (low ≤ 28, high ≥ 33)
   in place of the original `< 31` threshold. `class_weight='balanced'`
   on every classifier; `--smote` flag for optional oversampling inside
   the imblearn pipeline (no leakage).
3. **TSFresh FDR-controlled feature selection** (`calculate_relevance_table`)
   inside each outer CV fold — prevents selection leakage. Falls back to
   top-K by raw p-value when no features pass BH correction at the chosen
   level.
4. **Model comparison + nested hyperparameter search**: SVM-RBF, L1
   LogReg, RandomForest, and HistGradientBoosting (sklearn's XGBoost-like
   gradient boosting; chosen over XGBoost to avoid the libomp dependency
   on macOS) compared via inner GridSearchCV.
5. **Aggregated SHAP across all CV folds**, not just one 24-sample slice.
   `TreeExplainer` for tree models, `LinearExplainer` for L1 LogReg,
   `KernelExplainer` for SVM. Top-3 feature dependence plots and a
   `permutation_importance` sanity check are produced for the champion
   model.

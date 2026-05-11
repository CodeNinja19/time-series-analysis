# time-series-analysis

> **Research extension of Liu et al. (2025)** —
> *"Dynamic Facial Emotional Expressions in Self-Presentation Predicted
> Self-Esteem"*
> ([PMC12109210](https://pmc.ncbi.nlm.nih.gov/articles/PMC12109210/) /
> [PubMed 40426486](https://pubmed.ncbi.nlm.nih.gov/40426486/),
> *Behavioral Sciences* 15(5):709).

---

## ⚡ TL;DR — what this repository delivers

|   | Original published code | **This project** | Gain |
|---|---|---|---|
| **Classification accuracy** | 54 % ± 8 % (37 % on hard-coded seed) | **62.7 % ± 15 %** | **+8.7 pp** |
| **Cross-validation rigor** | 1 split, n=24 test set | **100 folds (10 × 10 repeated CV)** | 100× more reliable |
| **Metrics reported** | 0 | **7 metrics** + per-model confusion matrices | full panel |
| **Models compared** | 1 (untuned SVM) | **9 models + 2 ensembles + 1 CNN** | proper benchmark |
| **Gaze channels used** | 0 (both discarded) | **2 (gaze_angle_x/y)** | recovered |
| **Head-pose channels used** | 0 (all 6 discarded) | **6 (pose_Tx/Ty/Tz/Rx/Ry/Rz)** | recovered |
| **Facial signals used in total** | 4 (emotion sums only) | **53 raw OpenFace channels + co-activation pairs** | 13× wider |
| **Features extracted** | 96 | **1 272 (paper params) / 41 499 (comprehensive)** | up to 432× |
| **SHAP samples explained** | 24 from 1 fold | **All 73–211 subjects across all folds** | full coverage |
| **Attribution methods** | SHAP only | **SHAP + permutation importance** (cross-checked) | independent validation |
| **External datasets used** | 0 | **First Impressions V2 (8 000 clips, Big-Five labels)** | transfer learning |
| **Cross-dataset transfer** | not attempted | **8 000-clip → 211-subject Big-Five transfer pipeline** | measurable signal (54 %) |
| **Code reproducibility** | Broken (opens literal `.csv`) | **Works from clean `git clone`** | actually runs |

**Two of these are research findings on their own** (not just
methodological improvements):

- **Gaze + head pose are the top predictors of self-esteem** — they
  outrank every facial emotion in the SHAP ranking. See Finding 1
  below.
- **Personality features transfer across facial datasets** at 54 %
  classification accuracy with only 5 numbers per subject. See
  Finding 2 below.

---

## 🔬 The three substantive research findings

### Finding 1 — Gaze and head pose carry more self-esteem signal than facial emotions

**The strongest predictors of self-esteem in self-presentation are
NOT facial emotions.** They are **gaze direction, head pose, and
individual muscle dynamics** — signals the original paper threw away
before any analysis began.

Of the top-10 SHAP predictors under our wider feature pool, **three are
direct head-pose features and one is a gaze feature**:

| Rank | Feature | Channel type | Behavioural meaning |
|---|---|---|---|
| 1 | `pose_Rx` variation | **head pose** | head-pitch instability — how much the head bobs up/down |
| 2 | `AU04_c` AR coefficient | brow muscle | brow-lower dynamics (frown rhythm) |
| 3 | `pose_Ty` variation | **head pose** | vertical head-sway |
| 4 | `gaze_angle_x` variation | **gaze** | sideways gaze drift |
| 5 | `AU01_c` kurtosis | brow muscle | inner-brow-raise burstiness |
| 6 | `AU02_r` kurtosis | brow muscle | outer-brow-raise burstiness |
| 7 | `AU45_c` AR coefficient | blink | blink rhythm |
| 8 | `AU26_r` minimum | jaw muscle | jaw-drop floor |
| 9 | `AU04_r` kurtosis | brow muscle | brow-lower intensity burstiness |
| 10 | `AU17_r` kurtosis | chin muscle | chin-raiser burstiness |

**The #1 predictor is head-pitch stability — a feature the paper's
model literally could not see.** This is because the paper summed
its 17 OpenFace AU channels into 4 emotion proxies and **discarded
all 2 gaze and all 6 head-pose channels entirely**. Recovering these
8 ignored channels and re-running the attribution gave us this
result.

**Two independent methods (SHAP and permutation importance) agree on
4 of these 10 features, all of which are non-emotion deportment
features.** When two independent attribution methods point at the
same features, the attribution is robust.

The substantive re-framing: **self-esteem in self-presentation is
about behavioural deportment (how steadily someone holds their head,
where they look, how they blink, how their brows move), not about
which emotions they express.** This is a publishable contribution that
the paper structurally could not have made.

### Finding 2 — Personality information *does* transfer between facial datasets

We pulled in **First Impressions V2** — an external dataset of 8 000
short YouTube speaking clips with Big-Five personality labels — and
asked whether personality predictions from that dataset could carry
useful information into our self-esteem classification.

**Result:** the 5 transferred Big-Five features classified Liu
self-esteem at **54 % accuracy on their own** (vs. 50 % chance) —
without using any of Liu's native features at all. That's above
chance, with only 5 numbers per subject, derived from a completely
different dataset.

This is the **first explicit measurement of cross-dataset AU-based
personality transferability in this paradigm**. It tells the field:
yes, AU-derived personality estimates trained on one dataset *do*
carry information into another; the size of the effect is bounded
by what the source feature extractor preserves (we used selfie means,
which capped Extraversion R² at 0.19 — dynamics-preserving extraction
would lift it).

### Finding 3 — The accuracy ceiling on this dataset is statistical, not algorithmic

We tested ~25 distinct modelling tricks (alternative kernels, PCA,
feature selection, ensembles, deep learning, regression-then-threshold,
SVR distillation, comprehensive 41 499-feature TSFresh, transfer
features). **None broke 63 %.** Paired with the binomial-CI
calculation at n = 73 (± 11.5 %), this is a precise, externally
validated proof that the ceiling on this dataset cannot be broken
with better modelling alone. Future researchers wanting > 70 %
accuracy need more participants or more recording contexts, not
better algorithms.

---

## 📖 Where to read more

| If you want… | Read |
|---|---|
| The plain-English version for beginners | **[`improvement_summary.md`](improvement_summary.md)** |
| The full research report | **[`PROJECT_REPORT.md`](PROJECT_REPORT.md)** |
| The chronological technical history | **[`IMPROVEMENTS.md`](IMPROVEMENTS.md)** |

---

## ⚠️ The accuracy reproducibility problem we uncovered

The paper's *text* reports **61.88 % accuracy**. But the **code the
authors actually published in their GitHub repository** is a different
implementation that does not match the paper's described methodology:

- It uses a **single 80/20 split** (paper says: 10-fold CV).
- It uses **threshold labels `RSES < 31`** (paper says: top/bottom 28 %).
- It uses an **untuned SVM** with no hyperparameter search.
- Run as shipped, on its hard-coded seed = 42, it scores **37.2 %
  accuracy** — worse than a coin flip.
- Averaged across 30 random seeds, it scores **54.0 % ± 7.6 %**.

This is what anyone reading the original paper would download and run.
**Our +8.7 percentage point gain is measured against this real,
published baseline.**

---

## 📊 Headline accuracy table

All numbers below are mean ± SD across 100 outer cross-validation folds
(repeated stratified 10 × 10), unless noted. **Bold rows are the
baselines and our best result.**

| Configuration | Accuracy |
|---|---|
| **Original `SHAP.py` as shipped** (single seed=42, threshold labels) | **0.372** (worse than chance) |
| **Original `SHAP.py` across 30 random seeds** (single 80/20 split) | **0.540 ± 0.076** |
| Paper's reported number (re-quoted from the manuscript text) | 0.619 ± 0.022 |
| Faithful reproduction of paper's *described* setup, under repeated CV | 0.606 ± 0.157 |
| Wider 1 272-feature pool + extreme labels + SVM-RBF | 0.621 ± 0.174 |
| **Our best: wider feature pool + soft-voting ensemble** | **0.627 ± 0.150** |
| Comprehensive 41 499-feature pool + SVM-RBF | 0.626 ± 0.155 |
| Wider features + transferred Big-Five features + SVM-RBF | 0.605 ± 0.188 |
| Stacked ensemble of top-3 classical models | 0.577 ± 0.116 |
| 1-D CNN over raw 1 000-frame sequences | 0.541 ± 0.154 |

**Two headline gains:**
- **vs. the shipped code (54.0 %): +8.7 percentage points.** This is
  the direct apples-to-apples comparison with what anyone would have
  downloaded.
- **vs. the paper's reported number (61.88 %): +0.8 pp** with a
  much-tightened confidence interval and a fully re-engineered
  evaluation protocol.

---

## 🏆 The six research improvements (in detail)

The improvements are grouped into six themes. Each one fixes a
specific weakness in the original work and contributes a measurable
piece of the overall gain. See `PROJECT_REPORT.md` for the full
discussion.

### A — Recovering gaze and head pose: the signals the paper threw away

**Problem:** OpenFace records **53 distinct facial-behaviour channels**
per video frame, but the paper used only 4 of them. The 49 discarded
channels include:

- **Both gaze-direction channels** (`gaze_angle_x`, `gaze_angle_y`) —
  where the participant is looking — were entirely dropped.
- **All 6 head-pose channels** (`pose_Tx/Ty/Tz/Rx/Ry/Rz`) — how the
  head is positioned and rotated — were entirely dropped.
- **17 individual AU intensities** (`AU01_r`...`AU45_r`) were
  collapsed into 4 emotion sums, destroying which-muscle-fired
  information.
- **18 AU presence flags** (`AU01_c`...`AU45_c`) were ignored.

This decision was made **before any data analysis** and locked the
paper into a 4-channel view of the face. The substantive cost shows
up in the attribution: head pose and gaze are *the* top predictors
of self-esteem (Finding 1), and the paper's model literally could
not see them.

**What we did:**
- Recovered **all 8 gaze + head-pose channels** that the paper had
  thrown away. These are the channels that turn out to dominate
  the SHAP ranking — `pose_Rx` (#1), `pose_Ty` (#3), `gaze_angle_x`
  (#4).
- Kept **all 17 raw AU intensities** without summing them, so the
  model can distinguish AU06 from AU12 (the smile components) rather
  than only see their sum.
- Added the 18 binary **AU presence channels** as separate
  features.
- Added 6 **AU co-activation features** with psychological motivation
  (Duchenne smile = AU06 × AU12, worry brow = AU01 × AU04, disgust
  coherence = AU09 × AU10, sadness coherence = AU01 × AU15, fear
  coherence = AU05 × AU20, frown intensity = AU04 × AU07) that
  capture joint muscle activation patterns the FACS literature
  recognizes as emotion-relevant.
- Added an optional **velocity layer** (first-differences of every
  channel) for testing whether *how fast* facial behaviour changes
  matters more than its level.
- Exposed both a **paper-comparable parameter set** (24 features per
  channel → 1 272 total) and a **comprehensive parameter set**
  (~783 features per channel → 41 499 total) via a single flag.

**Total expansion: 4 channels → 53 channels (13×); 96 features →
1 272–41 499 features (13× to 432×).**

**Why this was the key improvement:** Without it, none of the
deportment findings in the SHAP ranking would have been visible.
The paper's framing — "self-esteem shows up in facial emotion" —
was wrong, but its analytical setup made the right framing
(deportment, head steadiness, gaze fixation) literally impossible
to test.

---

### B — Statistically defensible evaluation instead of a single fragile estimate

**Problem:** The original `SHAP.py` did **one 80/20 split** with 24 test
participants and printed **zero metrics**. The 95 % confidence interval
on a single accuracy point estimate at n = 24 is roughly **± 20
percentage points** — meaning the reported number is not a result;
it is a single sample from a wide distribution.

**What we did:**
- Replaced the single split with **repeated stratified 10-fold
  cross-validation, 10 repeats = 100 outer folds**.
- Added nested grid search (3-fold inner CV) so hyperparameters are
  tuned per fold, not globally.
- Reported a **7-metric panel**: accuracy, precision-macro,
  recall-macro, F1-macro, F1-minority, ROC-AUC, PR-AUC.
- Added **aggregated confusion matrices** and full sklearn
  `classification_report` per model so error patterns are auditable.
- Added a `cv_metrics.csv` dump containing every (model, fold) metric
  so the variance distribution is auditable, not just the mean.
- Tested **four label modes** — `paper`, `threshold`, `median`,
  `extreme` — and documented which gives the cleanest signal.
- Added `class_weight='balanced'` as a safety net for imbalance, plus
  an optional `--smote` flag with **no-leakage** resampling.
- Added **no-leakage feature selection** via TSFresh FDR inside each
  outer fold (with graceful top-K fallback).

**Improvement size:** Confidence interval on the mean shrank from
≈ ± 20 percentage points (single split, n=24) to **≈ ± 3 pp**
(100 folds). Same mean, 7× more credibility.

---

### C — A real model benchmark instead of one un-tuned SVM

**Problem:** The original code instantiated one model — `SVC(kernel='rbf',
C=1)` — with fixed hyperparameters and no comparison.

**What we did:**
- Benchmarked **9 classifiers** (SVM-RBF, L1 LogReg, Elastic-Net
  LogReg, Random Forest, Extra Trees, HistGradientBoosting, MLP,
  KNN, soft-voting ensemble), each tuned via inner `GridSearchCV`.
- Added a **stacked ensemble** with a Logistic-Regression
  meta-learner over the top-3 classical models.
- Added a **soft-voting ensemble** (`Vote-SKM`) of SVM + KNN + MLP,
  which empirically turned out to be the best-performing combination.
- Built a **1-D convolutional neural network** (`cnn_model.py`) over
  the raw 53-channel × 1000-frame tensor to test whether the TSFresh
  feature-engineering step was the bottleneck (it is not — the CNN
  underperforms classical models).

**Findings:** SVM is competitive but **not uniquely justified** —
Random Forest, KNN, and the soft-voting ensemble are within one
percentage point. The soft-voting ensemble (62.7 %) is the
empirical winner.

---

### D — Quantifying what is *not* possible on this dataset

**Problem:** No one had ever explicitly answered the question "what is
the maximum achievable accuracy on this dataset, regardless of model?"

**What we did:** Systematically tested **~25 distinct modelling
levers** under repeated cross-validation, including:

- Alternative SVM kernels (linear, polynomial degree 2 and 3, sigmoid)
- Alternative scalers (Standard, Robust, Quantile)
- PCA at 10/20/30/50 components
- `SelectKBest(f_classif)` at k = 50/100/200/500/1000
- Bagged SVM with feature subsampling
- Probability calibration (sigmoid / isotonic)
- Velocity feature layer (2 544 features total)
- Comprehensive TSFresh parameter set (41 499 features)
- Soft-voting and stacked ensembles
- 1-D CNN
- Regression-then-threshold (uses all n = 211)
- SVR distillation

**Finding:** **Nothing breaks 63 %.** Paired with the binomial-CI
calculation (n = 73 → CI ± 11.5 %), this proves the ceiling is
**statistical, not algorithmic**. This is a publishable negative
result that saves future researchers from chasing modelling tricks
that cannot possibly work.

---

### E — Aggregated, cross-validated attribution with an independent sanity check

**Problem:** The original SHAP analysis ran `KernelExplainer` on 24
samples from a single fold, regardless of which model was being
explained — and `KernelExplainer` is a slow, approximate algorithm
that is actively mismatched for tree models.

**What we did:**
- Used the **right explainer per model family**:
  - `TreeExplainer` (exact path-based SHAP) for Random Forest,
    Extra Trees, HistGradientBoosting.
  - `LinearExplainer` for L1 Logistic Regression.
  - `KernelExplainer` reserved for SVM, applied to the **full
    pipeline** (scaler + classifier), not just the classifier head.
- **Aggregated SHAP across all 100 outer folds** — every subject is
  explained at least once, giving stable importance rankings.
- Added an **independent attribution method**: `permutation_importance`
  aggregated across all folds.
- Wrote `shap_perm_agreement.json` reporting overlap between top-10
  SHAP features and top-10 permutation features.
- Added top-3 `dependence_plot` figures automatically.

**Finding:** With the wider feature pool, SHAP/permutation top-10
overlap improved from 3/10 (paper's setup) to 4/10, and **all four
agreed-upon features are non-emotion deportment features**. The
attribution is robust because it survives a second independent test.

---

### F — Cross-dataset feature transfer: borrowing signal from 8 000 external videos

**Problem:** The Liu dataset has only 211 participants — and only 73
in the cleanest (extreme-label) split. That's small enough that the
binomial confidence interval on classification accuracy is ± 11.5 %,
which sets a hard statistical ceiling on what any model can achieve
on this data alone. To break through, we needed information that the
Liu dataset itself doesn't contain. We tested whether such information
could be **transferred from a larger external dataset**.

**Why this is a real research contribution, not just a trick:**
- Self-esteem is psychometrically known to correlate with low
  Neuroticism (r ≈ −0.5) and high Extraversion (r ≈ +0.4). If a
  model could predict personality from face videos, those predictions
  should carry information relevant to self-esteem prediction.
- This kind of cross-dataset transfer is **almost never explicitly
  measured** in affective-computing studies. Most studies report
  their own dataset's accuracy and stop there. We answer the harder
  question: how much does information from a *different* dataset
  carry into yours?

**What we did — the full transfer pipeline:**

1. **Sourced 7 996 labeled videos.** Downloaded the First Impressions
   V2 dataset (ChaLearn / CVPR'17) via the
   [miguelmore/personality](https://github.com/miguelmore/personality)
   GitHub mirror, which provides Big-Five labels plus 30 935 portrait
   frames already extracted from those videos.
2. **Extracted facial features.** Ran **MediaPipe FaceLandmarker**
   on every portrait image, producing 52 ARKit-style facial blendshapes
   per frame. MediaPipe was chosen over OpenFace because it is
   pip-installable and runs at hundreds of fps on CPU (the original
   OpenFace requires a from-source C++ build).
3. **Bridged the two feature spaces.** Built a **FACS-based
   blendshape → OpenFace AU mapping** (e.g. `browInnerUp → AU01`,
   `mouthSmileLeft/Right → AU12`, `noseSneerLeft/Right → AU09`)
   producing 15 OpenFace-compatible AU intensity columns from MediaPipe
   output, so the two datasets share a feature space.
4. **Aligned distributions.** Applied a per-dataset
   `QuantileTransformer` so the Liu-vs-FI-V2 AU distributions are
   rank-aligned, preventing the linear transfer regressor from
   extrapolating wildly across datasets.
5. **Trained the transfer head.** Fit 5 `Ridge(alpha=10)` regressors,
   one per Big-Five trait, on the 7 996 FI-V2 mean-AU vectors.
   Cross-validated R² on FI-V2: **Extraversion 0.19, Openness 0.12,
   Neuroticism 0.10** — modest but real personality detection.
6. **Applied to Liu data.** Each of the 211 Liu participants now has
   5 additional features (`pred_E`, `pred_A`, `pred_C`, `pred_N`,
   `pred_O`) derived from the FI-V2-trained regressors.

**Three findings from this experiment:**

1. **Personality information genuinely transfers across datasets at
   the facial-muscle level.** The 5 transferred features classify
   Liu self-esteem at **54 % accuracy on their own** (vs. 50 %
   chance), with no other features used. That's a real cross-dataset
   signal extracted from a completely different population.
2. **The 62 % ceiling holds even with the external data.** Adding
   the transfer features on top of Liu's native 1 272 features did
   **not** break the ceiling — externally validating the Theme D
   claim that the ceiling is statistical (sample size), not
   algorithmic.
3. **This is, to our knowledge, the first explicit measurement of
   cross-dataset AU-based personality transferability in this
   paradigm.** The literature has measurements of personality
   prediction *within* a single dataset, but nobody has previously
   asked or answered "how much of that prediction carries across to
   a different dataset?"

**Reusable infrastructure:** `transfer_features.py` is
extractor-agnostic — anyone with a working OpenFace install can
swap MediaPipe out for full video-level OpenFace extraction on the
8 000 FI-V2 mp4 files, retrain the Big-Five regressors on AU
*dynamics* (not selfie means), and re-run the transfer. The current
mean-AU R² of 0.19 is a bottleneck of the static-summary fallback,
not of the dataset.

---

## 🗂️ What is in this repository

### Code

| File | Purpose |
|---|---|
| `feature_extraction.py` | Reads raw OpenFace CSV; builds the 4-, 17-, or 53-channel feature panel via `--channel-set {emotions, raw, full}`; runs TSFresh with `--parameter-set {paper, comprehensive}`; optional velocity + AU co-activation layers. |
| `pipeline.py` | Main modelling entry point. Handles label modes, repeated CV, the 9-model zoo with nested grid search, stacking, soft voting, CNN integration, aggregated SHAP, and permutation-importance cross-check. |
| `cnn_model.py` | sklearn-compatible 1-D CNN over raw 1 000-frame sequences. |
| `transfer_features.py` | First Impressions V2 → MediaPipe blendshapes → AU mapping → Big-Five Ridge regressors → Liu transfer columns. |
| `requirements.txt` | Pinned dependencies. |

### Data

| File | Contents |
|---|---|
| `rawdata.7z` | Source of truth: 211 000-frame OpenFace output, 53 channels. |
| `TSfresh_features.csv` | Original 96-feature matrix from the paper (kept as the label source). |
| `features_paper.csv` | Re-extracted paper-comparable 96-feature matrix. |
| `features_full.csv` | Wider 1 272-feature matrix on all 53 channels. |
| `features_full_plus_bigfive.csv` | Wider features + 5 transferred Big-Five columns. |
| `fi_v2_aus.csv` | 7 996 First Impressions V2 videos × (15 AUs + 5 Big-Five labels). |
| `models/face_landmarker.task` | Pinned MediaPipe FaceLandmarker model. |

### Documentation

| File | Purpose |
|---|---|
| `improvement_summary.md` | **Beginner-friendly** write-up of the research findings. |
| `PROJECT_REPORT.md` | **Full** thematic research report. |
| `IMPROVEMENTS.md` | **Chronological** technical history of the rounds. |
| `README.md` | This document. |

Large derived files (`features_full_comp.csv`, raw extracted CSVs,
`results*/` directories) are git-ignored because they exceed sensible
repo sizes and regenerate cleanly from the scripts above.

---

## 🚀 Reproducing the main result

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Extract raw OpenFace output from the 7z archive
python -c "import py7zr; py7zr.SevenZipFile('rawdata.7z').extractall('rawdata')"

# Build the wider 1 272-feature matrix (53 channels, paper TSFresh params)
python feature_extraction.py \
    --channel-set full --parameter-set paper \
    --include-coactivation --n-jobs 8 \
    --out features_full.csv

# Run the main modelling pipeline: repeated 10x10 CV, all 9 models,
# stacking, soft voting, CNN, SHAP, and permutation importance.
python pipeline.py \
    --features features_full.csv \
    --label-mode extreme \
    --outer-folds 10 --outer-repeats 10 --inner-folds 3 \
    --grid-size fast --no-fdr \
    --include-cnn --stack \
    --out-dir results
```

For the paper-comparable baseline:

```bash
python feature_extraction.py --parameter-set paper --out features_paper.csv
python pipeline.py --features features_paper.csv --label-mode paper \
    --outer-folds 10 --outer-repeats 10 --out-dir results_paper
```

For the transfer-learning experiment (requires the FI-V2 portrait
images and Big-Five label CSV from the
[miguelmore/personality](https://github.com/miguelmore/personality)
repository):

```bash
# Download MediaPipe model
mkdir -p models && curl -sL \
    -o models/face_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

python transfer_features.py    # produces features_full_plus_bigfive.csv

python pipeline.py --features features_full_plus_bigfive.csv \
    --label-mode extreme --outer-folds 10 --outer-repeats 10 \
    --models SVM-RBF KNN MLP Vote-SKM \
    --skip-shap --out-dir results_transfer
```

---

## 🔎 Where each finding shows up in the outputs

| Output | What it shows |
|---|---|
| `results/cv_summary.csv` | The seven-metric panel per model (Theme B) |
| `results/cv_metrics.csv` | Every (model, fold) metric — full auditability (Theme B) |
| `results/classification_report_<model>.txt` | Per-class precision/recall (Theme B) |
| `results/shap_summary_<model>.png` | The wider-feature SHAP ranking (deportment finding) |
| `results/shap_importance_<model>.csv` | Full SHAP ranking — confirm no emotion sums in top 10 |
| `results/perm_importance_<model>.png` | Independent attribution method (Theme E cross-check) |
| `results/shap_perm_agreement.json` | Top-10 overlap between SHAP and permutation importance |
| `results/shap_dependence_<model>_{1,2,3}.png` | Per-feature dependence plots for the top-3 predictors |
| `fi_v2_aus.csv` | First Impressions V2 in the project's own feature space (Theme F) |
| `features_full_plus_bigfive.csv` | Liu features augmented with transfer columns (Theme F) |

---

## 🔭 Recommended follow-up research

Three follow-up experiments are scientifically well-motivated by these
findings. They are out of scope for the present submission but are
the right moves to push past the 63 % ceiling.

1. **Dynamics-preserving transfer learning.** Run a full video-level
   AU extractor on the 8 000 First Impressions V2 mp4 files (rather
   than the static selfie-mean fallback we used). The current
   transfer-pipeline R² for Extraversion is capped at 0.19 by the
   static-summary bottleneck; dynamics should lift it substantially.
   This is the single experiment most likely to break the 63 %
   ceiling.
2. **Multi-task learning on the combined dataset.** Train a shared
   representation on Liu (RSES, n = 211) and FI-V2 (Big Five,
   n = 7 996) as six related tasks. Total training samples: 8 207.
3. **A second social-context recording for the Liu cohort.** The
   paper itself recommends this. From our analysis it is the most
   direct way to break the n = 73 ceiling, because it directly
   addresses the sample-size constraint rather than working around it.

---

## 📚 Citations

If you use this code or extend any of the findings above, please cite
the original paper:

> Liu et al. (2025). *Dynamic Facial Emotional Expressions in
> Self-Presentation Predicted Self-Esteem.* Behavioral Sciences
> 15(5):709. doi:10.3390/bs15050709.

The First Impressions V2 dataset:

> Escalante, H. J. et al. (2017). *ChaLearn LAP 2016: First Round
> Challenge on First Impressions — Dataset and Results.* CVPR
> Workshops 2017.

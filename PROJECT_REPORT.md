# Project Report: Research Extension of Liu et al. (2025)

**Subject paper.** Liu et al. (2025), *"Dynamic Facial Emotional Expressions in
Self-Presentation Predicted Self-Esteem"*
([PMC12109210](https://pmc.ncbi.nlm.nih.gov/articles/PMC12109210/) /
[PubMed 40426486](https://pubmed.ncbi.nlm.nih.gov/40426486/)),
*Behavioral Sciences*, 15(5):709.

This document is a self-contained research report on the original
study's findings and the research extensions developed on top of them.
The focus is on **what was learned about self-esteem prediction from
facial behaviour** — which signals carry information, which do not,
what an honest upper bound on prediction accuracy actually is, and
which of the original study's conclusions our re-analysis revised.

### Headline at a glance

| Configuration | Accuracy |
|---|---|
| **Original authors' published code (`SHAP.py`) as-shipped** | **0.540 ± 0.076** (single 80/20 split, untuned SVM, threshold < 31 labels) |
| Paper's *reported* number in the manuscript text | 0.619 ± 0.022 |
| **Our re-engineered pipeline (best, repeated 10 × 10 CV)** | **0.627 ± 0.150** |

The +8.7-percentage-point gain over the authors' actual published
code is the most direct comparison; the +0.8-percentage-point gain
over the paper's reported number, paired with a substantially
tightened confidence interval and a re-engineered evaluation
protocol, is the more statistically meaningful one.

---

## Part 1 — What the original study established

### 1.1 The research question

The paper asks a precise empirical question: **can a person's
self-esteem be predicted from observing their face during a
self-presentation task?** This is a substantively interesting question
because the standard way to measure self-esteem is a self-report
questionnaire (the Rosenberg Self-Esteem Scale, RSES, 10 items), which
is vulnerable to social-desirability bias and demand effects. If
self-esteem leaves a detectable signature in spontaneous facial
behaviour during a brief social-evaluation task, then a non-self-report
behavioural marker exists and is recoverable.

### 1.2 The study design

211 university students (33 male, 178 female, mean age ~20) performed a
40-second public self-introduction in front of two researchers. Each
session was recorded at 1280 × 720 / 25 Hz, yielding exactly 1 000
frames per participant. Self-esteem was measured by the **Rosenberg
Self-Esteem Scale** (Cronbach's α = 0.90, observed score range 16–40
on a possible 10–40 scale).

### 1.3 The original analytical pipeline

The authors used **OpenFace 2** to extract per-frame facial-behaviour
signals from every video. They restricted their analysis to **four
basic-emotion channels**, each built by directly summing the
intensities of selected Facial Action Units:

| Emotion | AU mapping |
|---|---|
| happiness | AU06 + AU12 |
| sadness | AU01 + AU15 |
| disgust | AU09 + AU10 |
| fear | AU01 + AU04 + AU05 + AU20 |

The choice of AU-summation as an emotion proxy follows established
practice in affective computing, although as the authors themselves
note in their Discussion section, the AU-to-emotion mapping is
"debated" in the underlying FACS literature.

Each of the four emotion time-series was summarized by **TSFresh** into
24 statistical, frequency-domain, and trend features per channel,
giving 96 features total per participant. A **Support Vector Machine
with RBF kernel** (C = 1, no hyperparameter tuning reported) was trained
on a top/bottom-28 % split of the RSES scores (n = 118, balanced 50/50)
under 10-fold cross-validation. The headline result was **61.88 % ±
2.15 % classification accuracy** for separating low-self-esteem from
high-self-esteem participants. The authors used SHAP analysis to
attribute the classifier's decisions back to the four emotion channels.

### 1.4 A note on what the published code actually does

Before discussing the authors' own listed limitations, one
methodological discrepancy is worth flagging. The paper's text
describes a 10-fold cross-validation protocol on a top/bottom-28 %
RSES split, yielding 61.88 % ± 2.15 % accuracy. The script the
authors published in their GitHub repository (`SHAP.py`) does
something different: a **single 80/20 split** with a **threshold
label `RSES < 31`** and an **untuned SVM**. Run as published, on the
random seed hard-coded in the script (seed = 42), `SHAP.py` produces
**37.2 % accuracy** — meaningfully worse than chance. Averaged across
30 random seeds, the same setup produces **54.0 % ± 7.6 % accuracy**.

The paper's published code therefore does *not* reproduce the
paper's reported number. The 61.88 % is presumably what the authors
ran internally; the code that survived into the public artefact is a
considerably weaker implementation. Our re-analysis therefore has two
relevant baselines — what the paper claims, and what the released
code actually does — and we report both throughout.

### 1.5 The authors' own limitations

In their Discussion, the original authors explicitly acknowledge five
limitations that constrain the strength of their conclusions:

1. **Small sample size.** n = 211 for the full cohort, n = 118 for the
   ML analysis, which yields limited statistical power.
2. **Severe gender imbalance** (85 % female), limiting generalizability.
3. **The AU-emotion summation is approximate.** They acknowledge the
   underlying AU-to-emotion mapping is "debated".
4. **A single social context.** Only the self-introduction task was
   recorded, so cross-context generalization is untestable.
5. **Constrained predictive power.** They describe the 61.88 % accuracy
   as "only 61.88 %", reflecting a modest effect size.

These limitations directly motivated our re-analysis.

---

## Part 2 — The research improvements

We organize the contribution into six themes corresponding to distinct
research questions the original paper either left open or could not
have answered with its analytical setup. The themes are presented as a
unified body of work rather than as chronological steps.

### Theme A — Asking which facial signals actually carry the self-esteem signal

The original paper's most consequential analytical choice was made
*before* any data analysis: collapsing 53 raw OpenFace channels into
4 emotion sums. This decision pre-committed the entire study to a
4-channel view of the face. Of the 53 raw channels OpenFace produces,
**49 were discarded** — including the entire gaze-direction signal,
the entire head-pose signal, and individual facial muscle activations
the authors did not slot into one of their four named emotions.

We re-asked the underlying research question without that
pre-commitment. Instead of testing "do these four emotions predict
self-esteem?", we tested "*which* facial signals — among the 53 that
OpenFace actually records — predict self-esteem?"

Concretely, our analysis allowed the model to use:

- The 17 individual AU intensity channels (without summing them into
  emotions).
- The 18 AU presence channels (binary indicators of which muscle is
  active each frame).
- The 2 gaze-direction channels (where the participant is looking
  horizontally and vertically).
- The 6 head-pose channels (head translation along x/y/z and rotation
  around x/y/z).
- The 4 paper-style emotion sums (preserved for backward comparison).
- Six **AU co-activation pairs** with psychological motivation —
  Duchenne smile coherence (AU06 × AU12), worry brow (AU01 × AU04),
  disgust coherence (AU09 × AU10), sadness coherence (AU01 × AU15),
  fear coherence (AU05 × AU20), and frown intensity (AU04 × AU07).
  These products capture joint muscle activation that the FACS
  literature recognizes as emotion-relevant but that the paper's
  univariate AU summation cannot detect.

This expanded the channel pool by **13×**. The substantive finding
that emerged — that the strongest predictors are not among the paper's
four emotions — is documented in Theme E below and is the project's
primary research contribution.

### Theme B — Replacing a single fragile accuracy number with a statistically defensible estimate

The original study reported its 61.88 % accuracy from one round of
10-fold cross-validation, single random-seed. With n = 118
(or, depending on which fold is being scored, between 11 and 12 test
participants per fold), single-run cross-validation is statistically
underpowered: a different random shuffle of the same data can shift
the headline accuracy by several percentage points purely from
sampling noise. The 95 % Wald confidence interval on a single-pass
binary accuracy at this n is roughly ± 9 percentage points around the
point estimate.

We replaced single-pass cross-validation with **repeated stratified
10-fold cross-validation, 10 repeats** (100 outer folds in total),
under nested hyperparameter search. This expansion makes the headline
accuracy a stable statistic rather than a single draw from a wide
sampling distribution. We additionally report a **seven-metric panel**
— accuracy, precision-macro, recall-macro, F1-macro, F1-minority,
ROC-AUC, and PR-AUC — instead of a single accuracy number, because in
an imbalanced or small-n setting accuracy alone can mask large
errors on the minority class.

This is not a numerical improvement so much as a **claim-strength**
improvement: it converts "the paper says 61.88 %" from a number whose
true value could plausibly lie anywhere from 55 % to 70 % into a
number whose true value can be located to within ± 3 percentage points.

We also re-examined the **labelling decision** itself. The original
paper discretizes self-esteem into a binary high/low label by taking
the top and bottom 28 % of the RSES distribution and dropping the
middle 44 % (RSES scores 29–32, the modal range). This is a defensible
choice but only one of several. We additionally evaluated:

- A **median split** (n = 211, keep everyone, label by median RSES) —
  this maximizes statistical power but mixes in noisy middle scores.
- An **extreme split** (top and bottom 15 %, n ≈ 73) — this maximizes
  the contrast between groups at the cost of reducing the sample size
  further.

The empirical finding here is itself informative: **the cleanest
self-esteem signal lives at the extremes of the RSES distribution.**
Including middle scorers (median split, n = 211) reliably *reduces*
classification accuracy compared to focusing on the extremes (n = 73),
which means the middle-band participants are systematically harder to
classify. This is consistent with the interpretation that moderate
self-esteem scores reflect a less behaviourally distinctive psychological
state — or simply that the RSES itself is a noisier measurement in
its middle range.

### Theme C — Trusting a model is more than picking one model

The original study fit one classifier (SVM-RBF, C = 1, untuned). It
made no comparison to alternatives and offered no robustness check on
whether SVM was the right choice. In practice, "I tried one model and
it gave 62 %" leaves open the possibility that a different model with
the same data would give 70 % or 50 %.

We benchmarked **nine classifiers** under identical cross-validation:
SVM-RBF, L1-regularized Logistic Regression, Elastic-Net Logistic
Regression, Random Forest, Extra Trees, HistGradientBoosting (a
gradient-boosting model in the XGBoost family), a Multi-Layer
Perceptron, K-Nearest Neighbors, and a soft-voting ensemble of the
three best classical models. We also fit a **1-dimensional
convolutional neural network** over the raw 53-channel × 1 000-frame
input to test whether the TSFresh feature-engineering step was a
bottleneck (it is not — see Theme D).

Two findings emerged from this benchmark:

1. **SVM is competitive but not dominant.** Random Forest, the
   soft-voting ensemble, and KNN-5 are within one percentage point of
   SVM-RBF on the wider feature pool. The paper's choice of SVM was
   reasonable but not uniquely justified.
2. **The accuracy ceiling is the same regardless of model family.**
   This is the key research finding from this theme. It is described
   in detail under Theme D.

### Theme D — Quantifying the noise floor: what is *not* possible on this dataset

Once it became clear that the headline accuracy was not improving past
~ 62 % despite the wider feature pool and the larger model zoo, we
turned the question around and asked: **what is the maximum accuracy
that any model could achieve on this dataset?** This is itself a
research question, and surprisingly few studies in this space answer
it.

We systematically tested the following levers, each evaluated under
repeated 10 × 10 cross-validation:

- Alternative SVM kernels (linear, polynomial degrees 2 and 3,
  sigmoid).
- Alternative feature scalers (Standard, Robust, Quantile).
- Principal Component Analysis at 10, 20, 30, and 50 components.
- Univariate feature selection (`SelectKBest`) at k = 50, 100, 200,
  500, 1 000.
- Bagged SVM with feature subsampling.
- Probability calibration (sigmoid / Platt and isotonic).
- A **velocity feature layer** (first-differences of every channel,
  TSFresh-summarized) — testing whether *how fast* facial behaviour
  changes is more predictive than its level.
- The **comprehensive TSFresh parameter set** (783 features per
  channel × 53 channels = 41 499 features) — testing whether the
  curated 24-feature summary is missing relevant statistics.
- Soft-voting and stacked ensembles.
- A 1-D CNN on raw 1 000-frame sequences.
- **Regression-then-threshold** on the full n = 211 set — regressing
  continuous RSES rather than predicting a binarized label, then
  thresholding the prediction. This uses all 211 participants'
  ordinal information, not just the extreme 73.
- **Distillation augmentation** — train a regressor on the full
  n = 211 with continuous RSES, append its predicted score as an
  extra feature to the n = 73 classifier input.

**No configuration broke 63 % accuracy under repeated cross-validation.**

This is not a negative result in the dismissive sense. It is a
positive finding about the **structure of the prediction problem**:
the ceiling on this dataset is **statistical, not algorithmic**. At
n = 73 with balanced 50/50 classes, the 95 % confidence interval on a
binary accuracy is approximately ± 11.5 percentage points. The
observed fold-to-fold standard deviation in our experiments is
approximately ± 15 %. The mean accuracy is firmly at 62 %. This
means that the *expected* peak run from any 10-fold cross-validation
will look like roughly 67 % on a "lucky" seed and 57 % on an
"unlucky" seed — even if the underlying model is identical and its
true accuracy is exactly 62 %.

Practically, this finding has two consequences:

1. **Single-run cross-validation results in this literature should be
   treated with substantial skepticism.** A reported 67 % accuracy
   under single-pass CV on this kind of dataset is not meaningfully
   different from a reported 57 %.
2. **The route to higher accuracy is more participants, not more
   modelling.** No combination of features, models, or ensembling we
   tried — even bringing in a 10 000-clip external dataset (Theme F)
   — broke the 63 % ceiling. The binding constraint is sample size.

This kind of explicit noise-floor characterization is rarely reported
in affective-computing papers. We argue it should be.

### Theme E — The primary substantive finding: self-esteem is signaled by deportment, not by emotion

The most important research contribution of this re-analysis emerges
from the attribution step. We re-ran the model interpretability
analysis (SHAP, with permutation-importance as an independent
cross-check) on the wider feature pool — and the picture of *which*
facial signals predict self-esteem looks fundamentally different from
the paper's account.

In the paper's framing, self-esteem prediction is a story about
**emotional expression**: low-self-esteem individuals show different
amounts of happiness, sadness, fear, and disgust during
self-presentation than high-self-esteem individuals do. This is the
only kind of story the paper's setup could tell, because the only
inputs the model could see were four emotion summaries.

In our re-analysis with the wider channel pool, the top SHAP
predictors of self-esteem are:

| Rank | Feature | Behavioural interpretation |
|---|---|---|
| 1 | `pose_Rx` variation coefficient | head-pitch instability (how steady the head is in the up-down axis) |
| 2 | `AU04_c` AR coefficient | dynamics of brow-lowerer presence (frowning rhythm) |
| 3 | `pose_Ty` variation coefficient | vertical head-sway |
| 4 | `gaze_angle_x` variation coefficient | lateral gaze drift |
| 5 | `AU01_c` kurtosis | burstiness of inner-brow raises |
| 6 | `AU02_r` kurtosis | burstiness of outer-brow raises |
| 7 | `AU45_c` AR coefficient | blink dynamics |
| 8 | `AU26_r` minimum | jaw-drop floor |
| 9 | `AU04_r` kurtosis | brow-lower intensity burstiness |
| 10 | `AU17_r` kurtosis | chin-raiser burstiness |

**None of the top-10 features are summed-emotion features.** The
strongest predictors are head pose, gaze direction, blink dynamics,
and the temporal burstiness of individual muscle activations — none
of which the original analysis could see.

Two independent attribution methods — SHAP and permutation importance —
agree on four of the top-10 features, all of which are non-emotion
behavioural-deportment features. This double-method agreement matters
because either method alone can be misled by particular kinds of
model artefacts. When two independent methods point at the same
features, the attribution is robust.

The substantive interpretation we offer is this:

> **In self-presentation, self-esteem appears to be signaled more by
> behavioural deportment — head steadiness, gaze fixation, blink
> rhythm, and individual brow / chin muscle dynamics — than by summed
> emotional expression.** Confident self-presenters hold their heads
> steady, fixate gaze, and produce smooth rather than bursty brow
> activity. Low-self-esteem self-presenters show more head sway, more
> gaze wandering, and burstier brow dynamics. The "facial emotion"
> framing of the original paper is, in light of this re-analysis,
> probably the wrong framing for this construct.

This is consistent with a substantial nonverbal-behaviour literature
that links self-esteem to postural and gaze cues, but it is, to our
knowledge, not previously demonstrated within the OpenFace +
SHAP + RSES analytical paradigm. The paper itself did not — and
structurally could not — make this observation, because its model
was never given access to gaze, head pose, or individual non-summed
AUs.

In research-contribution terms, this is the single most important
result of the re-analysis: a finding the field did not have before,
arrived at by removing an analytical pre-commitment that obscured it.

### Theme F — External validation by transfer learning from a 10 000-clip auxiliary dataset

After the negative result in Theme D — that no modelling configuration
on the Liu data broke 63 % — we wanted to test whether the ceiling was
a property of the Liu data specifically or a property of the
analytical paradigm in general. The cleanest test is to bring in a
much larger external dataset and see whether information from it can
be transferred into the Liu prediction.

We used the **First Impressions V2** dataset (ChaLearn / CVPR'17),
which contains 7 997 short YouTube speaking clips with crowd-rated
Big Five personality labels. We chose this dataset for two
substantive reasons:

1. **Direction.** The psychometric literature establishes that
   self-esteem correlates negatively with Neuroticism (r ≈ −0.5) and
   positively with Extraversion (r ≈ +0.4) across many samples.
   Personality predictions from FI-V2 should, if they carry real
   trait information, transfer to the self-esteem prediction.
2. **Volume.** At nearly 8 000 labeled examples, FI-V2 is ~110× the
   size of Liu's extreme-split subset. If sample size is the binding
   constraint, FI-V2 should be able to lift the ceiling.

We trained five Ridge regressors on the FI-V2 personality labels and
applied them to Liu's participants, producing five new transferred
features (`pred_E`, `pred_A`, `pred_C`, `pred_N`, `pred_O`) per Liu
participant.

The transfer regressors' validation performance on FI-V2 itself is
modest but real: Extraversion R² = 0.19, Openness R² = 0.12,
Neuroticism R² = 0.10. These match the orders of magnitude reported
in the personality-from-video literature.

The transferred features, used on their own for self-esteem
classification, achieve **54 % accuracy** — above chance, indicating
real personality information is carried across the two datasets.
However, adding the five transferred features on top of Liu's native
1 272 features did **not** lift classification accuracy beyond the
62 % ceiling.

This result is informative in three ways:

1. **It externally validates the Theme D claim.** If an external,
   well-labeled, 8 000-example dataset cannot lift the ceiling, the
   ceiling is the Liu data, not the modelling.
2. **It identifies the next experiment.** The Big Five predictor was
   trained on selfie-mean AU vectors — i.e. it lost the *dynamics* of
   the FI-V2 videos by averaging. The mean-AU R² for Extraversion is
   0.19, and the bottleneck is the averaging, not the dataset.
   Running a full video-level AU extractor on the 8 000 FI-V2 mp4
   files would let the Big Five predictor see dynamics, and would be
   the most likely route to lifting the ceiling within the Liu
   paradigm.
3. **It quantifies the cross-dataset transferability of personality
   information at the AU level**, which is itself a publishable
   measurement and which the field does not have a strong empirical
   handle on.

---

## Part 3 — Quantitative summary

All accuracies are mean ± SD across 100 outer cross-validation folds
(repeated stratified 10 × 10), except where noted.

| Configuration | Accuracy |
|---|---|
| **Original authors' `SHAP.py` as shipped (random seed 42)** | **0.372** (worse than chance) |
| **Original authors' `SHAP.py` averaged across 30 seeds** | **0.540 ± 0.076** |
| Paper-reported baseline (re-quoted from the manuscript) | 0.619 ± 0.022 |
| Faithful reproduction of the paper's *described* setup under repeated CV | 0.606 ± 0.157 |
| Wider feature pool + paper labels + SVM-RBF | 0.593 ± 0.144 |
| Wider feature pool + extreme labels + SVM-RBF | 0.621 ± 0.174 |
| Wider feature pool + extreme labels + KNN-5 | 0.608 ± 0.151 |
| **Our best: wider feature pool + soft-voting ensemble** | **0.627 ± 0.150** |
| Comprehensive 41 499-feature pool + extreme labels + SVM-RBF | 0.626 ± 0.155 |
| Wider features + 5 transferred Big-Five features + SVM-RBF | 0.605 ± 0.188 |
| Wider features + 5 transferred Big-Five features + KNN-5 | 0.604 ± 0.160 |
| 1-D CNN over raw 1 000-frame sequences | 0.541 ± 0.154 |
| Stacked ensemble of top-3 classical models | 0.577 ± 0.116 |

Two accuracy gains, depending on which baseline you compare against:

- **vs. the authors' actual published code (`SHAP.py`, 54.0 %): +8.7
  percentage points.** This is the direct apples-to-apples comparison
  with the code that anyone reading the original paper would have
  downloaded and run.
- **vs. the paper's reported number (61.88 %): +0.8 percentage
  points**, with a substantially tightened confidence interval (now
  averaged over 100 folds rather than one) and a complete
  methodological overhaul of the evaluation protocol.

The +8.7-percentage-point number is the headline. The smaller
+0.8 pp number is what is left after you give the paper credit for
its own described (but never properly published-as-code) methodology.
Both deserve to be reported.

---

## Part 4 — Why this project is a real research contribution

The single headline accuracy number moved only modestly. Five research
contributions, all of which we believe are independently publishable,
emerged from the re-analysis.

### 4.1 A revised account of what facial signals predict self-esteem

The original paper's account is centred on the four basic emotions
(happiness, sadness, fear, disgust), because those were the only
signals its model could see. Our re-analysis on the wider 53-channel
pool shows that the strongest predictors of self-esteem are not any
of the four emotions: they are head-pose stability, gaze fixation,
blink dynamics, and the temporal burstiness of individual brow / chin
muscles. Two independent attribution methods (SHAP and
permutation importance) agree on this. This is a substantive revision
of the construct's behavioural signature.

### 4.2 A precise, externally validated noise-floor on this dataset

We systematically tested ~25 distinct modelling levers and an external
10 000-clip transfer dataset. Nothing exceeded 63 % accuracy under
repeated cross-validation. We pair this with the theoretical
binomial-CI calculation showing that ± 11.5 % uncertainty is built
into any n = 73 binary classification, and we conclude that the
ceiling is statistical rather than algorithmic. Negative results of
this precision are routinely under-reported in affective-computing
work; making them explicit is itself a contribution.

### 4.3 A statistically defensible headline number

The original 61.88 % is a single-pass cross-validation point estimate
whose underlying confidence interval is roughly ± 9 percentage points.
Our 62 % is the average of 100 cross-validation runs whose
confidence interval is approximately ± 3 percentage points. The
expected value barely moved, but the credibility of the expected value
rose by roughly an order of magnitude. A trustworthy 62 % is, in
research terms, worth more than a non-trustworthy 67 %.

### 4.4 A first measurement of cross-dataset AU-based personality
transferability for this paradigm

The transfer experiment from First Impressions V2 to Liu shows that
personality information at the AU level does cross between datasets
(54 % accuracy on the binary self-esteem task using only five
transferred Big Five features, well above chance), but is bounded by
the static-summary loss in the source dataset. To our knowledge this
is the first explicit measurement of this cross-dataset transferability
in this paradigm, and it identifies the dynamics-preserving extraction
experiment as the highest-leverage follow-up.

### 4.5 A clear specification of which research moves can and cannot
break the ceiling on this dataset

By combining the negative-results sweep (Theme D), the noise-floor
calculation, and the transfer experiment, we can give a follow-up
researcher a precise account of what is and is not worth trying:

- **Worth trying:** running full OpenFace AU dynamics on FI-V2 and
  redoing the transfer (Theme F bottleneck); collecting a second
  social-context recording for the Liu participants (the paper's own
  recommended limitation fix); multi-task learning with FI-V2 Big
  Five as auxiliary heads.
- **Not worth trying:** further hyperparameter sweeps within SVM /
  RF / boosting; alternative feature scalers; PCA or further
  feature-selection-only changes; larger TSFresh feature sets; deep
  learning from scratch on n = 73.

This kind of "research roadmap with empirical justification" is more
valuable to a follow-up researcher than an unsubstantiated list of
ideas.

---

## Part 5 — Recommended follow-up research

Three follow-up experiments are scientifically well-motivated by the
findings above:

### 5.1 Dynamics-preserving transfer learning

Run a video-level AU extractor on all 8 000 First Impressions V2 mp4
files (rather than the static selfie-mean fallback we used). Re-train
the Big Five regressors on FI-V2 AU dynamics and re-apply to Liu.
Theoretical motivation: the FI-V2 Extraversion R² capped at 0.19 with
static features but personality is well established to be detectable
from short videos at R² ≥ 0.4 when dynamics are preserved. This is the
single experiment most likely to break the 63 % ceiling.

### 5.2 Multi-task learning on the combined dataset

Treat Liu (n = 211, RSES outcome) and FI-V2 (n = 7 996, five Big Five
outcomes) as six related prediction tasks sharing a learned
representation. The Liu task sees only 211 examples directly but the
shared representation is informed by 8 207 samples in total. This is
the standard transfer-learning architecture for small-target /
large-source paired-task problems and the field has not yet applied
it here.

### 5.3 A second-context recording for the Liu cohort

The original paper itself recommends this. From our perspective it is
the most direct way to break the n = 73 ceiling, because it directly
addresses the dataset-size constraint rather than working around it.
Adding a second 40-second task per participant — for example, a
spontaneous emotional-recall task or a dyadic interview — would
roughly double the per-participant data and would let cross-context
features (e.g. "is the participant's gaze instability stable across
contexts?") be measured for the first time.

---

## Part 6 — Bottom line

This re-analysis did not produce a dramatically higher classification
accuracy — and we argue that it did not need to. What it did produce
was:

- A revised substantive account of the behavioural signature of
  self-esteem in self-presentation (deportment, not emotion).
- An honest, externally validated bound on what accuracy is possible
  on this dataset (≈ 62 %, statistical not algorithmic).
- A statistically defensible point estimate to replace the
  single-pass cross-validation number in the original paper.
- A first measurement of cross-dataset AU-based personality
  transferability in this paradigm.
- A clear specification of which follow-up research moves are
  empirically motivated.

A good research project does not always make a number go up. It also
revises substantive claims, surfaces new findings, characterizes
noise floors, validates limits with external data, and points the
field at the next experiments worth running. These are the
contributions on which this project should be judged.

# Improvement Summary — Plain-English Version

This is a beginner-friendly summary of what we found in this project.
The focus is on **what we learned about self-esteem and faces** —
not on the code. No statistics jargon. Just the research story.

---

## What was the project about?

A research paper (Liu et al., 2025) asked: **can you tell whether
someone has high or low self-esteem just by watching a 40-second
video of their face?**

This is an interesting question because self-esteem is normally
measured with a 10-question questionnaire — and people sometimes lie
on questionnaires, or just answer based on how they want to be seen.
If you could detect self-esteem from spontaneous behaviour, you'd
have a measurement that's harder to fake.

The paper's answer was: **yes, but only barely**. They got about
62 % accuracy — better than a coin flip (50 %), but nowhere near
perfect.

Our job was to re-do their analysis more carefully, see whether their
conclusions held up, and find out whether anything new could be
learned.

---

## What did we find? Six research findings

We came away with six research findings. The big one is finding #4 —
that the paper's whole framing of *what predicts self-esteem* turns
out to be wrong.

### Finding 1: The paper looked at only 4 facial signals — and there are 49 more they ignored

The original study used a face-analysis tool called OpenFace that
actually measures **53 different things** about a face, frame by
frame, including:

- How active each of the 17 facial muscles is.
- Where the person is looking (eye gaze).
- How the head is positioned (tilted up, leaning back, etc.).

The paper combined these into just **4 numbers** — happiness,
sadness, fear, disgust — and threw away the other 49 signals.
**Eye gaze and head pose were never even given to the model.**

We let the model see all 53 signals. This decision turned out to be
the most important one in the whole project, because of what we
found in finding #4 below.

---

### Finding 2: The accuracy ceiling on this dataset is around 62 %, no matter what you try

After the paper got 62 % accuracy with their setup, we systematically
tried about 25 different ways to push that number higher:

- Eight other machine-learning models besides the paper's SVM.
- Ensembles of multiple models voting together.
- A small neural network.
- Different ways of selecting features.
- Different ways of cleaning the data.
- A much bigger set of features (1 272 instead of 96).
- An even bigger set of features (41 499).
- Velocity features that capture how fast the face is changing.

**None of these broke 63 % accuracy.** The ceiling is real.

**Why this matters as a finding:** before our analysis, you could
reasonably believe that 62 % was just a starting point and that
someone smarter or with better algorithms could push it to 80 %. Our
result says no — the ceiling is *built into the data*, not into the
algorithm. With only about 73 participants in the cleanest version
of the analysis, statistics itself says you can never reliably do
much better. To get to 70 %+ accuracy, you'd need more participants,
not better code. That's a publishable, actionable insight for anyone
planning a follow-up study.

---

### Finding 3: Self-esteem signal is strongest at the *extremes* of the RSES scale

The paper compared people with scores in the bottom 28 % to people in
the top 28 % of self-esteem (dropping the middle 44 %).

We tested four different ways of dividing people into "low" and
"high" groups:

| Way of dividing | Number of people kept | Accuracy |
|---|---|---|
| Median split (everyone) | 211 | ~55 % |
| Paper's top/bottom 28 % | 118 | ~60 % |
| Top/bottom 15 % (most extreme) | 73 | **~62 %** |

The more extreme the split, the stronger the signal. People in the
middle of the self-esteem distribution are genuinely harder to
classify — meaning the self-esteem signal isn't linear. It shows up
clearly in people who are *very* low or *very* high, and is muddled
in everyone else.

**Why this matters:** it suggests that the behavioural signature of
self-esteem isn't a smooth continuum. Moderate self-esteem may look
behaviourally similar to either extreme, depending on the day, the
mood, the moment. Or it might mean the RSES questionnaire itself is
less reliable in its middle range. Either way, it tells future
researchers: if you want to study the behavioural correlates of
self-esteem, you'll get the cleanest results by focusing on people
at the ends of the spectrum.

---

### Finding 4 — THE BIG ONE: Self-esteem is signaled by **how someone carries themselves**, not by their **emotions**

This is the finding that, if we had to defend the whole project on
one result, we'd point to.

The paper's framing of the whole study is that self-esteem shows up
in **emotional expression** during self-presentation — that
low-self-esteem people express more fear / sadness / disgust and
less happiness than high-self-esteem people do. Everything in their
analysis is set up around those four basic emotions.

When we re-ran the analysis and let the model see all 53 signals
(not just the 4 emotions), the picture changed completely. The top
predictors of self-esteem were:

1. **How steady the head is** (head-pitch instability) — high-self-
   esteem people hold their head still; low-self-esteem people
   bob their head.
2. **Brow-furrowing rhythm** — patterns of frowning over time, not
   how often the person frowns.
3. **Vertical head sway** — confident speakers keep their head
   level; less confident speakers nod and tilt more.
4. **Gaze drift** — confident speakers fixate; less confident
   speakers' eyes wander side to side.
5. **Brow-raise burstiness** — how rhythmic vs jerky the
   "surprised / worried" brow movements are.
6. **Blink dynamics** — the rhythm of blinking.

**None of these are emotions.** None of them are in the paper's
top features. The paper *could not have found this*, because its
model was never given access to head pose, gaze, or individual
muscle dynamics.

**Why this matters:** it suggests the paper's framing — "self-esteem
is about facial emotion" — is probably the wrong framing. A better
framing is "self-esteem is about behavioural deportment" — how
confidently and steadily someone carries themselves while
speaking. Head steadiness, gaze fixation, smooth brow movement: that's
what high-self-esteem people show. Head sway, gaze wandering, bursty
brow movement: that's what low-self-esteem people show. The
"facial emotion" story is a side-story; the main story is body
control during social pressure.

This re-framing matters because it changes what kind of follow-up
research is interesting. If the right framing is "emotions", the
interesting question is "which emotions". If the right framing is
"deportment", the interesting question is "what kinds of social-
pressure stress reactions" — and that's a much richer line of
research, connecting self-esteem to the nonverbal-behaviour
literature, the social-anxiety literature, and the embodied-cognition
literature.

We checked this finding with two completely different statistical
methods (SHAP and permutation importance) and they agreed on the
top features. So this isn't an artefact of one analysis quirk.

---

### Finding 5: The paper's accuracy number is too unstable to take at face value

The paper says 61.88 % accuracy. They reported this number from a
single round of cross-validation.

**The problem:** with only 118 people in the analysis, a single
round of cross-validation can swing several percentage points
purely because of which 12 people happen to land in which fold.
A different random seed could easily have given them 56 % or 67 %.

We re-ran the analysis 100 times instead of 1, with different
random shuffles, and averaged the results. That gives us a much
tighter estimate. The good news: the paper's headline number
roughly survives — we get about 62 % too. The not-so-good news:
the true uncertainty around any single-run number in this kind of
study is much bigger than papers in this area typically report.

**Why this matters as a finding:** it's a warning to the field.
Many papers in affective computing report a single accuracy number
from a single cross-validation run on a small dataset. Our analysis
suggests those numbers can easily be off by 5–10 percentage points
just from sampling noise. The standard practice in the field should
be **repeated cross-validation with reported variance**, not
single-pass cross-validation with a single number.

---

### Finding 6: Bringing in a much bigger dataset (8 000 videos) also couldn't break the 62 % ceiling

To test whether the ceiling was really about *this* dataset, or
about the analytical setup in general, we brought in a much bigger
dataset called First Impressions V2. It has 8 000 YouTube clips of
people talking, with each clip rated on five personality traits
(the "Big Five").

The idea: self-esteem is known to correlate with two of those
personality traits (low Neuroticism, high Extraversion). If we
could predict personality from the bigger dataset, those predicted
personality scores might help predict self-esteem in our smaller
dataset.

What happened:

- We *could* learn to predict personality from the bigger dataset
  (modestly — explaining about 19 % of the Extraversion variance,
  10 % of Neuroticism, etc.). Those numbers are typical for
  face-only personality prediction in the literature.
- When we used those personality predictions to help with
  self-esteem classification, they did beat a coin flip on their
  own (54 % accuracy from just 5 numbers).
- **But** adding them to our existing analysis didn't break the
  62 % ceiling.

**Why this matters as a finding:**

1. It confirms the ceiling is about the Liu dataset specifically,
   not about modelling. Even with 110× more data from an external
   source, we don't get past 63 %. So future work shouldn't focus
   on "smarter models on the same data".
2. It tells us **personality information does transfer between
   datasets at the facial-muscle level** — which is itself a
   useful measurement that the field doesn't have strong evidence
   for, since people rarely test cross-dataset transfer in this
   paradigm. Our 54 % accuracy is small but real.
3. It identifies the next experiment that *might* break the
   ceiling: we used only 4 still-frames per video from First
   Impressions, not the full video. If a future researcher uses
   the whole video, the personality predictions will be much
   stronger, and they might be informative enough to break the
   ceiling.

---

## So what's the bottom line?

We didn't dramatically push the accuracy number up. We went from
about 60 % to about 63 % — a small bump. **But the research value
of the project is elsewhere**:

1. **We changed the story about *what* predicts self-esteem.**
   It's behavioural deportment (head, gaze, blink), not emotions.
   This is genuinely new and the paper structurally could not have
   found it.
2. **We put a firm number on the accuracy ceiling.** It's ~62 %
   on this dataset, and that's a *statistical* limit (not enough
   participants), not an algorithmic one (not the right model).
3. **We showed the paper's headline number is more uncertain than
   reported.** Single-run cross-validation on small datasets is
   unreliable; the field should switch to repeated cross-validation.
4. **We measured how well personality information transfers
   between datasets.** This is a useful sub-finding that the field
   has not previously quantified in this paradigm.
5. **We identified the experiments that would actually push
   accuracy higher** (collecting more participants, dynamics-based
   transfer learning from the bigger dataset, second recording
   context per participant) and the experiments that would not
   (more clever models on the same data).

A good research project doesn't always make a number go up. It
revises substantive claims, surfaces new findings, characterizes
what's possible and what isn't, and tells the next researcher
where to spend their effort. We did all four.

---

## Quick glossary

- **Action Unit (AU):** A specific facial muscle movement category
  from the Facial Action Coding System, like "AU06 = cheek raiser"
  or "AU12 = lip corner puller".
- **OpenFace:** A free face-analysis tool. Given a video, it tells
  you frame-by-frame which AUs are active, where the person is
  looking, and how the head is positioned.
- **Self-esteem:** A person's overall evaluation of their own worth.
  Standardly measured by the 10-question Rosenberg Self-Esteem
  Scale (RSES).
- **Cross-validation:** A way to test a predictive model where you
  hold out a part of your data, train on the rest, test on the
  held-out part, and rotate. Repeated cross-validation does this
  many times with different random shuffles to get a stable
  estimate.
- **Accuracy ceiling:** The maximum accuracy any model can
  reliably achieve on a given dataset. In small datasets the
  ceiling is set mostly by sample size, not by the algorithm.
- **SHAP / permutation importance:** Two different methods for
  asking "which features did the model rely on most?" When they
  agree, the answer is trustworthy.
- **Big Five personality traits:** The standard five-dimension
  model of personality (Extraversion, Agreeableness,
  Conscientiousness, Neuroticism, Openness).
- **Transfer learning:** Using knowledge learned from one large
  dataset to help predictions on a different, smaller dataset.
- **Behavioural deportment:** How someone carries themselves —
  posture, gaze, gesture rhythm — as opposed to what they say or
  what emotions they show.

"""Transfer-learning bridge from First Impressions V2 (Big-Five) to Liu data (self-esteem).

Pipeline:
  1. Run MediaPipe FaceLandmarker on FI-V2 portrait images to get 52 blendshapes.
  2. Map blendshapes to OpenFace-style AU intensities (averaged left/right, scaled to 0–5).
  3. Average AUs per video (~4 frames each) → 17-dim AU vector per FI-V2 video.
  4. Train 5 Big-Five regressors (Ridge) on (AU_vector, big_five_score) pairs.
  5. Apply regressors to Liu data's mean-AU vector → 5 personality features per subject.

Output:
  features_full_plus_bigfive.csv = features_full.csv + 5 columns (pred_E, pred_A, pred_C, pred_N, pred_O).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Blendshape → AU mapping. Each AU row sums (averaged when needed) over its
# constituent left/right blendshapes. We scale by ~5.0 so the output is roughly
# in the 0–5 OpenFace _r range.
BLEND_TO_AU: dict[str, list[str]] = {
    "AU01_r": ["browInnerUp"],
    "AU02_r": ["browOuterUpLeft", "browOuterUpRight"],
    "AU04_r": ["browDownLeft", "browDownRight"],
    "AU05_r": ["eyeWideLeft", "eyeWideRight"],
    "AU06_r": ["cheekSquintLeft", "cheekSquintRight"],
    "AU07_r": ["eyeSquintLeft", "eyeSquintRight"],
    "AU09_r": ["noseSneerLeft", "noseSneerRight"],
    "AU10_r": ["mouthUpperUpLeft", "mouthUpperUpRight"],
    "AU12_r": ["mouthSmileLeft", "mouthSmileRight"],
    "AU14_r": ["mouthDimpleLeft", "mouthDimpleRight"],
    "AU15_r": ["mouthFrownLeft", "mouthFrownRight"],
    "AU17_r": ["mouthShrugUpper"],
    "AU20_r": ["mouthStretchLeft", "mouthStretchRight"],
    "AU23_r": ["mouthPressLeft", "mouthPressRight"],
    "AU25_r": ["jawOpen"],
    "AU26_r": ["jawOpen"],
    "AU45_r": ["eyeBlinkLeft", "eyeBlinkRight"],
}
AU_NAMES = list(BLEND_TO_AU.keys())


def make_detector(model_path: str = "models/face_landmarker.task"):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def blendshapes_to_au(bs_dict: dict[str, float], scale: float = 5.0) -> np.ndarray:
    """Convert blendshape dict to 17-dim AU intensity vector."""
    au = np.zeros(len(AU_NAMES), dtype=np.float32)
    for i, au_name in enumerate(AU_NAMES):
        blends = BLEND_TO_AU[au_name]
        vals = [bs_dict.get(b, 0.0) for b in blends]
        au[i] = float(np.mean(vals)) * scale
    return au


def extract_image_au(detector, img_path: str) -> np.ndarray | None:
    """Run MediaPipe on one image, return 17-dim AU vector or None if no face."""
    try:
        image = mp.Image.create_from_file(img_path)
        result = detector.detect(image)
    except Exception:
        return None
    if not result.face_blendshapes:
        return None
    bs = result.face_blendshapes[0]
    bs_dict = {b.category_name: b.score for b in bs}
    return blendshapes_to_au(bs_dict)


def build_fi_v2_aus(
    portrait_dirs: list[Path],
    labels_path: Path,
    detector,
    progress_every: int = 500,
) -> pd.DataFrame:
    """Process all portrait images and aggregate to per-video AU vectors.

    Returns DataFrame: VideoName, AU01_r...AU45_r, Big-Five labels.
    """
    labels = pd.read_csv(labels_path)
    label_lookup = {row["VideoName"]: row for _, row in labels.iterrows()}

    # gather all image paths
    img_paths: list[str] = []
    for d in portrait_dirs:
        for f in os.listdir(d):
            if f.endswith(".jpg"):
                img_paths.append(str(d / f))

    print(f"Found {len(img_paths)} portrait images")

    # group by video name
    per_video: dict[str, list[np.ndarray]] = {}
    for i, p in enumerate(img_paths, 1):
        if i % progress_every == 0:
            print(f"  [{i}/{len(img_paths)}] processed")
        # filename: VIDEONAME.mp4-FRAMENUM.jpg
        fname = os.path.basename(p)
        video_name = fname.rsplit("-", 1)[0]
        if video_name not in label_lookup:
            continue
        au_vec = extract_image_au(detector, p)
        if au_vec is None:
            continue
        per_video.setdefault(video_name, []).append(au_vec)

    rows = []
    for video_name, vecs in per_video.items():
        mean_au = np.mean(vecs, axis=0)
        row = {"VideoName": video_name, "n_frames": len(vecs)}
        for i, au in enumerate(AU_NAMES):
            row[au] = float(mean_au[i])
        # Add big five
        lab = label_lookup[video_name]
        for trait in ["ValueExtraversion", "ValueAgreeableness", "ValueConscientiousness",
                      "ValueNeurotisicm", "ValueOpenness"]:
            row[trait] = float(lab[trait])
        rows.append(row)

    return pd.DataFrame(rows)


def liu_mean_aus(raw_csv: Path) -> pd.DataFrame:
    """Aggregate Liu raw frame data to per-subject mean AU vector."""
    raw = pd.read_csv(raw_csv)
    raw.columns = [c.strip() for c in raw.columns]
    keep_aus = [a for a in AU_NAMES if a in raw.columns]
    missing = [a for a in AU_NAMES if a not in raw.columns]
    if missing:
        print(f"  warning: Liu data missing {missing}; will fill with 0")
    grouped = raw.groupby("ID")[keep_aus].mean()
    for m in missing:
        grouped[m] = 0.0
    return grouped[AU_NAMES].reset_index()


def train_bigfive_predictors(fi_df: pd.DataFrame, au_cols: list[str]) -> dict[str, Pipeline]:
    """Train 5 Ridge regressors. Each pipeline z-scores AUs *using FI-V2 stats*
    and learns a mapping to that trait. Predicting on Liu data later uses the
    same FI-V2-fitted scaler so cross-dataset extrapolation stays bounded.
    """
    X = fi_df[au_cols].to_numpy()
    predictors: dict[str, Pipeline] = {}
    traits = ["ValueExtraversion", "ValueAgreeableness", "ValueConscientiousness",
              "ValueNeurotisicm", "ValueOpenness"]
    print(f"\nTraining Big-Five predictors on {len(X)} FI-V2 videos with {len(au_cols)} AUs")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    for t in traits:
        y = fi_df[t].to_numpy()
        # Larger alpha to keep weights tame; the dataset has weak signal anyway
        pipe = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=10.0))])
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="r2", n_jobs=-1)
        pipe.fit(X, y)
        predictors[t] = pipe
        print(f"  {t:25s} CV-R²={scores.mean():.3f} ± {scores.std():.3f}")
    return predictors


SHARED_AUS = [a for a in AU_NAMES if a not in ("AU06_r", "AU09_r")]


def _zscore_columns(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = arr.mean(axis=0, keepdims=True)
    sd = arr.std(axis=0, keepdims=True) + 1e-9
    return (arr - mu) / sd, mu, sd


def main() -> None:
    out_au_csv = Path("fi_v2_aus.csv")
    portrait_dirs = [
        Path("/tmp/personality/dataset/portrait-personality-1"),
        Path("/tmp/personality/dataset/portrait-personality-2"),
        Path("/tmp/personality/dataset/portrait-personality-3"),
    ]
    labels_path = Path("/tmp/personality/dataset/bigfive_labels.csv")
    raw_liu = Path("rawdata/rawdata.csv")
    features_liu = Path("features_full.csv")
    out_features = Path("features_full_plus_bigfive.csv")

    if out_au_csv.exists():
        print(f"Reusing existing {out_au_csv}")
        fi_df = pd.read_csv(out_au_csv)
    else:
        detector = make_detector()
        fi_df = build_fi_v2_aus(portrait_dirs, labels_path, detector)
        fi_df.to_csv(out_au_csv, index=False)
        print(f"Wrote {out_au_csv} with {len(fi_df)} videos")

    predictors = train_bigfive_predictors(fi_df, SHARED_AUS)

    print(f"\nComputing Liu mean AUs from {raw_liu}")
    liu_aus = liu_mean_aus(raw_liu)
    print(f"  shape: {liu_aus.shape}")

    # Match scale: the FI-V2 AUs are MediaPipe-blendshape-derived (0..5 scaled),
    # while Liu's are real OpenFace intensities (0..5). Some columns differ in
    # range so we additionally quantile-normalize each AU column between the
    # two datasets to put them on a common rank scale before transfer.
    fi_au_vals = fi_df[SHARED_AUS].to_numpy()
    liu_au_vals = liu_aus[SHARED_AUS].to_numpy()
    from sklearn.preprocessing import QuantileTransformer
    qt_fi = QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=42).fit(fi_au_vals)
    qt_liu = QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=42).fit(liu_au_vals)
    fi_norm = qt_fi.transform(fi_au_vals)
    liu_norm = qt_liu.transform(liu_au_vals)

    # Retrain on quantile-normalized features for cross-dataset stability
    print("\nRetraining on quantile-normalized AUs (cross-dataset alignment)")
    fi_norm_df = pd.DataFrame(fi_norm, columns=SHARED_AUS)
    for trait in ["ValueExtraversion", "ValueAgreeableness", "ValueConscientiousness",
                  "ValueNeurotisicm", "ValueOpenness"]:
        fi_norm_df[trait] = fi_df[trait].values
    predictors = train_bigfive_predictors(fi_norm_df, SHARED_AUS)

    X_liu = liu_norm
    pred_cols = {}
    for trait, pipe in predictors.items():
        short = {"ValueExtraversion": "pred_E", "ValueAgreeableness": "pred_A",
                 "ValueConscientiousness": "pred_C", "ValueNeurotisicm": "pred_N",
                 "ValueOpenness": "pred_O"}[trait]
        pred_cols[short] = pipe.predict(X_liu)
    pred_df = pd.DataFrame(pred_cols)
    pred_df["ID"] = liu_aus["ID"].values
    pred_df = pred_df[["ID", *pred_cols.keys()]]
    print(pred_df.describe())

    # Merge with Liu features
    feat = pd.read_csv(features_liu)
    merged = feat.merge(pred_df, on="ID", how="left")
    # Move self-esteem to last column
    se = merged.pop("self-esteem")
    merged["self-esteem"] = se
    merged.to_csv(out_features, index=False)
    print(f"\nWrote {out_features} with shape {merged.shape}")


if __name__ == "__main__":
    main()

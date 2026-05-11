"""TSFresh feature extraction for self-esteem prediction.

Reads raw OpenFace output (rawdata/rawdata.csv) and produces a tabular feature
matrix for downstream modeling.

Three channel sets are supported:
  --channel-set emotions  4 channels: summed-AU emotions (paper-comparable)
  --channel-set raw      17 channels: raw AU intensities (_r)
  --channel-set full     47 channels: 17 AU_r + 18 AU_c + 2 gaze + 6 pose
                                       + 4 derived emotions

Two parameter sets are supported:
  --parameter-set paper          curated dict matching Liu et al. (~24 features/channel)
  --parameter-set comprehensive  ComprehensiveFCParameters (~783 features/channel)

Two derived dynamics layers can be enabled:
  --include-velocity      run TSFresh on np.diff of each channel
  --include-coactivation  add AU co-activation products (smile coherence etc.)

Usage:
    python feature_extraction.py --channel-set full --out features_full.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.feature_selection.relevance import calculate_relevance_table
from tsfresh.utilities.dataframe_functions import impute

EMOTION_AU_MAP: dict[str, list[str]] = {
    "happiness": ["AU06_r", "AU12_r"],
    "sadness": ["AU01_r", "AU15_r"],
    "disgust": ["AU09_r", "AU10_r"],
    "fear": ["AU01_r", "AU04_r", "AU05_r", "AU20_r"],
}

# AU intensity (_r) channels — 17 total
AU_R_CHANNELS = [
    "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r",
    "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r", "AU20_r", "AU23_r",
    "AU25_r", "AU26_r", "AU45_r",
]

# AU presence (_c, binary) channels — 18 total
AU_C_CHANNELS = [
    "AU01_c", "AU02_c", "AU04_c", "AU05_c", "AU06_c", "AU07_c", "AU09_c",
    "AU10_c", "AU12_c", "AU14_c", "AU15_c", "AU17_c", "AU20_c", "AU23_c",
    "AU25_c", "AU26_c", "AU28_c", "AU45_c",
]

GAZE_CHANNELS = ["gaze_angle_x", "gaze_angle_y"]
POSE_CHANNELS = ["pose_Tx", "pose_Ty", "pose_Tz", "pose_Rx", "pose_Ry", "pose_Rz"]

# AU co-activation pairs — psychologically meaningful pairings
COACTIVATION_PAIRS: list[tuple[str, str, str]] = [
    ("AU06_r", "AU12_r", "smile_coherence"),    # genuine smile (Duchenne)
    ("AU01_r", "AU04_r", "worry_brow"),         # inner brow raise + brow lower
    ("AU09_r", "AU10_r", "disgust_coherence"),  # nose wrinkle + upper lip raise
    ("AU01_r", "AU15_r", "sadness_coherence"),  # inner brow raise + lip corner depressor
    ("AU05_r", "AU20_r", "fear_coherence"),     # upper lid raise + lip stretch
    ("AU04_r", "AU07_r", "frown_intensity"),    # brow lowerer + lid tightener
]

PAPER_PARAMETERS: dict = {
    "abs_energy": None,
    "kurtosis": None,
    "skewness": None,
    "mean": None,
    "standard_deviation": None,
    "variation_coefficient": None,
    "median": None,
    "minimum": None,
    "maximum": None,
    "mean_change": None,
    "mean_abs_change": None,
    "number_peaks": [{"n": 4}],
    "sum_values": None,
    "cid_ce": [{"normalize": True}],
    "sample_entropy": None,
    "approximate_entropy": [{"m": 5, "r": 0.1}],
    "fft_coefficient": [
        {"coeff": 1, "attr": "abs"},
        {"coeff": 1, "attr": "real"},
        {"coeff": 1, "attr": "imag"},
    ],
    "ar_coefficient": [{"coeff": 1, "k": 5}],
    "autocorrelation": [{"lag": 5}],
    "linear_trend": [
        {"attr": "rvalue"},
        {"attr": "intercept"},
        {"attr": "slope"},
    ],
}


def _load_raw(raw_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(raw_path)
    raw.columns = [c.strip() for c in raw.columns]
    return raw


def _emotions_from_aus(raw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"ID": raw["ID"], "frame": raw["frame"]})
    for emotion, aus in EMOTION_AU_MAP.items():
        out[emotion] = raw[aus].sum(axis=1)
    return out


def _coactivation_from_aus(raw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"ID": raw["ID"], "frame": raw["frame"]})
    for a, b, name in COACTIVATION_PAIRS:
        out[name] = raw[a] * raw[b]
    return out


def build_channel_dataframe(raw_path: Path, channel_set: str, include_coactivation: bool) -> pd.DataFrame:
    """Return a long-form DataFrame [ID, frame, <channel cols>] for the requested set."""
    raw = _load_raw(raw_path)
    out = raw[["ID", "frame"]].copy()

    if channel_set == "emotions":
        for emotion, aus in EMOTION_AU_MAP.items():
            out[emotion] = raw[aus].sum(axis=1)
    elif channel_set == "raw":
        for c in AU_R_CHANNELS:
            out[c] = raw[c]
    elif channel_set == "full":
        for c in AU_R_CHANNELS + AU_C_CHANNELS + GAZE_CHANNELS + POSE_CHANNELS:
            out[c] = raw[c]
        for emotion, aus in EMOTION_AU_MAP.items():
            out[emotion] = raw[aus].sum(axis=1)
    else:
        raise ValueError(f"unknown channel_set {channel_set!r}")

    if include_coactivation:
        coact = _coactivation_from_aus(raw)
        for col in coact.columns:
            if col not in ("ID", "frame"):
                out[col] = coact[col]

    return out


def _channels_of(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("ID", "frame")]


def extract_tsfresh_block(
    channel_df: pd.DataFrame,
    parameter_set: str,
    label: str,
    n_jobs: int = 0,
) -> pd.DataFrame:
    """Run TSFresh on every channel in `channel_df`. Returns wide DataFrame indexed by ID."""
    channels = _channels_of(channel_df)
    long_form = channel_df.melt(
        id_vars=["ID", "frame"],
        value_vars=channels,
        var_name="channel",
        value_name="value",
    )
    long_form["id"] = long_form["ID"].astype(str) + "__" + long_form["channel"]

    if parameter_set == "comprehensive":
        settings: dict | ComprehensiveFCParameters = ComprehensiveFCParameters()
    elif parameter_set == "paper":
        settings = PAPER_PARAMETERS
    else:
        raise ValueError(f"unknown parameter_set {parameter_set!r}")

    feats = extract_features(
        long_form[["id", "frame", "value"]],
        column_id="id",
        column_sort="frame",
        default_fc_parameters=settings,
        n_jobs=n_jobs,
        disable_progressbar=True,
    )
    impute(feats)

    feats[["ID", "channel"]] = feats.index.to_series().str.split("__", expand=True).values
    feats["ID"] = feats["ID"].astype(int)
    pivot = feats.pivot(index="ID", columns="channel")
    pivot.columns = [f"{label}__{ch}__{feat}" for feat, ch in pivot.columns]
    return pivot.sort_index()


def velocity_dataframe(channel_df: pd.DataFrame) -> pd.DataFrame:
    """Per-ID first-difference of every channel column. Keeps ID/frame, drops first frame per ID."""
    channels = _channels_of(channel_df)
    out_rows = []
    for sub_id, g in channel_df.groupby("ID", sort=False):
        g = g.sort_values("frame")
        diff = g[channels].diff().iloc[1:]
        diff.insert(0, "frame", g["frame"].iloc[1:].values)
        diff.insert(0, "ID", sub_id)
        out_rows.append(diff)
    return pd.concat(out_rows, ignore_index=True)


def attach_labels(features: pd.DataFrame, labels_path: Path) -> pd.DataFrame:
    labels = pd.read_csv(labels_path)[["ID", "self-esteem"]]
    merged = features.merge(labels, on="ID", how="inner")
    if len(merged) != len(features):
        dropped = set(features["ID"]) - set(merged["ID"])
        raise RuntimeError(f"Lost rows on label merge: {sorted(dropped)[:10]}")
    return merged


def select_relevant_features(
    X: pd.DataFrame, y: pd.Series, fdr_level: float = 0.05
) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = calculate_relevance_table(X, y.astype(bool), ml_task="classification", fdr_level=fdr_level)
    keep = table[table["relevant"]]["feature"].tolist()
    return X[keep].copy(), table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="rawdata/rawdata.csv")
    parser.add_argument("--labels", default="TSfresh_features.csv")
    parser.add_argument("--out", default="features.csv")
    parser.add_argument(
        "--channel-set",
        choices=["emotions", "raw", "full"],
        default="emotions",
    )
    parser.add_argument(
        "--parameter-set",
        choices=["comprehensive", "paper"],
        default="paper",
    )
    parser.add_argument("--include-velocity", action="store_true")
    parser.add_argument("--include-coactivation", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=0)
    args = parser.parse_args()

    raw_path = Path(args.raw)
    labels_path = Path(args.labels)
    out_path = Path(args.out)

    print(f"[1/4] Building channel time-series ({args.channel_set}) from {raw_path}")
    channel_df = build_channel_dataframe(raw_path, args.channel_set, args.include_coactivation)
    channels = _channels_of(channel_df)
    print(f"      shape: {channel_df.shape}, channels: {len(channels)}")

    print(f"[2/4] Extracting TSFresh features (parameter_set={args.parameter_set})")
    feats_static = extract_tsfresh_block(channel_df, args.parameter_set, label="static", n_jobs=args.n_jobs)
    print(f"      static features: {feats_static.shape[1]}")

    if args.include_velocity:
        print("[2b/4] Extracting velocity (first-difference) features")
        vel_df = velocity_dataframe(channel_df)
        vel_df.columns = ["ID", "frame", *[f"vel_{c}" for c in channels]]
        feats_vel = extract_tsfresh_block(vel_df, args.parameter_set, label="velocity", n_jobs=args.n_jobs)
        print(f"       velocity features: {feats_vel.shape[1]}")
        feats = feats_static.join(feats_vel)
    else:
        feats = feats_static

    print(f"[3/4] Total features: {feats.shape[1]} (over {feats.shape[0]} subjects)")
    feats = feats.reset_index()

    print(f"[4/4] Attaching labels from {labels_path}")
    final = attach_labels(feats, labels_path)
    final.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with shape {final.shape}")


if __name__ == "__main__":
    main()

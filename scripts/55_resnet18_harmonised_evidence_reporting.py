#!/usr/bin/env python3
"""
Stage 55: post-hoc harmonised RRA / DC-DCEC-DCEW reporting for the completed
450-image ResNet-18 classification-preserving localisation attack.

NO attack is rerun. The script reads the retained exact NPZ bundles.

ResNet-specific rule:
  valid support Omega = stored content_mask (fixed horizontal padding excluded)
  GT manipulation mask = stored union_mask

Primary evidence thresholds:
  tau_E = 0.5
  tau_RRA = 0.5

RRA uses the exact same helper/tie semantics as the final TruFor analysis:
  scripts/trufor/trufor_evidence_metrics.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

TRUFOR_DIR = ROOT / "scripts" / "trufor"
sys.path.insert(0, str(TRUFOR_DIR))
from trufor_evidence_metrics import relevance_rank_accuracy  # noqa: E402


SRC = ROOT / "output" / "resnet18_localisation_attack_preserve_cls_full_archive_v2"
PER_IMAGE = SRC / "localisation_attack_full_per_image.csv"
CONFIG = SRC / "localisation_attack_full_config.json"
SOURCE_SUMMARY = SRC / "localisation_attack_full_summary.csv"
SOURCE_SELECTION = SRC / "localisation_attack_full_selection.csv"
ARCHIVE = SRC / "exact_archive"

OUT = SRC / "harmonised_evidence_reporting"
TRANSFER = ROOT / "analysis_transfer_bundles"
TAR = TRANSFER / "IMTA135_resnet18_harmonised_evidence_report.tar.gz"
SIDE = Path(str(TAR) + ".sha256")

RRA_HELPER = TRUFOR_DIR / "trufor_evidence_metrics.py"

N_EXPECTED = 450
TAU_E = 0.5
TAU_RRA = 0.5
GRID = [0.25, 0.5, 0.75]
N_BOOT = 10_000
BOOT_SEED = 20260926

EXPECTED_CKPT = (
    "25ad8b1482be20e9d5e450770d6970b8"
    "20ebc9470c9da3c4ec46db2558009402"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().str.strip().isin(["true", "1", "yes"])


def rate(a: int, b: int) -> float:
    return float(a / b) if b else np.nan


def recompute(cam, union, content):
    cam = np.asarray(cam, dtype=np.float64)
    union = np.asarray(union, dtype=bool)
    content = np.asarray(content, dtype=bool)

    if cam.ndim != 2 or union.shape != cam.shape or content.shape != cam.shape:
        raise RuntimeError("CAM/mask geometry mismatch")
    if np.any(union & ~content):
        raise RuntimeError("union_mask escapes content_mask")
    if not np.isfinite(cam[content]).all():
        raise RuntimeError("non-finite CAM values inside valid support")

    n_u = int(union.sum())
    n_c = int(content.sum())
    if n_u <= 0 or n_c <= 0:
        raise RuntimeError("empty GT or content support")

    A = n_u / n_c
    denom = float(cam[content].sum())

    if denom <= 1e-12:
        E = 0.0
        PG = 0.0
    else:
        E = float(cam[union].sum() / denom)
        valid_cam = np.where(content, cam, -np.inf)
        y, x = np.unravel_index(int(np.argmax(valid_cam)), cam.shape)
        PG = float(union[y, x])

    return float(A), float(E), float(E / A), float(PG)


def eval_threshold(df: pd.DataFrame, te: float, tr: float) -> dict:
    dc0 = as_bool(df["DC_clean"])
    dc1 = as_bool(df["DC_adv"])
    dcec = dc0 & (df["E_clean"] >= te) & (df["RRA_clean"] >= tr)
    ef = df["E_adv"] < te
    rf = df["RRA_adv"] < tr
    dcew = dcec & dc1 & (ef | rf)
    flips = dcec & ~dc1
    return {
        "tau_E": te,
        "tau_RRA": tr,
        "n_total": len(df),
        "n_DCEC": int(dcec.sum()),
        "DCEC_rate_all": rate(int(dcec.sum()), len(df)),
        "n_DCEW": int(dcew.sum()),
        "DCEW_rate_given_DCEC": rate(int(dcew.sum()), int(dcec.sum())),
        "n_classification_flips_given_DCEC": int(flips.sum()),
        "classification_flip_rate_given_DCEC": rate(int(flips.sum()), int(dcec.sum())),
        "n_DCEW_E_only": int((dcew & ef & ~rf).sum()),
        "n_DCEW_RRA_only": int((dcew & ~ef & rf).sum()),
        "n_DCEW_both": int((dcew & ef & rf).sum()),
    }


def subgroup_row(g: pd.DataFrame, group: str, value: str) -> dict:
    dcec = as_bool(g["DCEC"])
    dcew = as_bool(g["DCEW"])
    return {
        "group": group,
        "value": value,
        "n": len(g),
        "n_stems": int(g["file_stem"].nunique()),
        "DC_adv_rate": float(as_bool(g["DC_adv"]).mean()),
        "DCEC_n": int(dcec.sum()),
        "DCEC_rate_all": rate(int(dcec.sum()), len(g)),
        "DCEW_n": int(dcew.sum()),
        "DCEW_rate_given_DCEC": rate(int(dcew.sum()), int(dcec.sum())),
        "classification_flip_n": int(as_bool(g["classification_flip"]).sum()),
        "A_median": float(g["A_union"].median()),
        "E_clean_mean": float(g["E_clean"].mean()),
        "E_adv_mean": float(g["E_adv"].mean()),
        "relative_E_degradation_median": float(g["relative_E_degradation"].median()),
        "RRA_clean_mean": float(g["RRA_clean"].mean()),
        "RRA_adv_mean": float(g["RRA_adv"].mean()),
        "relative_RRA_degradation_median": float(g["relative_RRA_degradation"].median()),
        "mu_clean_median": float(g["mu_clean"].median()),
        "mu_adv_median": float(g["mu_adv"].median()),
        "RRA_lift_clean_median": float(g["RRA_lift_clean"].median()),
        "RRA_lift_adv_median": float(g["RRA_lift_adv"].median()),
        "PG_clean_mean": float(g["PG_clean"].mean()),
        "PG_adv_mean": float(g["PG_adv"].mean()),
        "pixel_linf_max": float(g["pixel_linf"].max()),
    }


def tie_summary(df: pd.DataFrame) -> dict:
    dc0 = as_bool(df["DC_clean"])
    dc1 = as_bool(df["DC_adv"])
    p_dcec = as_bool(df["DCEC"])
    p_dcew = as_bool(df["DCEW"])

    dmin = dc0 & (df["E_clean"] >= TAU_E) & (df["RRA_clean_tie_min"] >= TAU_RRA)
    dmax = dc0 & (df["E_clean"] >= TAU_E) & (df["RRA_clean_tie_max"] >= TAU_RRA)

    wmin = dmin & dc1 & ((df["E_adv"] < TAU_E) | (df["RRA_adv_tie_min"] < TAU_RRA))
    wmax = dmax & dc1 & ((df["E_adv"] < TAU_E) | (df["RRA_adv_tie_max"] < TAU_RRA))

    return {
        "clean_DCEC_label_changes_under_tie_min": int((dmin != p_dcec).sum()),
        "clean_DCEC_label_changes_under_tie_max": int((dmax != p_dcec).sum()),
        "adv_DCEW_label_changes_under_tie_min": int((wmin != p_dcew).sum()),
        "adv_DCEW_label_changes_under_tie_max": int((wmax != p_dcew).sum()),
        "DCEC_n_tie_min": int(dmin.sum()),
        "DCEC_n_primary": int(p_dcec.sum()),
        "DCEC_n_tie_max": int(dmax.sum()),
        "DCEW_n_tie_min": int(wmin.sum()),
        "DCEW_n_primary": int(p_dcew.sum()),
        "DCEW_n_tie_max": int(wmax.sum()),
    }


def continuous_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "A_union", "E_clean", "E_adv", "relative_E_degradation",
        "RRA_clean", "RRA_adv", "relative_RRA_degradation",
        "mu_clean", "mu_adv", "RRA_lift_clean", "RRA_lift_adv",
        "PG_clean", "PG_adv", "pixel_linf",
    ]
    rows = []
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce").dropna()
        rows.append({
            "metric": c,
            "n": len(x),
            "mean": float(x.mean()),
            "median": float(x.median()),
            "q25": float(x.quantile(.25)),
            "q75": float(x.quantile(.75)),
            "q95": float(x.quantile(.95)),
            "min": float(x.min()),
            "max": float(x.max()),
        })
    return pd.DataFrame(rows)


def bootstrap(df: pd.DataFrame) -> pd.DataFrame:
    stem = df["file_stem"].astype(str).to_numpy()
    unique = sorted(set(stem))
    groups = [np.flatnonzero(stem == s) for s in unique]
    rng = np.random.default_rng(BOOT_SEED)

    cols = [
        "E_clean", "E_adv", "relative_E_degradation",
        "RRA_clean", "RRA_adv", "relative_RRA_degradation",
        "mu_clean", "mu_adv", "RRA_lift_clean", "RRA_lift_adv",
        "PG_clean", "PG_adv", "pixel_linf",
    ]
    arr = {c: pd.to_numeric(df[c], errors="coerce").to_numpy(float) for c in cols}
    arr["DC_adv"] = as_bool(df["DC_adv"]).to_numpy(bool)
    arr["DCEC"] = as_bool(df["DCEC"]).to_numpy(bool)
    arr["DCEW"] = as_bool(df["DCEW"]).to_numpy(bool)

    names = [
        "classification_preservation_rate", "DCEC_rate", "DCEW_rate_given_DCEC",
        "E_clean_mean", "E_adv_mean", "relative_E_degradation_median",
        "RRA_clean_mean", "RRA_adv_mean", "relative_RRA_degradation_median",
        "mu_clean_median", "mu_adv_median",
        "RRA_lift_clean_median", "RRA_lift_adv_median",
        "PG_clean_mean", "PG_adv_mean", "pixel_linf_mean",
    ]
    vals = {n: np.empty(N_BOOT, float) for n in names}

    for b in range(N_BOOT):
        sampled = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[int(i)] for i in sampled])
        dcec = arr["DCEC"][idx]
        dcew = arr["DCEW"][idx]

        vals["classification_preservation_rate"][b] = arr["DC_adv"][idx].mean()
        vals["DCEC_rate"][b] = dcec.mean()
        vals["DCEW_rate_given_DCEC"][b] = dcew[dcec].mean() if dcec.any() else np.nan
        vals["E_clean_mean"][b] = np.nanmean(arr["E_clean"][idx])
        vals["E_adv_mean"][b] = np.nanmean(arr["E_adv"][idx])
        vals["relative_E_degradation_median"][b] = np.nanmedian(arr["relative_E_degradation"][idx])
        vals["RRA_clean_mean"][b] = np.nanmean(arr["RRA_clean"][idx])
        vals["RRA_adv_mean"][b] = np.nanmean(arr["RRA_adv"][idx])
        vals["relative_RRA_degradation_median"][b] = np.nanmedian(arr["relative_RRA_degradation"][idx])
        vals["mu_clean_median"][b] = np.nanmedian(arr["mu_clean"][idx])
        vals["mu_adv_median"][b] = np.nanmedian(arr["mu_adv"][idx])
        vals["RRA_lift_clean_median"][b] = np.nanmedian(arr["RRA_lift_clean"][idx])
        vals["RRA_lift_adv_median"][b] = np.nanmedian(arr["RRA_lift_adv"][idx])
        vals["PG_clean_mean"][b] = np.nanmean(arr["PG_clean"][idx])
        vals["PG_adv_mean"][b] = np.nanmean(arr["PG_adv"][idx])
        vals["pixel_linf_mean"][b] = np.nanmean(arr["pixel_linf"][idx])

        if (b + 1) % 2000 == 0:
            print(f"bootstrap {b+1}/{N_BOOT}")

    dcec_full = arr["DCEC"]
    point = {
        "classification_preservation_rate": float(arr["DC_adv"].mean()),
        "DCEC_rate": float(dcec_full.mean()),
        "DCEW_rate_given_DCEC": float(arr["DCEW"][dcec_full].mean()),
        "E_clean_mean": float(np.nanmean(arr["E_clean"])),
        "E_adv_mean": float(np.nanmean(arr["E_adv"])),
        "relative_E_degradation_median": float(np.nanmedian(arr["relative_E_degradation"])),
        "RRA_clean_mean": float(np.nanmean(arr["RRA_clean"])),
        "RRA_adv_mean": float(np.nanmean(arr["RRA_adv"])),
        "relative_RRA_degradation_median": float(np.nanmedian(arr["relative_RRA_degradation"])),
        "mu_clean_median": float(np.nanmedian(arr["mu_clean"])),
        "mu_adv_median": float(np.nanmedian(arr["mu_adv"])),
        "RRA_lift_clean_median": float(np.nanmedian(arr["RRA_lift_clean"])),
        "RRA_lift_adv_median": float(np.nanmedian(arr["RRA_lift_adv"])),
        "PG_clean_mean": float(np.nanmean(arr["PG_clean"])),
        "PG_adv_mean": float(np.nanmean(arr["PG_adv"])),
        "pixel_linf_mean": float(np.nanmean(arr["pixel_linf"])),
    }

    rows = []
    for n in names:
        finite = vals[n][np.isfinite(vals[n])]
        lo, hi = np.quantile(finite, [.025, .975])
        rows.append({
            "metric": n,
            "estimate": point[n],
            "ci95_low": float(lo),
            "ci95_high": float(hi),
            "bootstrap_reps": N_BOOT,
            "cluster": "file_stem",
            "seed": BOOT_SEED,
        })
    return pd.DataFrame(rows)


def main():
    for p in [PER_IMAGE, CONFIG, SOURCE_SUMMARY, SOURCE_SELECTION, ARCHIVE, RRA_HELPER]:
        if not p.exists():
            raise RuntimeError(f"Missing required input: {p}")

    cfg = json.loads(CONFIG.read_text())
    if int(cfg["n_images"]) != N_EXPECTED:
        raise RuntimeError("Source config is not the expected 450-image run")
    if cfg["checkpoint_sha256"] != EXPECTED_CKPT:
        raise RuntimeError("Unexpected ResNet-18 checkpoint SHA")
    if float(cfg["epsilon_255"]) != 1.0:
        raise RuntimeError("Expected epsilon=1/255 source run")
    if not bool(cfg["padding_frozen"]):
        raise RuntimeError("Expected frozen padding")

    src = pd.read_csv(PER_IMAGE, keep_default_na=False)
    if len(src) != N_EXPECTED or src["image_path"].duplicated().any():
        raise RuntimeError("Source per-image table population/uniqueness failure")

    archive_count = len(list(ARCHIVE.rglob("*__exact.npz")))
    if archive_count != N_EXPECTED:
        raise RuntimeError(f"Expected 450 exact bundles, found {archive_count}")

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    TRANSFER.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("RESNET-18 STAGE 55 — HARMONISED RRA / DCEC / DCEW")
    print("=" * 72)
    print("population       :", len(src))
    print("exact bundles    :", archive_count)
    print("valid support    : stored content_mask")
    print("GT               : stored union_mask")
    print("tau_E / tau_RRA  :", TAU_E, "/", TAU_RRA)
    print("RRA helper SHA   :", sha256(RRA_HELPER))
    print()

    rows = []
    maxerr = dict(A=0.0, Ec=0.0, Ea=0.0, muc=0.0, mua=0.0, PGc=0.0, PGa=0.0, prob=0.0, margin=0.0)

    for i, (_, s) in enumerate(src.iterrows(), 1):
        p = Path(str(s["exact_bundle_path"]))
        if not p.is_absolute():
            p = ROOT / p
        if not p.is_file():
            raise RuntimeError(f"Missing exact bundle: {p}")
        if int(s["exact_bundle_bytes"]) != p.stat().st_size:
            raise RuntimeError(f"Exact bundle byte-size mismatch: {p}")

        with np.load(p, allow_pickle=False) as z:
            if str(np.asarray(z["schema_version"]).item()) != "resnet_exact_archive_v1":
                raise RuntimeError("Unexpected exact archive schema")
            if str(np.asarray(z["image_path"]).item()) != str(s["image_path"]):
                raise RuntimeError("image_path mismatch in exact archive")

            cc = np.asarray(z["clean_cam_full"], np.float32)
            ac = np.asarray(z["adv_cam_full"], np.float32)
            content = np.asarray(z["content_mask"], bool)
            union = np.asarray(z["union_mask"], bool)
            cp = float(np.asarray(z["clean_probability_attack"]).item())
            ap = float(np.asarray(z["adv_probability_attack"]).item())
            cm = float(np.asarray(z["clean_margin"]).item())
            am = float(np.asarray(z["adv_margin"]).item())

        A, Ec, muc, PGc = recompute(cc, union, content)
        A2, Ea, mua, PGa = recompute(ac, union, content)
        if abs(A - A2) > 1e-15:
            raise RuntimeError("clean/adv A mismatch")

        rc = relevance_rank_accuracy(cc, union, valid_mask=content)
        ra = relevance_rank_accuracy(ac, union, valid_mask=content)

        maxerr["A"] = max(maxerr["A"], abs(A - float(s["A"])))
        maxerr["Ec"] = max(maxerr["Ec"], abs(Ec - float(s["E_clean"])))
        maxerr["Ea"] = max(maxerr["Ea"], abs(Ea - float(s["E_adv"])))
        maxerr["muc"] = max(maxerr["muc"], abs(muc - float(s["mu_w_clean"])))
        maxerr["mua"] = max(maxerr["mua"], abs(mua - float(s["mu_w_adv"])))
        maxerr["PGc"] = max(maxerr["PGc"], abs(PGc - float(s["PG_clean"])))
        maxerr["PGa"] = max(maxerr["PGa"], abs(PGa - float(s["PG_adv"])))
        maxerr["prob"] = max(maxerr["prob"], abs(cp-float(s["clean_probability_attack"])), abs(ap-float(s["adv_probability_attack"])))
        maxerr["margin"] = max(maxerr["margin"], abs(cm-float(s["clean_margin"])), abs(am-float(s["adv_margin"])))

        dc0 = cm >= 0.0
        dc1 = am >= 0.0
        src_preserved = str(s["classification_preserved"]).strip().lower() in ["true", "1", "yes"]
        if dc1 != src_preserved:
            raise RuntimeError("DC_adv disagrees with source classification_preserved")

        dcec = bool(dc0 and Ec >= TAU_E and rc.rra >= TAU_RRA)
        ef = Ea < TAU_E
        rf = ra.rra < TAU_RRA
        dcew = bool(dcec and dc1 and (ef or rf))

        r = s.to_dict()
        r.update({
            "eval_split": str(s["evaluation_split"]),
            "classification_margin_floor": 0.0,
            "tau_E": TAU_E,
            "tau_RRA": TAU_RRA,
            "valid_support": "content_mask",
            "DC_clean": dc0,
            "DC_adv": dc1,
            "A_union": A,
            "E_clean": Ec,
            "E_adv": Ea,
            "relative_E_degradation": ((Ec-Ea)/Ec if Ec > 0 else np.nan),
            "mu_clean": muc,
            "mu_adv": mua,
            "PG_clean": PGc,
            "PG_adv": PGa,
            "RRA_clean": rc.rra,
            "RRA_adv": ra.rra,
            "delta_RRA_adv_minus_clean": ra.rra - rc.rra,
            "relative_RRA_degradation": ((rc.rra-ra.rra)/rc.rra if rc.rra > 0 else np.nan),
            "RRA_clean_tie_expected": rc.rra_tie_expected,
            "RRA_clean_tie_min": rc.rra_tie_min,
            "RRA_clean_tie_max": rc.rra_tie_max,
            "RRA_clean_tie_width": rc.rra_tie_max - rc.rra_tie_min,
            "RRA_clean_cutoff": rc.cutoff,
            "RRA_clean_cutoff_tie": rc.cutoff_tie_crosses_boundary,
            "RRA_adv_tie_expected": ra.rra_tie_expected,
            "RRA_adv_tie_min": ra.rra_tie_min,
            "RRA_adv_tie_max": ra.rra_tie_max,
            "RRA_adv_tie_width": ra.rra_tie_max - ra.rra_tie_min,
            "RRA_adv_cutoff": ra.cutoff,
            "RRA_adv_cutoff_tie": ra.cutoff_tie_crosses_boundary,
            "RRA_K": rc.k,
            "RRA_N_valid": rc.n_valid,
            "RRA_lift_clean": rc.rra / A,
            "RRA_lift_adv": ra.rra / A,
            "RRA_excess_over_area_clean": rc.rra - A,
            "RRA_excess_over_area_adv": ra.rra - A,
            "DCEC": dcec,
            "E_adv_below_tau": ef,
            "RRA_adv_below_tau": rf,
            "DCEW": dcew,
            "classification_flip": bool(dcec and not dc1),
            "DCEW_E_only": bool(dcew and ef and not rf),
            "DCEW_RRA_only": bool(dcew and not ef and rf),
            "DCEW_both": bool(dcew and ef and rf),
        })
        rows.append(r)

        if i % 50 == 0 or i == len(src):
            print(f"metrics {i}/{len(src)}")

    df = pd.DataFrame(rows)

    # Audit tolerances: exact archive and original CSV should agree to numerical precision.
    if not as_bool(df["DC_clean"]).all():
        raise RuntimeError("Source population contains clean decision failures")
    if not as_bool(df["DC_adv"]).all():
        raise RuntimeError("Expected all 450 classifications to be preserved")
    if maxerr["A"] > 1e-12:
        raise RuntimeError(f"A recomputation error: {maxerr['A']}")
    if max(maxerr["Ec"], maxerr["Ea"]) > 5e-6:
        raise RuntimeError(f"E recomputation error: {maxerr}")
    if max(maxerr["muc"], maxerr["mua"]) > 5e-5:
        raise RuntimeError(f"mu recomputation error: {maxerr}")
    if max(maxerr["PGc"], maxerr["PGa"]) != 0.0:
        raise RuntimeError(f"PG recomputation error: {maxerr}")
    if maxerr["prob"] > 1e-12 or maxerr["margin"] > 1e-12:
        raise RuntimeError(f"exact archive scalar disagreement: {maxerr}")

    # Primary labels.
    p = eval_threshold(df, TAU_E, TAU_RRA)
    primary_mask = (
        as_bool(df["DC_clean"])
        & (df["E_clean"] >= TAU_E)
        & (df["RRA_clean"] >= TAU_RRA)
    )
    df["DCEC"] = primary_mask
    df["DCEW"] = (
        primary_mask
        & as_bool(df["DC_adv"])
        & ((df["E_adv"] < TAU_E) | (df["RRA_adv"] < TAU_RRA))
    )

    p.update({
        "population": N_EXPECTED,
        "population_definition": "clean-correct ResNet-18 classification-preserving localisation-attack population",
        "n_DC_clean": int(as_bool(df["DC_clean"]).sum()),
        "n_DC_adv": int(as_bool(df["DC_adv"]).sum()),
        "classification_preservation_rate_all": float(as_bool(df["DC_adv"]).mean()),
        "classification_margin_floor": 0.0,
        "valid_support": "stored content_mask; fixed horizontal padding excluded",
        "gt_mask": "stored union_mask",
        "checkpoint_sha256": EXPECTED_CKPT,
        "source_epsilon_255": float(cfg["epsilon_255"]),
        "RRA_helper_sha256": sha256(RRA_HELPER),
        "E_clean_mean": float(df["E_clean"].mean()),
        "E_clean_median": float(df["E_clean"].median()),
        "E_adv_mean": float(df["E_adv"].mean()),
        "E_adv_median": float(df["E_adv"].median()),
        "relative_E_degradation_mean": float(df["relative_E_degradation"].mean()),
        "relative_E_degradation_median": float(df["relative_E_degradation"].median()),
        "RRA_clean_mean": float(df["RRA_clean"].mean()),
        "RRA_clean_median": float(df["RRA_clean"].median()),
        "RRA_adv_mean": float(df["RRA_adv"].mean()),
        "RRA_adv_median": float(df["RRA_adv"].median()),
        "relative_RRA_degradation_mean": float(df["relative_RRA_degradation"].mean()),
        "relative_RRA_degradation_median": float(df["relative_RRA_degradation"].median()),
        "mu_clean_median": float(df["mu_clean"].median()),
        "mu_adv_median": float(df["mu_adv"].median()),
        "RRA_lift_clean_median": float(df["RRA_lift_clean"].median()),
        "RRA_lift_adv_median": float(df["RRA_lift_adv"].median()),
        "PG_clean_mean": float(df["PG_clean"].mean()),
        "PG_adv_mean": float(df["PG_adv"].mean()),
        "clean_cutoff_ties": int(as_bool(df["RRA_clean_cutoff_tie"]).sum()),
        "adv_cutoff_ties": int(as_bool(df["RRA_adv_cutoff_tie"]).sum()),
        "max_clean_tie_width": float(df["RRA_clean_tie_width"].max()),
        "max_adv_tie_width": float(df["RRA_adv_tie_width"].max()),
        "pixel_linf_mean": float(df["pixel_linf"].mean()),
        "pixel_linf_max": float(df["pixel_linf"].max()),
    })
    p.update(tie_summary(df))

    sensitivity = pd.DataFrame(
        [eval_threshold(df, te, tr) for te in GRID for tr in GRID]
    )

    subrows = []
    for gname, col in [("variant","variant"), ("hardware_source","hardware_source"), ("eval_split","eval_split")]:
        for value, g in df.groupby(col, dropna=False):
            subrows.append(subgroup_row(g, gname, str(value)))
    subgroups = pd.DataFrame(subrows)

    cont = continuous_summary(df)

    print()
    print("Running 10,000-replicate file-stem clustered bootstrap...")
    boot = bootstrap(df)

    audit = {
        "status": "PASS",
        "population": N_EXPECTED,
        "exact_bundle_count": archive_count,
        "max_A_recompute_error": maxerr["A"],
        "max_E_clean_recompute_error": maxerr["Ec"],
        "max_E_adv_recompute_error": maxerr["Ea"],
        "max_mu_clean_recompute_error": maxerr["muc"],
        "max_mu_adv_recompute_error": maxerr["mua"],
        "max_PG_clean_recompute_error": maxerr["PGc"],
        "max_PG_adv_recompute_error": maxerr["PGa"],
        "max_probability_exact_bundle_error": maxerr["prob"],
        "max_margin_exact_bundle_error": maxerr["margin"],
        "pixel_linf_max": float(df["pixel_linf"].max()),
    }

    protocol = {
        "schema_version": 1,
        "name": "ResNet-18 harmonised DC-DCEC-DCEW evidence reporting",
        "attack_rerun": False,
        "relationship_to_trufor": "same E/RRA/DCEC/DCEW reporting semantics and identical RRA implementation/tie handling",
        "valid_support": "stored content_mask; fixed padding excluded",
        "ground_truth": "stored union_mask",
        "DC_clean": "clean_margin >= 0",
        "DC_adv": "adv_margin >= 0",
        "E": "sum CAM in union_mask / sum CAM in content_mask",
        "RRA": "K=|union_mask|; exact top-K CAM pixels within content_mask; GT overlap / K",
        "RRA_primary_tie_rule": "ascending flattened valid-pixel index within cutoff tie",
        "tau_E": TAU_E,
        "tau_RRA": TAU_RRA,
        "threshold_status": "study-defined majority criterion; not a literature-mandated cutoff",
        "DCEC": "DC_clean AND E_clean >= tau_E AND RRA_clean >= tau_RRA",
        "DCEW": "DCEC AND DC_adv AND (E_adv < tau_E OR RRA_adv < tau_RRA)",
        "classification_flip": "reported separately; never counted as DCEW",
        "sensitivity_grid": GRID,
        "bootstrap": {"cluster":"file_stem","reps":N_BOOT,"seed":BOOT_SEED,"CI":"percentile 95%"},
        "source_attack_objective": cfg["objective"],
        "source_epsilon_255": cfg["epsilon_255"],
        "checkpoint_sha256": EXPECTED_CKPT,
        "RRA_helper_sha256": sha256(RRA_HELPER),
    }

    # Write outputs.
    df.to_csv(OUT / "55_resnet18_harmonised_evidence_per_image.csv", index=False)
    (OUT / "55_resnet18_primary_summary.json").write_text(json.dumps(p, indent=2, sort_keys=True) + "\n")
    pd.DataFrame([p]).to_csv(OUT / "55_resnet18_primary_summary.csv", index=False)
    sensitivity.to_csv(OUT / "55_resnet18_threshold_sensitivity.csv", index=False)
    subgroups.to_csv(OUT / "55_resnet18_subgroup_summary.csv", index=False)
    cont.to_csv(OUT / "55_resnet18_continuous_metrics.csv", index=False)
    boot.to_csv(OUT / "55_resnet18_cluster_bootstrap_CI.csv", index=False)

    df[[
        "image_path","file_stem","variant","hardware_source","eval_split",
        "A_union","RRA_K","RRA_N_valid",
        "RRA_clean","RRA_clean_cutoff_tie","RRA_clean_tie_expected","RRA_clean_tie_min","RRA_clean_tie_max","RRA_clean_tie_width",
        "RRA_adv","RRA_adv_cutoff_tie","RRA_adv_tie_expected","RRA_adv_tie_min","RRA_adv_tie_max","RRA_adv_tie_width",
    ]].to_csv(OUT / "55_resnet18_RRA_tie_diagnostics.csv", index=False)

    (OUT / "55_resnet18_scientific_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    (OUT / "55_resnet18_evidence_protocol_snapshot.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")

    # Compact source-run metadata.
    source_dir = OUT / "source_run"
    source_dir.mkdir()
    for q in [CONFIG, SOURCE_SUMMARY, SOURCE_SELECTION]:
        shutil.copy2(q, source_dir / q.name)

    btab = boot.set_index("metric")
    def bv(metric, col): return float(btab.loc[metric, col])

    dcec_n = int(p["n_DCEC"])
    dcew_n = int(p["n_DCEW"])
    writer = f"""# FINAL ResNet-18 harmonised evidence-reporting handover

## Status
This is a **post-hoc reporting analysis** over the completed 450-image
classification-preserving ResNet-18 localisation attack. The attack was not rerun.

Exact scientific archive: **450/450 bundles**. Stage-55 scientific audit: **PASS**.

## Common reporting framework
The same final evidence vocabulary used for TruFor is applied here:
DC, E/RMA, RRA, DCEC and DCEW.

ResNet-specific valid support is the exact stored `content_mask`, because the
model uses a fixed 512x864 canvas. Fixed horizontal padding is excluded.
GT is the exact stored `union_mask`.

Primary thresholds:
- tau_E = 0.5
- tau_RRA = 0.5

These are study-defined majority criteria, not universal literature cutoffs.

## Primary result
- classification preserved: **{int(p['n_DC_adv'])}/450**
- clean DCEC: **{dcec_n}/450 = {100*float(p['DCEC_rate_all']):.2f}%**
- DCEC 95% file-stem clustered-bootstrap CI:
  **{100*bv('DCEC_rate','ci95_low'):.2f}% to {100*bv('DCEC_rate','ci95_high'):.2f}%**
- DCEW: **{dcew_n}/{dcec_n} = {100*float(p['DCEW_rate_given_DCEC']):.2f}% conditional on DCEC**
- DCEW 95% clustered-bootstrap CI:
  **{100*bv('DCEW_rate_given_DCEC','ci95_low'):.2f}% to {100*bv('DCEW_rate_given_DCEC','ci95_high'):.2f}%**
- classification flips within DCEC: **{int(p['n_classification_flips_given_DCEC'])}**
- DCEW decomposition: E-only **{int(p['n_DCEW_E_only'])}**,
  RRA-only **{int(p['n_DCEW_RRA_only'])}**, both **{int(p['n_DCEW_both'])}**

## Continuous degradation
- mean E: **{bv('E_clean_mean','estimate'):.6f} -> {bv('E_adv_mean','estimate'):.6f}**
- median relative E degradation: **{100*bv('relative_E_degradation_median','estimate'):.3f}%**
- mean RRA: **{bv('RRA_clean_mean','estimate'):.6f} -> {bv('RRA_adv_mean','estimate'):.6f}**
- median relative RRA degradation: **{100*bv('relative_RRA_degradation_median','estimate'):.3f}%**
- median mu_w: **{bv('mu_clean_median','estimate'):.6f} -> {bv('mu_adv_median','estimate'):.6f}**
- median RRA lift: **{bv('RRA_lift_clean_median','estimate'):.6f} -> {bv('RRA_lift_adv_median','estimate'):.6f}**
- mean Pointing Game: **{bv('PG_clean_mean','estimate'):.6f} -> {bv('PG_adv_mean','estimate'):.6f}**
- max physical L_inf in source run: **{float(p['pixel_linf_max']):.12f}**, approximately 1/255

## Threshold sensitivity
Same 3x3 grid as TruFor:
tau_E and tau_RRA in {{0.25, 0.50, 0.75}}.

DCEC denominator range: **{int(sensitivity['n_DCEC'].min())} to {int(sensitivity['n_DCEC'].max())}**.
Use `55_resnet18_threshold_sensitivity.csv` for exact DCEW rates/decomposition.

## Tie robustness
- clean cutoff ties: **{int(p['clean_cutoff_ties'])}**
- adversarial cutoff ties: **{int(p['adv_cutoff_ties'])}**
- DCEC label changes under tie-min / tie-max:
  **{int(p['clean_DCEC_label_changes_under_tie_min'])} / {int(p['clean_DCEC_label_changes_under_tie_max'])}**
- DCEW label changes under tie-min / tie-max:
  **{int(p['adv_DCEW_label_changes_under_tie_min'])} / {int(p['adv_DCEW_label_changes_under_tie_max'])}**

## Critical interpretation
DCEC/DCEW are a harmonised **reporting layer**, not the attack objective.
The executed source attack objective was `{cfg['objective']}`.
Do not rewrite the experiment as though RRA or DCEC/DCEW were optimised.

Retain the earlier continuous ResNet metrics and use RRA/DCEC/DCEW alongside
them for cohesion with the TruFor chapter.

## Source hierarchy
1. `55_resnet18_primary_summary.json`
2. `55_resnet18_cluster_bootstrap_CI.csv`
3. `55_resnet18_threshold_sensitivity.csv`
4. `55_resnet18_subgroup_summary.csv`
5. `55_resnet18_continuous_metrics.csv`
6. `55_resnet18_harmonised_evidence_per_image.csv`
7. `55_resnet18_scientific_audit.json`
8. `55_resnet18_evidence_protocol_snapshot.json`

Checkpoint SHA256:
`{EXPECTED_CKPT}`

RRA implementation:
`scripts/trufor/trufor_evidence_metrics.py`

RRA helper SHA256:
`{sha256(RRA_HELPER)}`

Git commit:
`{git('rev-parse','HEAD')}`
"""
    (OUT / "RESNET18_FINAL_WRITER_HANDOVER.md").write_text(writer)

    # Hash compact output state.
    sums = OUT / "SHA256SUMS.txt"
    files = sorted(q for q in OUT.rglob("*") if q.is_file() and q != sums)
    sums.write_text("".join(f"{sha256(q)}  {q.relative_to(OUT)}\n" for q in files))

    # Writer transfer bundle (does not duplicate the 1.8 GiB exact archive).
    if TAR.exists():
        TAR.unlink()
    with tarfile.open(TAR, "w:gz") as tf:
        tf.add(OUT, arcname=OUT.name)

    tar_sha = sha256(TAR)
    SIDE.write_text(f"{tar_sha}  {TAR.name}\n")

    print()
    print("=" * 72)
    print("RESNET-18 STAGE 55 — FINAL HARMONISED RESULT")
    print("=" * 72)
    print("population                  :", len(df))
    print("classification preserved    :", f"{int(as_bool(df['DC_adv']).sum())}/{len(df)}")
    print("clean DCEC                  :", p["n_DCEC"])
    print("DCEW                        :", p["n_DCEW"])
    print("DCEW rate | DCEC            :", p["DCEW_rate_given_DCEC"])
    print("classification flips | DCEC :", p["n_classification_flips_given_DCEC"])
    print("E-only / RRA-only / both    :", (p["n_DCEW_E_only"], p["n_DCEW_RRA_only"], p["n_DCEW_both"]))
    print("clean / adv cutoff ties     :", (p["clean_cutoff_ties"], p["adv_cutoff_ties"]))
    print("DCEC tie changes min/max    :", (p["clean_DCEC_label_changes_under_tie_min"], p["clean_DCEC_label_changes_under_tie_max"]))
    print("DCEW tie changes min/max    :", (p["adv_DCEW_label_changes_under_tie_min"], p["adv_DCEW_label_changes_under_tie_max"]))
    print()
    print("mean E clean -> adv         :", f"{p['E_clean_mean']:.6f} -> {p['E_adv_mean']:.6f}")
    print("mean RRA clean -> adv       :", f"{p['RRA_clean_mean']:.6f} -> {p['RRA_adv_mean']:.6f}")
    print("median rel E degradation    :", f"{p['relative_E_degradation_median']:.6f}")
    print("median rel RRA degradation  :", f"{p['relative_RRA_degradation_median']:.6f}")
    print("max L_inf                   :", f"{p['pixel_linf_max']:.12f}")
    print()
    print("outputs                     :", OUT)
    print("writer package              :", TAR)
    print("writer package SHA256       :", tar_sha)
    print()
    print("STAGE 55 PASS")


if __name__ == "__main__":
    main()

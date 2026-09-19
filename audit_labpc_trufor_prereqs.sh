#!/usr/bin/env bash
set -u

EXPECTED_TRUFOR_COMMIT="ae54475df6f41a491d7615100feb19263dec13f7"
EXPECTED_ZIP_MD5="7bee48f3476c75616c3c5721ab256ff8"
EXPECTED_CKPT_SHA256="ac1d90e329a72e0d66e8665e123a19e94bfae3209c3ef8a4f9ca3b91578c7844"
EXPECTED_INVENTORY_SHA256="54fa68d9e3695ffbe200917ad59b47f9c2a855d47d974896f53a5fd171abfe6a"

if ! ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"; then
  echo "ERROR: run this from inside ~/ForgedDocDeepID"
  exit 2
fi

mkdir -p "$ROOT/logs/LABPC"
STAMP="$(date +%Y%m%d_%H%M%S)"
REPORT="$ROOT/logs/LABPC/labpc_prereq_audit_${STAMP}.txt"
exec > >(tee "$REPORT") 2>&1
cd "$ROOT"

echo "===== REPO / MACHINE ====="
echo "repo root: $(pwd)"
echo "hostname:  $(hostname)"
echo "branch:    $(git branch --show-current)"
echo "HEAD:      $(git rev-parse HEAD)"
git status --short || true
echo

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU:"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader || true
fi
echo "Python: $(python --version 2>&1)"
python - <<'PY' || true
try:
    import torch
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"torch GPU {i}: {p.name}; {p.total_memory/2**30:.2f} GiB")
except Exception as e:
    print("PyTorch audit unavailable:", e)
PY

echo
echo "===== TRUFOR ====="
for p in   external/TruFor   external/TruFor/TruFor_train_test   external/TruFor/test_docker/src/TruFor_weights.zip   external/TruFor/test_docker/src/weights/trufor.pth.tar
do
  [[ -e "$p" ]] && echo "[OK] $p" || echo "[MISSING] $p"
done

if [[ -d external/TruFor/.git ]]; then
  actual="$(git -C external/TruFor rev-parse HEAD 2>/dev/null || true)"
  echo "TruFor HEAD: $actual"
  [[ "$actual" == "$EXPECTED_TRUFOR_COMMIT" ]] && echo "[OK] TruFor frozen commit" || echo "[WARN] TruFor commit differs"
  echo "Tracked TruFor changes:"
  git -C external/TruFor status --porcelain --untracked-files=no || true
fi

if [[ -f external/TruFor/test_docker/src/TruFor_weights.zip ]]; then
  actual="$(md5sum external/TruFor/test_docker/src/TruFor_weights.zip | awk '{print $1}')"
  echo "weights ZIP MD5: $actual"
  [[ "$actual" == "$EXPECTED_ZIP_MD5" ]] && echo "[OK] ZIP MD5" || echo "[WARN] ZIP MD5 mismatch"
fi

if [[ -f external/TruFor/test_docker/src/weights/trufor.pth.tar ]]; then
  actual="$(sha256sum external/TruFor/test_docker/src/weights/trufor.pth.tar | awk '{print $1}')"
  echo "checkpoint SHA256: $actual"
  [[ "$actual" == "$EXPECTED_CKPT_SHA256" ]] && echo "[OK] checkpoint SHA256" || echo "[WARN] checkpoint SHA256 mismatch"
fi

echo
echo "===== FANTASYID / POLICY-C ====="
for p in   data/FantasyID   data/processed/policy_c   data/processed/policy_c_official_test   output/policy_c_cache_index.csv   output/fantasyid_official_test_policy_c_index.csv   output/fantasyid_inventory_2026-09-05_023126.xlsx
do
  [[ -e "$p" ]] && echo "[OK] $p" || echo "[MISSING] $p"
done

if [[ -f output/fantasyid_inventory_2026-09-05_023126.xlsx ]]; then
  actual="$(sha256sum output/fantasyid_inventory_2026-09-05_023126.xlsx | awk '{print $1}')"
  echo "inventory SHA256: $actual"
  [[ "$actual" == "$EXPECTED_INVENTORY_SHA256" ]] && echo "[OK] inventory SHA256" || echo "[WARN] inventory SHA256 mismatch"
fi

if [[ -d data/FantasyID ]]; then
  echo "raw image files: $(find data/FantasyID -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)"
fi

python - <<'PY'
from pathlib import Path
import csv
root = Path.cwd()
for label, rel, expected in [
    ("project Policy-C", "output/policy_c_cache_index.csv", 1899),
    ("official Policy-C", "output/fantasyid_official_test_policy_c_index.csv", 1385),
]:
    p = root/rel
    print(f"\n{label}:")
    if not p.is_file():
        print("  index missing")
        continue
    with p.open(newline="") as f:
        rows = list(csv.DictReader(f))
    missing = [r.get("cache_path","") for r in rows if not r.get("cache_path") or not (root/r["cache_path"]).is_file()]
    print(f"  index rows: {len(rows)} (expected {expected})")
    print(f"  referenced cache files present: {len(rows)-len(missing)}/{len(rows)}")
    print(f"  referenced cache files missing: {len(missing)}")
    for x in missing[:10]:
        print("   ", x)
    if len(missing) > 10:
        print(f"    ... plus {len(missing)-10} more")
PY

echo
echo "===== TRUFOR SCRIPTS ====="
for p in   scripts/trufor/trufor_common.py   scripts/trufor/01_verify_trufor_policy_c_native.py   scripts/trufor/02_infer_policy_c_native.py   scripts/trufor/03_calibrate_dev_threshold.py   scripts/trufor/04_eval_frozen_threshold.py   scripts/trufor/03_calibrate_dev_threshold_accuracy.py   scripts/trufor/04_eval_frozen_threshold_accuracy.py
do
  if [[ -f "$p" ]]; then
    if git ls-files --error-unmatch "$p" >/dev/null 2>&1; then
      echo "[OK tracked]   $p"
    else
      echo "[WARN untracked] $p"
    fi
  else
    echo "[ABSENT]       $p"
  fi
done

echo
echo "===== EXISTING STAGE-2 OUTPUTS ====="
for p in   output/trufor_pretrained_policy_c_native/stage01_provenance.json   output/trufor_pretrained_policy_c_native/stage02_inference_provenance.json   output/trufor_pretrained_policy_c_native/inference_manifest.csv
do
  [[ -f "$p" ]] && echo "[OK] $p" || echo "[MISSING] $p"
done

python - <<'PY'
from pathlib import Path
import csv
root = Path.cwd()
p = root/"output/trufor_pretrained_policy_c_native/inference_manifest.csv"
if p.is_file():
    with p.open(newline="") as f:
        rows = list(csv.DictReader(f))
    missing = [r.get("map_path","") for r in rows if not r.get("map_path") or not (root/r["map_path"]).is_file()]
    print(f"Stage-2 manifest rows: {len(rows)} (expected 1844)")
    print(f"Stage-2 referenced .npz maps present: {len(rows)-len(missing)}/{len(rows)}")
    print(f"Stage-2 referenced .npz maps missing: {len(missing)}")
    for x in missing[:10]:
        print("  ", x)
    if len(missing) > 10:
        print(f"   ... plus {len(missing)-10} more")
else:
    print("Stage-2 manifest absent.")
PY

if [[ -d output/trufor_pretrained_policy_c_native/maps ]]; then
  echo "physical .npz count: $(find output/trufor_pretrained_policy_c_native/maps -type f -name '*.npz' | wc -l) (expected 1844 for completed old run)"
else
  echo "Stage-2 maps directory absent."
fi

echo
echo "===== LABPC NAMESPACE ====="
for d in logs/LABPC output/LABPC output/LABPC/trufor_pretrained_policy_c_native output/LABPC/trufor_policy_c_frozen_protocol; do
  [[ -d "$d" ]] && echo "[EXISTS] $d" || echo "[NOT YET] $d"
done

echo
echo "===== DISK ====="
df -h "$ROOT" | sed -n '1,2p'

echo
echo "===== INTERPRETATION ====="
echo "- Old Stage-2 .npz maps are optional if we choose a fresh LABPC rerun."
echo "- For a fresh LABPC rerun, TruFor source/weights + FantasyID data + frozen manifests are essential."
echo "- Complete Policy-C caches let us rerun TruFor immediately; if incomplete, regenerate Policy-C first."
echo "- This script is read-only except for writing this report."
echo
echo "Report: $REPORT"

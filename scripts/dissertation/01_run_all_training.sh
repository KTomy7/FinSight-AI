#!/usr/bin/env bash
# ==============================================================================
# Script 1 — Single-Cutoff Training & Leaderboard
# ==============================================================================
# Trains all five models with a fixed train/test split (cutoff 2024-01-01)
# and produces a CLI leaderboard. Walk-forward training is handled by Script 02.
#
# Run from repo root:
#   bash scripts/dissertation/01_run_all_training.sh
# ==============================================================================
set -euo pipefail

CUTOFF="2024-01-01"
END="2024-12-31"
YEARS=3
ARTIFACTS_DIR="artifacts/runs"
DISS_DIR="artifacts/dissertation"
MODELS="naive_zero naive_mean ridge hist_gbdt xgboost"

mkdir -p "$DISS_DIR"

# ── Step 1: Train all models ─────────────────────────────────────────────────
echo "============================================================"
echo " Step 1: Single-Cutoff Training"
echo "   cutoff=$CUTOFF  end=$END  years=$YEARS"
echo "   models: $MODELS"
echo "============================================================"
echo ""

finsight train \
    --cutoff "$CUTOFF" \
    --years "$YEARS" \
    --end "$END" \
    --model-types $MODELS \
    --artifacts-dir "$ARTIFACTS_DIR"

echo ""

# ── Step 2: Leaderboard ─────────────────────────────────────────────────────
echo "============================================================"
echo " Step 2: Model Comparison Leaderboard"
echo "============================================================"
echo ""

finsight compare \
    --model-ids $MODELS \
    --rank-by mae rmse direction_accuracy \
    --artifacts-dir "$ARTIFACTS_DIR" \
    | tee "$DISS_DIR/leaderboard.txt"

echo ""
echo "Leaderboard saved to $DISS_DIR/leaderboard.txt"
echo "Single-cutoff training complete. Run Script 02 next."

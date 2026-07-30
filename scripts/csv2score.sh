#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================


CSV_PATH="/data/rlawlsgurjh/work/gitlab/CUE/cue_vs/results/6S89/CUE/Trial_1/Proxy/PI/06_sa_score_filtered.csv"
OUTPUT="/data/rlawlsgurjh/work/gitlab/CUE/cue_vs/results/6S89/CUE/Trial_1/Proxy/PI/06_ra_score.csv"

# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"

python csv2score.py \
    --csv_path ${CSV_PATH} \
    --output ${OUTPUT}

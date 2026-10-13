#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================


CSV_PATH="/HDD1/phj318/Work/Autodock/EGFR_V2/ligand/triple_only_hits.csv"
OUTPUT="/HDD1/phj318/Work/Autodock/EGFR_V2/ligand/triple_only_hits_output.csv"

# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"

python csv2score.py \
    --csv_path ${CSV_PATH} \
    --output ${OUTPUT}

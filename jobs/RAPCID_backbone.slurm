#!/bin/bash
#SBATCH -J RAPCID_backbone
#SBATCH -p amd_a200nv_8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH --comment="field=sciai;appl=pytorch"
#SBATCH -o /scratch/$USER/RAPCID_2610_output/logs/RAPCID_backbone_%j.out
#SBATCH -e /scratch/$USER/RAPCID_2610_output/logs/RAPCID_backbone_%j.err

set -euo pipefail

ROOT="/home01/$USER/RAPCID_2610"
INPUT="$ROOT/inputs/RAPCID_backbone.json"
OUT="/scratch/$USER/RAPCID_2610_output/backbone"
RFD3="/scratch/$USER/rfd3/env/bin/rfd3"

export FOUNDRY_CHECKPOINT_DIRS="/scratch/$USER/rfd3/checkpoints"

mkdir -p "$OUT"

echo "=== RAPCID backbone generation ==="
echo "Host: $(hostname)"
echo "Input: $INPUT"
echo "Output: $OUT"
echo "Target backbones: 2000"
echo "Started: $(date)"

cd "$ROOT"

"$RFD3" design \
    inputs="$INPUT" \
    out_dir="$OUT" \
    n_batches=250 \
    diffusion_batch_size=8

echo "Finished: $(date)"

N_CIF=$(find "$OUT" -type f -name '*.cif.gz' | wc -l)
echo "Generated CIF files: $N_CIF"
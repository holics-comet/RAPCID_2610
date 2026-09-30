#!/bin/bash
#SBATCH -J RAPCID_AF3_RAP
#SBATCH -p amd_h200nv_8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 16:00:00
#SBATCH --comment="field=sciai;appl=pytorch"
#SBATCH -o /home01/%u/RAPCID_2610/jobs/af3_plusRAP_%j.out
#SBATCH -e /home01/%u/RAPCID_2610/jobs/af3_plusRAP_%j.err

set -euo pipefail

USER_NAME="${USER}"

INPUT_DIR="/scratch/${USER_NAME}/RAPCID_2610_output/af3_plusRAP_inputs"
OUTPUT_ROOT="/scratch/${USER_NAME}/RAPCID_2610_output/af3_plusRAP"

SINGULARITY="/apps/applications/singularity/4.3.4/bin/singularity"
SIF="/scratch/${USER_NAME}/alphafold3_singularity/alphafold3.sif"
MODEL_DIR="/scratch/${USER_NAME}/alphafold3_models"

mapfile -t JSONS < <(
    find "$INPUT_DIR" -maxdepth 1 -type f -name '*.json' | sort
)

N=${#JSONS[@]}

for (( IDX=0; IDX<N; IDX++ )); do
    JSON="${JSONS[$IDX]}"
    NAME="$(basename "$JSON" .json)"
    OUTDIR="${OUTPUT_ROOT}/${NAME}"

    mkdir -p "$OUTDIR"

    "$SINGULARITY" exec --nv \
        -B /home01:/home01 \
        -B /scratch:/scratch \
        "$SIF" \
        python /app/alphafold/run_alphafold.py \
            --json_path="$JSON" \
            --model_dir="$MODEL_DIR" \
            --output_dir="$OUTDIR" \
            --run_data_pipeline=false
done
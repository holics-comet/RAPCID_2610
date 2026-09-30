#!/bin/bash
#SBATCH -J RAPCID_LMPNN
#SBATCH -p amd_a100nv_8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH --comment="field=sciai;appl=pytorch"
#SBATCH -o /home01/%u/RAPCID_2610/jobs/RAPCID_LMPNN_%j.out
#SBATCH -e /home01/%u/RAPCID_2610/jobs/RAPCID_LMPNN_%j.err

set -euo pipefail

ROOT="/home01/$USER/RAPCID_2610"
LMPNN="/scratch/$USER/LigandMPNN"
PYTHON="/scratch/$USER/rfd3/env/bin/python"

PDB_MULTI="$ROOT/inputs/ligandmpnn_pdb_paths.json"
FIXED_MULTI="$ROOT/inputs/ligandmpnn_fixed_residues.json"

OUT="/scratch/$USER/RAPCID_2610_output/ligandmpnn"

mkdir -p "$OUT"

cd "$LMPNN"

"$PYTHON" run.py \
    --model_type ligand_mpnn \
    --pdb_path_multi "$PDB_MULTI" \
    --fixed_residues_multi "$FIXED_MULTI" \
    --out_folder "$OUT" \
    --temperature 0.15 \
    --batch_size 8 \
    --number_of_batches 1 \
    --seed 42 \
    --ligand_mpnn_use_atom_context 1 \
    --ligand_mpnn_use_side_chain_context 1 \
    --save_stats 1 \
    --pack_side_chains 0 \
    --verbose 1

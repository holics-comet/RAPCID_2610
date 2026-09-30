#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import biotite.structure as struc
from biotite.structure.io import pdb


USER = os.environ["USER"]

ROOT = Path(f"/home01/{USER}/RAPCID_2610")
MANIFEST = ROOT / "analysis" / "ligandmpnn_input_manifest.tsv"

LMPNN_SEQ_DIR = Path(
    f"/scratch/{USER}/RAPCID_2610_output/ligandmpnn/seqs"
)

OUT_ROOT = Path(
    f"/scratch/{USER}/RAPCID_2610_output/ligandmpnn_filtered"
)
OUT_SEQ_DIR = OUT_ROOT / "seqs"

ANALYSIS = ROOT / "analysis"

TARGET_BACKBONES = 500
SEQS_PER_BACKBONE = 2

LIGAND_RESN = "RAP"
ATOM_BURIED_RASA = 0.25

TARGET_BURIED_ATOMS = {
    "O2","C3","C4","C5","O3",
    "C13","C18","C19","C20","C21","C22","C23","C24",
    "C36","C38","C41","C43","C44","C45","C46","C47",
    "C48","C49","C50","O6","O10","O11",
}

TARGET_PARTIAL_ATOMS = {
    "C42","C51","C52","O7","O8","O12","O13",
}

TARGET_EXPOSED_ATOMS = {"C40","O9"}

# Final score weights: 85% burial geometry, 15% LigandMPNN confidence
W_SASA_BURIAL = 0.45
W_ATOM_BURIAL = 0.25
W_TARGET_BURIED = 0.15
W_LIGAND_CONF = 0.10
W_OVERALL_CONF = 0.05


def read_manifest():
    with MANIFEST.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_pdb(path: Path):
    pf = pdb.PDBFile.read(path)
    return pdb.get_structure(pf, model=1)


def atom_sasa(arr):
    return np.asarray(struc.sasa(arr), dtype=float)


def rap_burial_metrics(pdb_path: Path):
    arr = load_pdb(pdb_path)

    # RFD3 virtual / pseudo atoms 제거
    atom_names = np.array([str(x).strip() for x in arr.atom_name])

    keep = (
        (arr.res_name != "UNK")
        & (~np.char.startswith(atom_names, "V"))
    )

    arr = arr[keep]

    rap_mask = arr.res_name == LIGAND_RESN
    rap = arr[rap_mask]

    if len(rap) == 0:
        raise ValueError(f"RAP not found: {pdb_path}")

    sasa_complex_all = atom_sasa(arr)
    sasa_complex = sasa_complex_all[rap_mask]

    sasa_isolated = atom_sasa(rap)

    total_iso = float(np.nansum(sasa_isolated))
    total_complex = float(np.nansum(sasa_complex))

    if total_iso <= 0:
        raise ValueError(f"Invalid isolated RAP SASA: {pdb_path}")

    sasa_burial_fraction = float(
        np.clip(1.0 - total_complex / total_iso, 0.0, 1.0)
    )

    valid = sasa_isolated > 1e-6
    rasa = np.ones(len(rap), dtype=float)
    rasa[valid] = sasa_complex[valid] / sasa_isolated[valid]
    rasa = np.clip(rasa, 0.0, 1.5)

    names = [str(x).strip() for x in rap.atom_name]
    valid_idx = np.where(valid)[0]

    atom_buried_fraction = (
        float(np.mean(rasa[valid_idx] <= ATOM_BURIED_RASA))
        if len(valid_idx) else 0.0
    )

    def class_stats(atom_names):
        idx = [
            i for i, name in enumerate(names)
            if name in atom_names and valid[i]
        ]
        if not idx:
            return {
                "n": 0,
                "buried_frac": float("nan"),
                "mean_rasa": float("nan"),
            }
        return {
            "n": len(idx),
            "buried_frac": float(
                np.mean(rasa[idx] <= ATOM_BURIED_RASA)
            ),
            "mean_rasa": float(np.mean(rasa[idx])),
        }

    target_buried = class_stats(TARGET_BURIED_ATOMS)
    target_partial = class_stats(TARGET_PARTIAL_ATOMS)
    target_exposed = class_stats(TARGET_EXPOSED_ATOMS)

    exposed_idx = [
        i for i, name in enumerate(names)
        if name in TARGET_EXPOSED_ATOMS and valid[i]
    ]
    exposed_preserved = (
        float(np.mean(rasa[exposed_idx] >= 0.50))
        if exposed_idx else float("nan")
    )

    return {
        "rap_sasa_isolated_A2": total_iso,
        "rap_sasa_complex_A2": total_complex,
        "rap_sasa_burial_fraction": sasa_burial_fraction,
        "rap_atom_buried_fraction": atom_buried_fraction,
        "target_buried_atom_fraction": target_buried["buried_frac"],
        "target_buried_mean_rasa": target_buried["mean_rasa"],
        "target_partial_atom_fraction": target_partial["buried_frac"],
        "target_partial_mean_rasa": target_partial["mean_rasa"],
        "target_exposed_atom_fraction": target_exposed["buried_frac"],
        "target_exposed_mean_rasa": target_exposed["mean_rasa"],
        "target_exposed_preserved_fraction": exposed_preserved,
    }


HEADER_RE = re.compile(
    r"id=(?P<id>\d+).*?"
    r"overall_confidence=(?P<overall>[0-9.]+).*?"
    r"ligand_confidence=(?P<ligand>[0-9.]+).*?"
    r"seq_rec=(?P<seqrec>[0-9.]+)"
)


def parse_fasta(path: Path):
    records = []
    header = None
    seq_parts = []

    def flush():
        nonlocal header, seq_parts
        if header is None:
            return
        m = HEADER_RE.search(header)
        if m:
            records.append({
                "sequence_id": int(m.group("id")),
                "overall_confidence": float(m.group("overall")),
                "ligand_confidence": float(m.group("ligand")),
                "seq_rec": float(m.group("seqrec")),
                "sequence": "".join(seq_parts).strip(),
                "header": header,
            })
        header = None
        seq_parts = []

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                header = line[1:]
            else:
                seq_parts.append(line)
    flush()
    return records


def safe_score(x, default=0.0):
    try:
        x = float(x)
        return default if math.isnan(x) else x
    except Exception:
        return default


def write_csv(path, rows):
    if not rows:
        return
    fields = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    OUT_SEQ_DIR.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest()
    print(f"Manifest backbones: {len(manifest)}")

    backbone_rows = []
    seq_rows = []

    for i, row in enumerate(manifest, 1):
        pdb_path = Path(row["pdb"])
        stem = pdb_path.stem
        fasta = LMPNN_SEQ_DIR / f"{stem}.fa"

        if not fasta.exists():
            print(f"[WARN] FASTA missing: {fasta}")
            continue

        burial = rap_burial_metrics(pdb_path)

        burial_score = (
            W_SASA_BURIAL * safe_score(
                burial["rap_sasa_burial_fraction"]
            )
            + W_ATOM_BURIAL * safe_score(
                burial["rap_atom_buried_fraction"]
            )
            + W_TARGET_BURIED * safe_score(
                burial["target_buried_atom_fraction"]
            )
        )

        backbone_entry = {
            "group": row["group"],
            "backbone": stem,
            "pdb": str(pdb_path),
            "fasta": str(fasta),
            **burial,
            "burial_score_raw": burial_score,
        }
        backbone_rows.append(backbone_entry)

        for rec in parse_fasta(fasta):
            final_score = (
                burial_score
                + W_LIGAND_CONF * rec["ligand_confidence"]
                + W_OVERALL_CONF * rec["overall_confidence"]
            )
            seq_rows.append({
                **backbone_entry,
                **rec,
                "final_score": final_score,
            })

        if i % 50 == 0 or i == len(manifest):
            print(f"  {i}/{len(manifest)}")

    backbone_rows.sort(
        key=lambda r: (
            r["rap_sasa_burial_fraction"],
            r["rap_atom_buried_fraction"],
            safe_score(r["target_buried_atom_fraction"]),
            r["burial_score_raw"],
        ),
        reverse=True,
    )

    selected_backbones = backbone_rows[:TARGET_BACKBONES]
    selected_names = {r["backbone"] for r in selected_backbones}
    backbone_rank = {
        r["backbone"]: rank
        for rank, r in enumerate(selected_backbones, 1)
    }

    by_backbone = defaultdict(list)
    for row in seq_rows:
        if row["backbone"] in selected_names:
            by_backbone[row["backbone"]].append(row)

    selected_seq_rows = []

    for backbone in selected_names:
        candidates = by_backbone[backbone]
        candidates.sort(
            key=lambda r: (
                r["ligand_confidence"],
                r["overall_confidence"],
                r["final_score"],
            ),
            reverse=True,
        )
        for local_rank, row in enumerate(
            candidates[:SEQS_PER_BACKBONE], 1
        ):
            row["backbone_rank"] = backbone_rank[backbone]
            row["sequence_rank_within_backbone"] = local_rank
            selected_seq_rows.append(row)

    selected_seq_rows.sort(
        key=lambda r: (
            r["backbone_rank"],
            r["sequence_rank_within_backbone"],
        )
    )

    backbone_csv = ANALYSIS / "ligandmpnn_backbone_burial.csv"
    all_seq_csv = ANALYSIS / "ligandmpnn_all_sequences.csv"
    selected_csv = ANALYSIS / "ligandmpnn_selected_sequences.csv"

    write_csv(backbone_csv, backbone_rows)
    write_csv(all_seq_csv, seq_rows)
    write_csv(selected_csv, selected_seq_rows)

    combined = OUT_ROOT / "selected_sequences.fasta"
    with combined.open("w") as all_fa:
        for row in selected_seq_rows:
            header = (
                f"{row['backbone']}_seq{row['sequence_id']}"
                f"|group={row['group']}"
                f"|backbone_rank={row['backbone_rank']}"
                f"|burial={row['rap_sasa_burial_fraction']:.4f}"
                f"|atom_buried={row['rap_atom_buried_fraction']:.4f}"
                f"|lig_conf={row['ligand_confidence']:.4f}"
                f"|overall_conf={row['overall_confidence']:.4f}"
            )
            seq = row["sequence"]
            all_fa.write(f">{header}\n{seq}\n")
            (OUT_SEQ_DIR / f"{row['backbone']}_seq{row['sequence_id']}.fa").write_text(
                f">{header}\n{seq}\n"
            )

    print("\n=== LigandMPNN filtering summary ===")
    print(f"Backbones analyzed : {len(backbone_rows)}")
    print(f"Sequences analyzed : {len(seq_rows)}")
    print(f"Backbones selected : {len(selected_backbones)}")
    print(f"Sequences selected : {len(selected_seq_rows)}")

    if selected_backbones:
        vals = np.array([
            r["rap_sasa_burial_fraction"]
            for r in selected_backbones
        ])
        print(
            "Selected RAP SASA burial: "
            f"min={vals.min():.3f}, "
            f"median={np.median(vals):.3f}, "
            f"max={vals.max():.3f}"
        )

    print(f"\nBackbone CSV  : {backbone_csv}")
    print(f"All seq CSV   : {all_seq_csv}")
    print(f"Selected CSV  : {selected_csv}")
    print(f"Selected FASTA: {combined}")
    print(f"Per-seq FASTA : {OUT_SEQ_DIR}")


if __name__ == "__main__":
    main()

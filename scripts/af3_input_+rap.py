#!/usr/bin/env python3

import csv
import json
import os
import re
from pathlib import Path


USER = os.environ["USER"]

ROOT = Path(f"/home01/{USER}/RAPCID_2610")
SELECTED_CSV = ROOT / "analysis" / "C. Sequence selection.csv"

OUT_DIR = Path(
    f"/scratch/{USER}/RAPCID_2610_output/af3_plusRAP_inputs"
)

MANIFEST = ROOT / "analysis" / "af3_plusRAP_manifest.tsv"

MODEL_SEEDS = [1]

LIGAND_ID = "C"
LIGAND_CCD = "RAP"


def sanitize_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return name.strip("_")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Clear only old JSONs in this AF3 input directory.
    for p in OUT_DIR.glob("*.json"):
        p.unlink()

    with SELECTED_CSV.open() as f:
        rows = list(csv.DictReader(f))

    manifest_rows = []

    n_ok = 0
    n_bad = 0

    for row in rows:
        sequence = row["sequence"].strip()

        parts = sequence.split(":")

        if len(parts) != 2:
            print(
                f"[SKIP] expected exactly 2 chains: "
                f"{row.get('backbone', '')} seq={row.get('sequence_id', '')}"
            )
            n_bad += 1
            continue

        seq_a, seq_b = [x.strip() for x in parts]

        if not seq_a or not seq_b:
            print(
                f"[SKIP] empty chain: "
                f"{row.get('backbone', '')} seq={row.get('sequence_id', '')}"
            )
            n_bad += 1
            continue

        backbone = row["backbone"]
        seq_id = row["sequence_id"]

        job_name = sanitize_name(
            f"{backbone}_seq{seq_id}_plusRAP"
        )

        af3_input = {
            "name": job_name,
            "modelSeeds": MODEL_SEEDS,
            "sequences": [
                {
                    "protein": {
                        "id": "A",
                        "sequence": seq_a,
                        "unpairedMsa": "",
                        "pairedMsa": "",
                        "templates": []
                    }
                },
                {
                    "protein": {
                        "id": "B",
                        "sequence": seq_b,
                        "unpairedMsa": "",
                        "pairedMsa": "",
                        "templates": []
                    }
                },
                {
                    "ligand": {
                        "id": LIGAND_ID,
                        "ccdCodes": [LIGAND_CCD]
                    }
                }
            ],
            "dialect": "alphafold3",
            "version": 4
        }

        out_json = OUT_DIR / f"{job_name}.json"

        with out_json.open("w") as f:
            json.dump(af3_input, f, indent=2)

        manifest_rows.append({
            "job_name": job_name,
            "json_path": str(out_json),
            "group": row.get("group", ""),
            "backbone": backbone,
            "sequence_id": seq_id,
            "chainA_length": len(seq_a),
            "chainB_length": len(seq_b),
            "rap_sasa_burial_fraction":
                row.get("rap_sasa_burial_fraction", ""),
            "rap_atom_buried_fraction":
                row.get("rap_atom_buried_fraction", ""),
            "ligand_confidence":
                row.get("ligand_confidence", ""),
            "overall_confidence":
                row.get("overall_confidence", ""),
            "final_score":
                row.get("final_score", "")
        })

        n_ok += 1

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    fields = list(manifest_rows[0].keys()) if manifest_rows else []

    with MANIFEST.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print()
    print("=== AF3 +RAP input generation ===")
    print(f"Selected rows : {len(rows)}")
    print(f"JSON created  : {n_ok}")
    print(f"Skipped       : {n_bad}")
    print(f"Input dir     : {OUT_DIR}")
    print(f"Manifest      : {MANIFEST}")


if __name__ == "__main__":
    main()

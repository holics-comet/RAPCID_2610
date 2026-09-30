#!/usr/bin/env python3
from __future__ import annotations

import gzip
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from biotite.structure.io import pdb, pdbx


USER = os.environ["USER"]

ROOT = Path(f"/home01/{USER}/RAPCID_2610")
FILTERED = Path(f"/scratch/{USER}/RAPCID_2610_output/backbone_filtered")

PDB_OUT = Path(f"/scratch/{USER}/RAPCID_2610_output/ligandmpnn_inputs")
JSON_OUT = ROOT / "inputs"

GROUPS = ("backbone_denovo", "backbone_long", "backbone")
CONSERVED_GROUPS = {"backbone_long", "backbone"}

LIGAND_RESN = "RAP"


def load_structure(path: Path):
    suffix = "".join(path.suffixes).lower()

    if suffix.endswith(".cif.gz"):
        with gzip.open(path, "rt") as f:
            cf = pdbx.CIFFile.read(f)
        arr = pdbx.get_structure(cf, model=1)

    elif suffix.endswith(".cif"):
        cf = pdbx.CIFFile.read(path)
        arr = pdbx.get_structure(cf, model=1)

    elif suffix.endswith(".pdb.gz"):
        with gzip.open(path, "rt") as f:
            pf = pdb.PDBFile.read(f)
        arr = pdb.get_structure(pf, model=1)

    elif suffix.endswith(".pdb"):
        pf = pdb.PDBFile.read(path)
        arr = pdb.get_structure(pf, model=1)

    else:
        raise ValueError(f"Unsupported file: {path}")

    return arr


def structure_files(directory: Path):
    out = []
    for pattern in ("*.cif.gz", "*.cif", "*.pdb.gz", "*.pdb"):
        out.extend(directory.glob(pattern))
    return sorted(set(out))


def protein_chains(arr):
    protein = arr[(arr.res_name != LIGAND_RESN) & (~arr.hetero)]
    ca = protein[protein.atom_name == "CA"]

    sizes = defaultdict(int)
    for c in ca.chain_id:
        sizes[str(c)] += 1

    chains = sorted(sizes, key=sizes.get, reverse=True)[:2]

    if len(chains) != 2:
        raise ValueError(f"Expected 2 protein chains, found {dict(sizes)}")

    return protein, chains


def residue_key(chain, res_id, ins_code=""):
    return f"{chain}{int(res_id)}{str(ins_code).strip()}"


def closest_tyr_to_rap(arr, chain):
    rap = arr[arr.res_name == LIGAND_RESN]
    if len(rap) == 0:
        raise ValueError("RAP not found")

    tyr = arr[
        (arr.chain_id == chain)
        & (arr.res_name == "TYR")
        & (~arr.hetero)
    ]

    if len(tyr) == 0:
        raise ValueError(f"No TYR found in conserved chain {chain}")

    residues = defaultdict(list)

    for i, atom in enumerate(tyr):
        ins = ""
        if hasattr(tyr, "ins_code"):
            ins = str(tyr.ins_code[i]).strip()

        key = (str(atom.chain_id), int(atom.res_id), ins)
        residues[key].append(i)

    best_key = None
    best_d = float("inf")

    for key, idx in residues.items():
        coords = tyr.coord[idx]
        d = np.sqrt(
            np.min(
                np.sum(
                    (coords[:, None, :] - rap.coord[None, :, :]) ** 2,
                    axis=2,
                )
            )
        )

        if d < best_d:
            best_d = float(d)
            best_key = key

    chain_id, res_id, ins = best_key
    return residue_key(chain_id, res_id, ins), best_d


def write_pdb(arr, path: Path):
    pf = pdb.PDBFile()
    pf.set_structure(arr)
    pf.write(path)


def main():
    PDB_OUT.mkdir(parents=True, exist_ok=True)
    JSON_OUT.mkdir(parents=True, exist_ok=True)

    pdb_paths = {}
    fixed = {}
    manifest = []

    total = 0

    for group in GROUPS:
        source_dir = FILTERED / group
        files = structure_files(source_dir)

        print(f"[{group}] {len(files)} structures")

        group_out = PDB_OUT / group
        group_out.mkdir(parents=True, exist_ok=True)

        for src in files:
            arr = load_structure(src)
            _, chains = protein_chains(arr)

            stem = src.name
            for suffix in (".cif.gz", ".pdb.gz", ".cif", ".pdb"):
                if stem.endswith(suffix):
                    stem = stem[:-len(suffix)]
                    break

            dst = group_out / f"{stem}.pdb"
            write_pdb(arr, dst)

            dst_abs = str(dst.resolve())
            pdb_paths[dst_abs] = ""

            fixed_string = ""
            motif_info = ""

            if group in CONSERVED_GROUPS:
                fixed_res = []

                for chain in chains:
                    rid, dist = closest_tyr_to_rap(arr, chain)
                    fixed_res.append(rid)
                    motif_info += f"{rid}:{dist:.2f}A;"

                fixed_string = " ".join(fixed_res)

            fixed[dst_abs] = fixed_string

            manifest.append({
                "group": group,
                "source": str(src),
                "pdb": dst_abs,
                "fixed_residues": fixed_string,
                "motif_info": motif_info,
            })

            total += 1

    pdb_json = JSON_OUT / "ligandmpnn_pdb_paths.json"
    fixed_json = JSON_OUT / "ligandmpnn_fixed_residues.json"
    manifest_tsv = ROOT / "analysis" / "ligandmpnn_input_manifest.tsv"

    pdb_json.write_text(json.dumps(pdb_paths, indent=2) + "\n")
    fixed_json.write_text(json.dumps(fixed, indent=2) + "\n")

    manifest_tsv.parent.mkdir(parents=True, exist_ok=True)

    with manifest_tsv.open("w") as f:
        f.write("group\tsource\tpdb\tfixed_residues\tmotif_info\n")
        for row in manifest:
            f.write(
                f"{row['group']}\t{row['source']}\t{row['pdb']}\t"
                f"{row['fixed_residues']}\t{row['motif_info']}\n"
            )

    print()
    print(f"Prepared PDBs: {total}")
    print(f"PDB list:       {pdb_json}")
    print(f"Fixed residues: {fixed_json}")
    print(f"Manifest:       {manifest_tsv}")
    print(f"PDB directory:  {PDB_OUT}")


if __name__ == "__main__":
    main()

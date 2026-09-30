#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from biotite.structure.io import pdb, pdbx


# ============================================================
# Paths
# ============================================================

USER = os.environ["USER"]
ROOT = Path(f"/home01/{USER}/RAPCID_2610")
SCRATCH = Path(f"/scratch/{USER}/RAPCID_2610_output")
ANALYSIS = ROOT / "analysis"

GROUPS = {
    "backbone_denovo": SCRATCH / "backbone_denovo",
    "backbone_long": SCRATCH / "backbone_long",
    "backbone": SCRATCH / "backbone",
}

# Reuse the same output location
FILTERED_ROOT = SCRATCH / "backbone_filtered"

LIGAND_RESN = "RAP"


# ============================================================
# Hard-filter thresholds v2
# ============================================================

CLASH_CUTOFF = 2.0

RAP_CONTACT_CUTOFF = 4.5
DIRECT_CONTACT_CUTOFF = 4.0
HOTSPOT_CUTOFF = 4.5

# Much looser than v1:
MIN_RAP_CONTACT_RESIDUES_PER_CHAIN = 2
MIN_RAP_CONTACT_ATOMS_PER_CHAIN = 4

MIN_CONTACT_BALANCE = 0.10
MAX_CONTACT_BALANCE = 0.90

# Keep this deliberately permissive.
MIN_FACE_ANGLE_DEG = 60.0

# Only reject very large direct interfaces.
MAX_DIRECT_RESIDUE_PAIRS = 40

# Conserved sets only need at least one TYR per chain.
# Do NOT hard-filter by TYR-RAP distance.
CONSERVED_GROUPS = {"backbone_long", "backbone"}

TARGET_TOTAL = 1000
REFERENCE_FACE_ANGLE = 159.0

# Soft-ranking reference features
FACE_E = ("O2", "O3", "C5")
FACE_F = ("C21", "C45", "C50")
SHARED_SEAM = ("C49",)

REACH_MIN_A = 4.0
REACH_MAX_A = 10.0


# ============================================================
# IO
# ============================================================

def load_structure(path: Path):
    suffixes = "".join(path.suffixes).lower()

    if suffixes.endswith(".cif.gz"):
        with gzip.open(path, "rt") as f:
            cf = pdbx.CIFFile.read(f)
        arr = pdbx.get_structure(cf, model=1)

    elif suffixes.endswith(".cif"):
        cf = pdbx.CIFFile.read(path)
        arr = pdbx.get_structure(cf, model=1)

    elif suffixes.endswith(".pdb.gz"):
        with gzip.open(path, "rt") as f:
            pf = pdb.PDBFile.read(f)
        arr = pdb.get_structure(pf, model=1)

    elif suffixes.endswith(".pdb"):
        pf = pdb.PDBFile.read(path)
        arr = pdb.get_structure(pf, model=1)

    else:
        raise ValueError(f"Unsupported structure format: {path}")

    if hasattr(arr, "element"):
        arr = arr[np.char.upper(arr.element.astype(str)) != "H"]

    return arr


def structure_files(directory: Path):
    result = []
    for pattern in ("*.cif.gz", "*.cif", "*.pdb.gz", "*.pdb"):
        result.extend(directory.rglob(pattern))
    return sorted(set(result))


# ============================================================
# Geometry helpers
# ============================================================

def pairwise_min_distance(a, b):
    if len(a) == 0 or len(b) == 0:
        return float("inf")

    best = float("inf")
    for i in range(0, len(a), 256):
        d2 = np.sum((a[i:i+256, None, :] - b[None, :, :]) ** 2, axis=2)
        best = min(best, float(np.sqrt(d2.min())))
    return best


def min_distances(source, target):
    if len(source) == 0 or len(target) == 0:
        return np.full(len(source), np.inf)

    out = np.full(len(source), np.inf)

    for i in range(0, len(source), 256):
        d2 = np.sum(
            (source[i:i+256, None, :] - target[None, :, :]) ** 2,
            axis=2,
        )
        out[i:i+256] = np.sqrt(d2.min(axis=1))

    return out


def residue_keys(arr):
    return np.array(
        [
            f"{chain}:{resi}:{resn}"
            for chain, resi, resn
            in zip(arr.chain_id, arr.res_id, arr.res_name)
        ],
        dtype=object,
    )


def identify_components(arr):
    ligand = arr[arr.res_name == LIGAND_RESN]
    if len(ligand) == 0:
        raise ValueError("RAP not found")

    protein = arr[(arr.res_name != LIGAND_RESN) & (~arr.hetero)]
    ca = protein[protein.atom_name == "CA"]

    sizes = defaultdict(int)
    for c in ca.chain_id:
        sizes[str(c)] += 1

    if len(sizes) < 2:
        raise ValueError(f"Expected >=2 protein chains, found {dict(sizes)}")

    chains = sorted(sizes, key=sizes.get, reverse=True)[:2]

    c1 = protein[protein.chain_id == chains[0]]
    c2 = protein[protein.chain_id == chains[1]]

    return c1, c2, ligand, chains


def count_unique_contact_residues(chain, target, cutoff):
    d = min_distances(chain.coord, target.coord)
    hit = d <= cutoff
    keys = residue_keys(chain)

    return len(set(keys[hit])), int(hit.sum())


def count_direct_residue_pairs(c1, c2, cutoff):
    k1 = residue_keys(c1)
    k2 = residue_keys(c2)
    cutoff2 = cutoff * cutoff
    pairs = set()

    for i in range(0, len(c1), 256):
        d2 = np.sum(
            (c1.coord[i:i+256, None, :] - c2.coord[None, :, :]) ** 2,
            axis=2,
        )
        ii, jj = np.where(d2 <= cutoff2)

        for x, y in zip(ii, jj):
            pairs.add((k1[i + x], k2[y]))

    return len(pairs)


def angle_deg(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 == 0 or n2 == 0:
        return float("nan")

    x = np.dot(v1, v2) / (n1 * n2)
    return math.degrees(math.acos(np.clip(float(x), -1.0, 1.0)))


def ligand_face_angle(c1, c2, ligand):
    d1 = min_distances(c1.coord, ligand.coord)
    d2 = min_distances(c2.coord, ligand.coord)

    a1 = c1.coord[d1 <= RAP_CONTACT_CUTOFF]
    a2 = c2.coord[d2 <= RAP_CONTACT_CUTOFF]

    if len(a1) == 0 or len(a2) == 0:
        return float("nan")

    center = ligand.coord.mean(axis=0)
    v1 = a1.mean(axis=0) - center
    v2 = a2.mean(axis=0) - center

    return angle_deg(v1, v2)


def tripartite_count(chain, other, ligand):
    d_lig = min_distances(chain.coord, ligand.coord)
    d_other = min_distances(chain.coord, other.coord)

    keys = residue_keys(chain)
    status = defaultdict(lambda: [False, False])

    for key, dl, do in zip(keys, d_lig, d_other):
        if dl <= DIRECT_CONTACT_CUTOFF:
            status[key][0] = True
        if do <= DIRECT_CONTACT_CUTOFF:
            status[key][1] = True

    return sum(v[0] and v[1] for v in status.values())


def ligand_atom_map(ligand):
    return {
        str(name): coord
        for name, coord in zip(ligand.atom_name, ligand.coord)
    }


def hotspot_count(chain, ligand_map, names):
    n = 0

    for name in names:
        if name not in ligand_map:
            continue

        d = np.min(
            np.linalg.norm(chain.coord - ligand_map[name], axis=1)
        )

        if d <= HOTSPOT_CUTOFF:
            n += 1

    return n


def face_metrics(c1, c2, ligand):
    lm = ligand_atom_map(ligand)

    c1_e = hotspot_count(c1, lm, FACE_E)
    c1_f = hotspot_count(c1, lm, FACE_F)
    c2_e = hotspot_count(c2, lm, FACE_E)
    c2_f = hotspot_count(c2, lm, FACE_F)

    if c1_e + c2_f >= c1_f + c2_e:
        orientation = "chain1_E_chain2_F"
        assigned_1 = c1_e
        assigned_2 = c2_f
    else:
        orientation = "chain1_F_chain2_E"
        assigned_1 = c1_f
        assigned_2 = c2_e

    seam = int(
        hotspot_count(c1, lm, SHARED_SEAM)
        + hotspot_count(c2, lm, SHARED_SEAM)
        > 0
    )

    return {
        "orientation": orientation,
        "chain1_E_hotspots": c1_e,
        "chain1_F_hotspots": c1_f,
        "chain2_E_hotspots": c2_e,
        "chain2_F_hotspots": c2_f,
        "assigned_face1_hotspots": assigned_1,
        "assigned_face2_hotspots": assigned_2,
        "shared_seam_contact": seam,
    }


def reachable_count(chain, ligand, names):
    lm = ligand_atom_map(ligand)
    ca = chain[chain.atom_name == "CA"]

    if len(ca) == 0:
        return 0

    keys = residue_keys(ca)
    hit_res = set()

    for name in names:
        if name not in lm:
            continue

        d = np.linalg.norm(ca.coord - lm[name], axis=1)
        hit = (d >= REACH_MIN_A) & (d <= REACH_MAX_A)

        hit_res.update(keys[hit])

    return len(hit_res)


# ============================================================
# Single structure
# ============================================================

def analyze(path: Path, group: str):
    arr = load_structure(path)
    c1, c2, ligand, chains = identify_components(arr)

    n1 = len(c1[c1.atom_name == "CA"])
    n2 = len(c2[c2.atom_name == "CA"])

    min_pp = pairwise_min_distance(c1.coord, c2.coord)
    min_1_rap = pairwise_min_distance(c1.coord, ligand.coord)
    min_2_rap = pairwise_min_distance(c2.coord, ligand.coord)

    rap_res1, rap_atoms1 = count_unique_contact_residues(
        c1, ligand, RAP_CONTACT_CUTOFF
    )
    rap_res2, rap_atoms2 = count_unique_contact_residues(
        c2, ligand, RAP_CONTACT_CUTOFF
    )

    total = rap_atoms1 + rap_atoms2
    balance = rap_atoms1 / total if total else 0.0

    face_angle = ligand_face_angle(c1, c2, ligand)
    direct_pairs = count_direct_residue_pairs(
        c1, c2, DIRECT_CONTACT_CUTOFF
    )

    trip1 = tripartite_count(c1, c2, ligand)
    trip2 = tripartite_count(c2, c1, ligand)

    face = face_metrics(c1, c2, ligand)

    reach1_e = reachable_count(c1, ligand, FACE_E)
    reach1_f = reachable_count(c1, ligand, FACE_F)
    reach2_e = reachable_count(c2, ligand, FACE_E)
    reach2_f = reachable_count(c2, ligand, FACE_F)

    if face["orientation"] == "chain1_E_chain2_F":
        assigned_reach1 = reach1_e
        assigned_reach2 = reach2_f
    else:
        assigned_reach1 = reach1_f
        assigned_reach2 = reach2_e

    # For conserved sets: require Tyr existence only.
    tyr1 = len(
        set(
            residue_keys(
                c1[c1.res_name == "TYR"]
            )
        )
    )
    tyr2 = len(
        set(
            residue_keys(
                c2[c2.res_name == "TYR"]
            )
        )
    )

    pass_conserved = True
    if group in CONSERVED_GROUPS:
        pass_conserved = (tyr1 >= 1 and tyr2 >= 1)

    checks = {
        "pass_no_clash": (
            min_pp >= CLASH_CUTOFF
            and min_1_rap >= CLASH_CUTOFF
            and min_2_rap >= CLASH_CUTOFF
        ),

        "pass_dual_rap_contact": (
            rap_res1 >= MIN_RAP_CONTACT_RESIDUES_PER_CHAIN
            and rap_res2 >= MIN_RAP_CONTACT_RESIDUES_PER_CHAIN
            and rap_atoms1 >= MIN_RAP_CONTACT_ATOMS_PER_CHAIN
            and rap_atoms2 >= MIN_RAP_CONTACT_ATOMS_PER_CHAIN
        ),

        "pass_contact_balance": (
            MIN_CONTACT_BALANCE
            <= balance
            <= MAX_CONTACT_BALANCE
        ),

        "pass_face_angle": (
            not math.isnan(face_angle)
            and face_angle >= MIN_FACE_ANGLE_DEG
        ),

        "pass_direct_interface": (
            direct_pairs <= MAX_DIRECT_RESIDUE_PAIRS
        ),

        "pass_conserved_motif": pass_conserved,
    }

    hard_pass = all(checks.values())

    # ========================================================
    # Soft score
    # ========================================================

    angle_score = max(
        0.0,
        1.0 - abs(face_angle - REFERENCE_FACE_ANGLE) / 90.0
    )

    balance_score = max(
        0.0,
        1.0 - abs(balance - 0.5) / 0.5
    )

    hotspot_score = min(
        (
            face["assigned_face1_hotspots"]
            + face["assigned_face2_hotspots"]
            + face["shared_seam_contact"]
        ) / 7.0,
        1.0,
    )

    trip_score = min((trip1 + trip2) / 6.0, 1.0)

    reach_score = min(
        (assigned_reach1 + assigned_reach2) / 10.0,
        1.0,
    )

    # Prefer a small cooperative seam.
    # 0 pairs is allowed by hard filter, but soft rank favors ~8.
    direct_score = max(
        0.0,
        1.0 - abs(direct_pairs - 8) / 20.0
    )

    soft_score = (
        0.30 * angle_score
        + 0.20 * balance_score
        + 0.20 * hotspot_score
        + 0.10 * trip_score
        + 0.10 * reach_score
        + 0.10 * direct_score
    )

    failed = [
        name
        for name, passed in checks.items()
        if not passed
    ]

    return {
        "group": group,
        "file": path.name,
        "path": str(path),

        "chain1": chains[0],
        "chain2": chains[1],

        "chain1_length": n1,
        "chain2_length": n2,

        "min_chain_chain_dist_A": round(min_pp, 3),
        "min_chain1_RAP_dist_A": round(min_1_rap, 3),
        "min_chain2_RAP_dist_A": round(min_2_rap, 3),

        "chain1_RAP_contact_residues": rap_res1,
        "chain2_RAP_contact_residues": rap_res2,

        "chain1_RAP_contact_atoms": rap_atoms1,
        "chain2_RAP_contact_atoms": rap_atoms2,

        "RAP_contact_balance_chain1": round(balance, 4),
        "RAP_face_angle_deg": round(face_angle, 2),

        "orientation": face["orientation"],

        "chain1_E_hotspots": face["chain1_E_hotspots"],
        "chain1_F_hotspots": face["chain1_F_hotspots"],
        "chain2_E_hotspots": face["chain2_E_hotspots"],
        "chain2_F_hotspots": face["chain2_F_hotspots"],

        "assigned_face1_hotspots": face["assigned_face1_hotspots"],
        "assigned_face2_hotspots": face["assigned_face2_hotspots"],
        "shared_seam_contact": face["shared_seam_contact"],

        "direct_residue_pairs_4A": direct_pairs,

        "chain1_tripartite_residues": trip1,
        "chain2_tripartite_residues": trip2,

        "assigned_chain1_reachable_residues": assigned_reach1,
        "assigned_chain2_reachable_residues": assigned_reach2,

        "chain1_TYR_count": tyr1,
        "chain2_TYR_count": tyr2,

        **{k: int(v) for k, v in checks.items()},

        "hard_pass": int(hard_pass),
        "failed_filters": ";".join(failed),

        "soft_score": round(soft_score, 5),
    }


# ============================================================
# CSV helpers
# ============================================================

def write_csv(path, rows, fields=None):
    if not rows:
        return

    fields = fields or list(rows[0].keys())

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {k: row.get(k, "") for k in fields}
            )


# ============================================================
# Main
# ============================================================

def main():
    ANALYSIS.mkdir(parents=True, exist_ok=True)

    # Reuse output folder:
    # wipe only previous filtered copies, never source backbone folders.
    if FILTERED_ROOT.exists():
        shutil.rmtree(FILTERED_ROOT)

    FILTERED_ROOT.mkdir(parents=True, exist_ok=True)

    all_rows = []
    summary = []

    for group, directory in GROUPS.items():
        fs = structure_files(directory)

        print(f"\n[{group}] {len(fs)} structures")

        rows = []
        errors = 0

        for i, path in enumerate(fs, 1):
            try:
                rows.append(analyze(path, group))

            except Exception as exc:
                errors += 1

                rows.append(
                    {
                        "group": group,
                        "file": path.name,
                        "path": str(path),
                        "hard_pass": 0,
                        "failed_filters":
                            f"ERROR:{type(exc).__name__}:{exc}",
                        "soft_score": -1.0,
                    }
                )

            if i % 100 == 0 or i == len(fs):
                print(f"  {i}/{len(fs)}")

        valid = [
            r for r in rows
            if "chain1" in r
        ]

        hard_passed = [
            r for r in valid
            if r["hard_pass"] == 1
        ]

        hard_passed.sort(
            key=lambda r: r["soft_score"],
            reverse=True,
        )

        summary.append(
            {
                "group": group,
                "input_count": len(fs),
                "analysis_errors": errors,
                "hard_pass_count": len(hard_passed),
                "hard_pass_fraction":
                    round(len(hard_passed) / len(fs), 4)
                    if fs else 0,
            }
        )

        all_rows.extend(rows)

        print(
            f"  hard pass = {len(hard_passed)} / {len(fs)}"
        )

    # ========================================================
    # Global selection:
    # choose top 1000 from ALL hard-pass structures together
    # ========================================================

    global_passed = [
        r for r in all_rows
        if r.get("hard_pass") == 1
    ]

    global_passed.sort(
        key=lambda r: r["soft_score"],
        reverse=True,
    )

    selected = global_passed[:TARGET_TOTAL]

    chosen = {
        r["path"]
        for r in selected
    }

    # Copy into group-specific folders
    for rank, row in enumerate(selected, 1):
        group = row["group"]

        outdir = FILTERED_ROOT / group
        outdir.mkdir(parents=True, exist_ok=True)

        src = Path(row["path"])
        dst = outdir / src.name

        shutil.copy2(src, dst)

        row["selected_final"] = 1
        row["selected_rank_global"] = rank

    for row in all_rows:
        if row.get("path") not in chosen:
            row["selected_final"] = 0
            row["selected_rank_global"] = ""

    # Count selected per group
    for entry in summary:
        group = entry["group"]

        entry["selected_count"] = sum(
            r.get("selected_final") == 1
            and r.get("group") == group
            for r in all_rows
        )

        entry["selected_directory"] = str(
            FILTERED_ROOT / group
        )

    # Per-group CSV
    for group in GROUPS:
        rows = [
            r for r in all_rows
            if r.get("group") == group
        ]

        if rows:
            fields = []
            for row in rows:
                for key in row:
                    if key not in fields:
                        fields.append(key)

            write_csv(
                ANALYSIS / f"backbone_filter_{group}.csv",
                rows,
                fields,
            )

    # Combined CSV
    fields = []
    for row in all_rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    write_csv(
        ANALYSIS / "backbone_filter_all.csv",
        all_rows,
        fields,
    )

    write_csv(
        ANALYSIS / "backbone_filter_summary.csv",
        summary,
    )

    print("\n=== Summary ===")

    for row in summary:
        print(
            f"{row['group']}: "
            f"input={row['input_count']} "
            f"hard_pass={row['hard_pass_count']} "
            f"selected={row['selected_count']}"
        )

    print(
        f"\nTotal hard-pass = {len(global_passed)}"
    )

    print(
        f"Final selected = {len(selected)}"
    )

    print(
        f"\nFiltered structures -> {FILTERED_ROOT}"
    )

    print(
        f"CSV -> {ANALYSIS}"
    )


if __name__ == "__main__":
    main()

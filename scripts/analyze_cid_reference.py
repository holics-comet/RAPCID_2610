#!/usr/bin/env python3
"""
Analyze the 6M4U FKBP-RAP-FRB reference for dual-chain de novo CID design.

Project layout
--------------
~/RAPCID_2610/
├── scripts/
├── inputs/
│   └── 6M4U_mod.pdb
├── jobs/
└── analysis/

Outputs
-------
analysis/cid_summary.csv
analysis/rap_atom_proximity.csv
analysis/ef_residue_contacts.csv
analysis/tripartite_residues.csv
analysis/cid_hotspot_candidates.csv

Definitions
-----------
- RAP contact: heavy-atom distance <= 4.0 A
- proximity shell: heavy-atom distance <= 5.0 A
- direct E-F contact: heavy-atom distance <= 4.0 A
- polar candidate: N/O/S heavy-atom pair <= 3.5 A
  (geometry-only candidate; NOT a confirmed H-bond without H/angle)
- salt-bridge candidate: charged side-chain atom pair <= 4.0 A
- hydrophobic candidate: C/S pair <= 4.5 A
- tripartite residue: contacts RAP AND the opposite protein chain
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from typing import Iterable

import numpy as np


# ============================================================
# Paths
# ============================================================

ROOT = Path(os.path.expanduser("~/RAPCID_2610"))
INPUT_PDB = ROOT / "inputs" / "6M4U_mod.pdb"
ANALYSIS_DIR = ROOT / "analysis"

CHAIN_1 = "E"       # FKBP-like side in 6M4U
CHAIN_2 = "F"       # FRB-like side in 6M4U
LIGAND_RESN = "RAP"


# ============================================================
# Distance cutoffs
# ============================================================

CONTACT_CUTOFF = 4.0
PROXIMITY_CUTOFF = 5.0
POLAR_CUTOFF = 3.5
SALT_CUTOFF = 4.0
HYDROPHOBIC_CUTOFF = 4.5

# Contact-centroid angle around RAP.
# 6M4U reference should be around ~150-160 degrees.
ANGLE_CONTACT_CUTOFF = 4.5


# ============================================================
# Atom chemistry
# ============================================================

POLAR_ELEMENTS = {"N", "O", "S"}
HYDROPHOBIC_ELEMENTS = {"C", "S"}

POSITIVE_ATOMS = {
    ("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2"),
    ("LYS", "NZ"),
    ("HIS", "ND1"), ("HIS", "NE2"),
}

NEGATIVE_ATOMS = {
    ("ASP", "OD1"), ("ASP", "OD2"),
    ("GLU", "OE1"), ("GLU", "OE2"),
}


@dataclass(frozen=True)
class Atom:
    record: str
    serial: int
    name: str
    altloc: str
    resn: str
    chain: str
    resi: int
    icode: str
    coord: np.ndarray
    element: str

    @property
    def residue_id(self) -> tuple[str, int, str, str]:
        return self.chain, self.resi, self.icode, self.resn

    @property
    def residue_label(self) -> str:
        icode = self.icode if self.icode else ""
        return f"{self.chain}:{self.resn}{self.resi}{icode}"


def infer_element(atom_name: str) -> str:
    name = atom_name.strip()
    while name and name[0].isdigit():
        name = name[1:]
    if not name:
        return ""
    return name[0].upper()


def parse_pdb(path: Path) -> list[Atom]:
    atoms: list[Atom] = []

    with path.open() as fh:
        for line in fh:
            record = line[0:6].strip()
            if record not in {"ATOM", "HETATM"}:
                continue

            altloc = line[16].strip()
            if altloc not in {"", "A"}:
                continue

            element = line[76:78].strip().upper()
            atom_name = line[12:16].strip()
            if not element:
                element = infer_element(atom_name)

            # Heavy atoms only
            if element == "H":
                continue

            try:
                atom = Atom(
                    record=record,
                    serial=int(line[6:11]),
                    name=atom_name,
                    altloc=altloc,
                    resn=line[17:20].strip(),
                    chain=line[21].strip(),
                    resi=int(line[22:26]),
                    icode=line[26].strip(),
                    coord=np.array(
                        [
                            float(line[30:38]),
                            float(line[38:46]),
                            float(line[46:54]),
                        ],
                        dtype=float,
                    ),
                    element=element,
                )
            except ValueError:
                continue

            atoms.append(atom)

    return atoms


def distance(a: Atom, b: Atom) -> float:
    return float(np.linalg.norm(a.coord - b.coord))


def min_distance(atom: Atom, others: Iterable[Atom]) -> tuple[float, Atom | None]:
    best_d = float("inf")
    best_atom = None

    for other in others:
        d = distance(atom, other)
        if d < best_d:
            best_d = d
            best_atom = other

    return best_d, best_atom


def count_contacts(atom: Atom, others: Iterable[Atom], cutoff: float) -> int:
    return sum(distance(atom, other) <= cutoff for other in others)


def residue_key(atom: Atom) -> tuple[str, int, str, str]:
    return atom.chain, atom.resi, atom.icode, atom.resn


def group_by_residue(atoms: Iterable[Atom]):
    grouped = defaultdict(list)
    for atom in atoms:
        grouped[residue_key(atom)].append(atom)
    return grouped


def classify_pair(a: Atom, b: Atom, d: float) -> set[str]:
    classes = set()

    if d <= POLAR_CUTOFF and a.element in POLAR_ELEMENTS and b.element in POLAR_ELEMENTS:
        classes.add("polar_candidate")

    a_pos = (a.resn, a.name) in POSITIVE_ATOMS
    b_pos = (b.resn, b.name) in POSITIVE_ATOMS
    a_neg = (a.resn, a.name) in NEGATIVE_ATOMS
    b_neg = (b.resn, b.name) in NEGATIVE_ATOMS

    if d <= SALT_CUTOFF and ((a_pos and b_neg) or (a_neg and b_pos)):
        classes.add("salt_bridge_candidate")

    if (
        d <= HYDROPHOBIC_CUTOFF
        and a.element in HYDROPHOBIC_ELEMENTS
        and b.element in HYDROPHOBIC_ELEMENTS
    ):
        classes.add("hydrophobic_candidate")

    return classes


def safe_angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 == 0 or n2 == 0:
        return float("nan")

    cosang = np.dot(v1, v2) / (n1 * n2)
    cosang = float(np.clip(cosang, -1.0, 1.0))
    return math.degrees(math.acos(cosang))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None):
    if fieldnames is None:
        if not rows:
            raise ValueError(f"No rows to write: {path}")
        fieldnames = list(rows[0].keys())

    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Analysis 1: RAP atom proximity / face assignment
# ============================================================

def analyze_rap_atoms(rap, chain1, chain2):
    rows = []

    for atom in rap:
        d1, near1 = min_distance(atom, chain1)
        d2, near2 = min_distance(atom, chain2)

        n1_4 = count_contacts(atom, chain1, CONTACT_CUTOFF)
        n2_4 = count_contacts(atom, chain2, CONTACT_CUTOFF)
        n1_5 = count_contacts(atom, chain1, PROXIMITY_CUTOFF)
        n2_5 = count_contacts(atom, chain2, PROXIMITY_CUTOFF)

        # Structural face classification
        if d1 <= CONTACT_CUTOFF and d2 <= CONTACT_CUTOFF:
            face = "shared_seam"
        elif d1 <= CONTACT_CUTOFF and d2 > PROXIMITY_CUTOFF:
            face = f"{CHAIN_1}_face"
        elif d2 <= CONTACT_CUTOFF and d1 > PROXIMITY_CUTOFF:
            face = f"{CHAIN_2}_face"
        elif min(d1, d2) <= CONTACT_CUTOFF:
            face = "edge_contact"
        elif min(d1, d2) <= PROXIMITY_CUTOFF:
            face = "proximity_only"
        else:
            face = "exposed"

        # Simple geometric burial proxy:
        # more neighboring protein heavy atoms -> more occluded atom
        n_total_5 = n1_5 + n2_5
        if n_total_5 >= 12:
            burial_proxy = "strongly_buried"
        elif n_total_5 >= 6:
            burial_proxy = "buried"
        elif n_total_5 >= 2:
            burial_proxy = "partially_buried"
        else:
            burial_proxy = "exposed"

        rows.append(
            {
                "rap_atom": atom.name,
                "element": atom.element,
                f"min_dist_{CHAIN_1}_A": round(d1, 3),
                f"nearest_{CHAIN_1}_residue": near1.residue_label if near1 else "",
                f"nearest_{CHAIN_1}_atom": near1.name if near1 else "",
                f"min_dist_{CHAIN_2}_A": round(d2, 3),
                f"nearest_{CHAIN_2}_residue": near2.residue_label if near2 else "",
                f"nearest_{CHAIN_2}_atom": near2.name if near2 else "",
                f"{CHAIN_1}_contacts_4A": n1_4,
                f"{CHAIN_2}_contacts_4A": n2_4,
                f"{CHAIN_1}_contacts_5A": n1_5,
                f"{CHAIN_2}_contacts_5A": n2_5,
                "face_class": face,
                "burial_proxy": burial_proxy,
            }
        )

    return rows


# ============================================================
# Analysis 2: Direct E-F residue interface
# ============================================================

def analyze_residue_interface(chain1, chain2):
    res1 = group_by_residue(chain1)
    res2 = group_by_residue(chain2)

    rows = []

    for key1, atoms1 in res1.items():
        for key2, atoms2 in res2.items():
            min_d = float("inf")
            closest = None
            contact_pairs_4 = 0
            polar_pairs = 0
            salt_pairs = 0
            hydrophobic_pairs = 0

            for a in atoms1:
                for b in atoms2:
                    d = distance(a, b)
                    if d < min_d:
                        min_d = d
                        closest = (a, b)

                    if d <= CONTACT_CUTOFF:
                        contact_pairs_4 += 1

                    classes = classify_pair(a, b, d)
                    polar_pairs += "polar_candidate" in classes
                    salt_pairs += "salt_bridge_candidate" in classes
                    hydrophobic_pairs += "hydrophobic_candidate" in classes

            # Keep only residues near the direct interface
            if min_d > PROXIMITY_CUTOFF:
                continue

            a0, b0 = closest
            rows.append(
                {
                    f"{CHAIN_1}_residue": a0.residue_label,
                    f"{CHAIN_2}_residue": b0.residue_label,
                    "min_dist_A": round(min_d, 3),
                    f"closest_{CHAIN_1}_atom": a0.name,
                    f"closest_{CHAIN_2}_atom": b0.name,
                    "heavy_atom_pairs_4A": contact_pairs_4,
                    "polar_candidates_3p5A": polar_pairs,
                    "salt_bridge_candidates_4A": salt_pairs,
                    "hydrophobic_pairs_4p5A": hydrophobic_pairs,
                }
            )

    rows.sort(
        key=lambda x: (
            -x["heavy_atom_pairs_4A"],
            x["min_dist_A"],
        )
    )
    return rows


# ============================================================
# Analysis 3: Tripartite residues
# ============================================================

def residue_contact_stats(res_atoms, target_atoms, cutoff):
    contacts = []
    min_d = float("inf")

    for a in res_atoms:
        for b in target_atoms:
            d = distance(a, b)
            if d < min_d:
                min_d = d
            if d <= cutoff:
                contacts.append((a, b, d))

    return min_d, contacts


def analyze_tripartite(chain_self, chain_other, rap, chain_name):
    grouped = group_by_residue(chain_self)
    rows = []

    for _, atoms in grouped.items():
        d_rap, rap_contacts = residue_contact_stats(
            atoms, rap, CONTACT_CUTOFF
        )
        d_other, other_contacts = residue_contact_stats(
            atoms, chain_other, CONTACT_CUTOFF
        )

        if not rap_contacts or not other_contacts:
            continue

        residue = atoms[0]

        rap_atom_names = sorted({b.name for _, b, _ in rap_contacts})
        opposite_residues = sorted(
            {b.residue_label for _, b, _ in other_contacts}
        )

        polar_to_rap = 0
        polar_to_other = 0
        hydrophobic_to_rap = 0
        hydrophobic_to_other = 0

        for a, b, d in rap_contacts:
            cls = classify_pair(a, b, d)
            polar_to_rap += "polar_candidate" in cls
            hydrophobic_to_rap += "hydrophobic_candidate" in cls

        for a, b, d in other_contacts:
            cls = classify_pair(a, b, d)
            polar_to_other += "polar_candidate" in cls
            hydrophobic_to_other += "hydrophobic_candidate" in cls

        # Simple hotspot priority:
        # reward simultaneous ligand + opposite-chain contact,
        # with extra weight for directional polar contacts.
        score = (
            2.0 * len(rap_contacts)
            + 2.0 * len(other_contacts)
            + 2.5 * polar_to_rap
            + 2.5 * polar_to_other
            + 0.5 * hydrophobic_to_rap
            + 0.5 * hydrophobic_to_other
        )

        rows.append(
            {
                "chain": chain_name,
                "residue": residue.residue_label,
                "min_dist_to_RAP_A": round(d_rap, 3),
                "RAP_contact_pairs_4A": len(rap_contacts),
                "RAP_atoms_contacted": ";".join(rap_atom_names),
                "min_dist_to_opposite_chain_A": round(d_other, 3),
                "opposite_contact_pairs_4A": len(other_contacts),
                "opposite_residues_contacted": ";".join(opposite_residues),
                "polar_to_RAP": polar_to_rap,
                "polar_to_opposite": polar_to_other,
                "hydrophobic_to_RAP": hydrophobic_to_rap,
                "hydrophobic_to_opposite": hydrophobic_to_other,
                "tripartite_score": round(score, 2),
            }
        )

    rows.sort(key=lambda x: -x["tripartite_score"])
    return rows


# ============================================================
# Analysis 4: Global CID geometry
# ============================================================

def contact_atoms_to_ligand(chain, rap, cutoff):
    selected = []
    for atom in chain:
        if any(distance(atom, lig) <= cutoff for lig in rap):
            selected.append(atom)
    return selected


def contact_residue_ids(chain, target, cutoff):
    residues = set()
    for atom in chain:
        if any(distance(atom, t) <= cutoff for t in target):
            residues.add(atom.residue_label)
    return residues


def summarize(rap, chain1, chain2, rap_rows, ef_rows, trip_rows):
    rap_xyz = np.vstack([a.coord for a in rap])
    rap_centroid = rap_xyz.mean(axis=0)

    contact1 = contact_atoms_to_ligand(
        chain1, rap, ANGLE_CONTACT_CUTOFF
    )
    contact2 = contact_atoms_to_ligand(
        chain2, rap, ANGLE_CONTACT_CUTOFF
    )

    if contact1 and contact2:
        c1 = np.vstack([a.coord for a in contact1]).mean(axis=0)
        c2 = np.vstack([a.coord for a in contact2]).mean(axis=0)
        angle = safe_angle_deg(c1 - rap_centroid, c2 - rap_centroid)
    else:
        angle = float("nan")

    direct1 = contact_residue_ids(
        chain1, chain2, CONTACT_CUTOFF
    )
    direct2 = contact_residue_ids(
        chain2, chain1, CONTACT_CUTOFF
    )
    rap_res1 = contact_residue_ids(
        chain1, rap, CONTACT_CUTOFF
    )
    rap_res2 = contact_residue_ids(
        chain2, rap, CONTACT_CUTOFF
    )

    face_counts = defaultdict(int)
    burial_counts = defaultdict(int)
    for row in rap_rows:
        face_counts[row["face_class"]] += 1
        burial_counts[row["burial_proxy"]] += 1

    direct_pairs = sum(
        row["heavy_atom_pairs_4A"] for row in ef_rows
    )

    trip1 = sum(row["chain"] == CHAIN_1 for row in trip_rows)
    trip2 = sum(row["chain"] == CHAIN_2 for row in trip_rows)

    summary = [
        {"metric": "RAP_heavy_atoms", "value": len(rap)},
        {"metric": f"{CHAIN_1}_RAP_contact_residues_4A", "value": len(rap_res1)},
        {"metric": f"{CHAIN_2}_RAP_contact_residues_4A", "value": len(rap_res2)},
        {"metric": f"{CHAIN_1}_direct_interface_residues_4A", "value": len(direct1)},
        {"metric": f"{CHAIN_2}_direct_interface_residues_4A", "value": len(direct2)},
        {"metric": "direct_EF_heavy_atom_pairs_4A", "value": direct_pairs},
        {"metric": f"{CHAIN_1}_tripartite_residues", "value": trip1},
        {"metric": f"{CHAIN_2}_tripartite_residues", "value": trip2},
        {"metric": "RAP_contact_face_angle_deg", "value": round(angle, 2)},
    ]

    for key in [
        f"{CHAIN_1}_face",
        f"{CHAIN_2}_face",
        "shared_seam",
        "edge_contact",
        "proximity_only",
        "exposed",
    ]:
        summary.append(
            {"metric": f"RAP_atoms_{key}", "value": face_counts[key]}
        )

    for key in [
        "strongly_buried",
        "buried",
        "partially_buried",
        "exposed",
    ]:
        summary.append(
            {
                "metric": f"RAP_atoms_burial_proxy_{key}",
                "value": burial_counts[key],
            }
        )

    return summary


# ============================================================
# Analysis 5: Hotspot candidate table
# ============================================================

def build_hotspot_candidates(trip_rows, rap_rows):
    rows = []

    # Protein residues: tripartite residues are highest-value CID motifs
    for rank, row in enumerate(
        sorted(trip_rows, key=lambda x: -x["tripartite_score"]),
        start=1,
    ):
        rows.append(
            {
                "type": "protein_tripartite_residue",
                "candidate": row["residue"],
                "priority": rank,
                "reason": (
                    f"contacts RAP ({row['RAP_atoms_contacted']}) and opposite chain "
                    f"({row['opposite_residues_contacted']}); "
                    f"score={row['tripartite_score']}"
                ),
            }
        )

    # RAP atoms: shared seam first, then chain-specific contact faces
    face_priority = {
        "shared_seam": 1,
        "edge_contact": 2,
        f"{CHAIN_1}_face": 3,
        f"{CHAIN_2}_face": 3,
        "proximity_only": 4,
        "exposed": 5,
    }

    rap_sorted = sorted(
        rap_rows,
        key=lambda x: (
            face_priority.get(x["face_class"], 99),
            -(x[f"{CHAIN_1}_contacts_5A"] + x[f"{CHAIN_2}_contacts_5A"]),
        ),
    )

    for rank, row in enumerate(rap_sorted, start=1):
        rows.append(
            {
                "type": "RAP_atom",
                "candidate": row["rap_atom"],
                "priority": rank,
                "reason": (
                    f"{row['face_class']}; {row['burial_proxy']}; "
                    f"{CHAIN_1}/F 4A contacts="
                    f"{row[f'{CHAIN_1}_contacts_4A']}/"
                    f"{row[f'{CHAIN_2}_contacts_4A']}"
                ),
            }
        )

    return rows


def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_PDB.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_PDB}")

    atoms = parse_pdb(INPUT_PDB)

    chain1 = [
        a for a in atoms
        if a.record == "ATOM" and a.chain == CHAIN_1
    ]
    chain2 = [
        a for a in atoms
        if a.record == "ATOM" and a.chain == CHAIN_2
    ]
    rap = [
        a for a in atoms
        if a.record == "HETATM" and a.resn == LIGAND_RESN
    ]

    if not chain1:
        raise RuntimeError(f"No protein atoms found for chain {CHAIN_1}")
    if not chain2:
        raise RuntimeError(f"No protein atoms found for chain {CHAIN_2}")
    if not rap:
        raise RuntimeError(f"No {LIGAND_RESN} atoms found")

    print(f"Input: {INPUT_PDB}")
    print(f"{CHAIN_1}: {len(chain1)} heavy atoms")
    print(f"{CHAIN_2}: {len(chain2)} heavy atoms")
    print(f"{LIGAND_RESN}: {len(rap)} heavy atoms")

    rap_rows = analyze_rap_atoms(rap, chain1, chain2)
    ef_rows = analyze_residue_interface(chain1, chain2)

    trip_rows = (
        analyze_tripartite(chain1, chain2, rap, CHAIN_1)
        + analyze_tripartite(chain2, chain1, rap, CHAIN_2)
    )
    trip_rows.sort(key=lambda x: -x["tripartite_score"])

    summary_rows = summarize(
        rap, chain1, chain2, rap_rows, ef_rows, trip_rows
    )

    hotspot_rows = build_hotspot_candidates(trip_rows, rap_rows)

    write_csv(
        ANALYSIS_DIR / "rap_atom_proximity.csv",
        rap_rows,
    )
    write_csv(
        ANALYSIS_DIR / "ef_residue_contacts.csv",
        ef_rows,
    )
    write_csv(
        ANALYSIS_DIR / "tripartite_residues.csv",
        trip_rows,
    )
    write_csv(
        ANALYSIS_DIR / "cid_summary.csv",
        summary_rows,
    )
    write_csv(
        ANALYSIS_DIR / "cid_hotspot_candidates.csv",
        hotspot_rows,
    )

    print("\nWritten:")
    for name in [
        "cid_summary.csv",
        "rap_atom_proximity.csv",
        "ef_residue_contacts.csv",
        "tripartite_residues.csv",
        "cid_hotspot_candidates.csv",
    ]:
        print(f"  {ANALYSIS_DIR / name}")

    print("\nKey summary:")
    for row in summary_rows:
        print(f"  {row['metric']}: {row['value']}")


if __name__ == "__main__":
    main()

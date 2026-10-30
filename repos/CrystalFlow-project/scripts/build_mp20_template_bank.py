"""Build a composition -> template bank from MP-20 train.csv.

Template key: (num_atoms, sorted atomic numbers). No spacegroup.
For each key, keep the structure with lowest e_above_hull.
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from pymatgen.io.cif import CifParser

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "mp_20" / "train.csv"
OUT = ROOT / "data" / "mp_20" / "template_bank.pt"


def parse(cif):
    try:
        return CifParser.from_str(cif).parse_structures(primitive=False)[0]
    except Exception:
        return None


def key_of(Z):
    return (len(Z), tuple(sorted(int(z) for z in Z)))


def main():
    df = pd.read_csv(TRAIN)
    bank = {}
    for _, row in df.iterrows():
        s = parse(row["cif"])
        if s is None:
            continue
        order = sorted(range(len(s)), key=lambda i: (s[i].specie.Z, s[i].frac_coords[0], s[i].frac_coords[1]))
        Z = np.array([s[i].specie.Z for i in order], dtype=np.int64)
        X = np.array([s[i].frac_coords for i in order], dtype=np.float32) % 1.0
        L = np.array(s.lattice.matrix, dtype=np.float32)
        e = float(row["e_above_hull"]) if pd.notna(row["e_above_hull"]) else np.inf
        k = key_of(Z)
        if k not in bank or e < bank[k]["e_hull"]:
            bank[k] = {"L": torch.from_numpy(L), "X": torch.from_numpy(X),
                       "Z": torch.from_numpy(Z), "e_hull": e,
                       "formula": row["pretty_formula"]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, OUT)
    print("saved", OUT, "groups", len(bank))


if __name__ == "__main__":
    main()

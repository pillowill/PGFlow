"""Build a global prototype bank for MPTS-52 (no spacegroup)."""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from pymatgen.io.cif import CifParser
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "mp_20" / "train.csv"
OUT = ROOT / "data" / "mp_20" / "prototype_bank.pt"
K = 32
NBINS = 16
RMAX = 8.0


def lattice_polar_np(L):
    J = L @ L.T
    A, U = np.linalg.eigh(J)
    S = (U * (0.5 * np.log(A))) @ U.T
    return np.array([S[0,1], S[0,2], S[1,2],
                     (S[0,0]-S[1,1])/2,
                     (S[0,0]+S[1,1]-2*S[2,2])/6,
                     (S[0,0]+S[1,1]+S[2,2])/3], dtype=np.float32)


def pdd_hist(L, X):
    d = X[:, None, :] - X[None, :, :]
    d = d - np.round(d)
    cart = d @ L
    dist = np.linalg.norm(cart, axis=-1)
    iu = np.triu_indices(len(X), 1)
    vals = dist[iu]
    h, _ = np.histogram(vals, bins=NBINS, range=(0.0, RMAX), density=False)
    h = h.astype(np.float32)
    return h / max(h.sum(), 1.0)


def parse(cif):
    try:
        return CifParser.from_str(cif).parse_structures(primitive=False)[0]
    except Exception:
        return None


def main():
    df = pd.read_csv(CSV)
    feats, lats, coords, types, keys, e_hull = [], [], [], [], [], []
    for n, row in enumerate(df.itertuples(index=False)):
        s = parse(row.cif)
        if s is None:
            continue
        order = sorted(range(len(s)), key=lambda i: (s[i].specie.Z, s[i].frac_coords[0], s[i].frac_coords[1]))
        Z = np.array([s[i].specie.Z for i in order], dtype=np.int64)
        X = np.array([s[i].frac_coords for i in order], dtype=np.float32) % 1.0
        L = np.array(s.lattice.matrix, dtype=np.float32)
        k = lattice_polar_np(L)
        pdd = pdd_hist(L, X)
        vol = abs(np.linalg.det(L)) / len(X)
        f = np.r_[k, pdd, np.log(len(X)), np.log(vol + 1e-6)].astype(np.float32)
        feats.append(f); lats.append(k); coords.append(X); types.append(Z)
        keys.append((len(Z), tuple(sorted(Z.tolist()))))
        e_hull.append(float(row.e_above_hull) if np.isfinite(row.e_above_hull) else np.inf)
        if n % 2000 == 0:
            print("parsed", n, flush=True)

    F = np.stack(feats)
    scaler = StandardScaler().fit(F)
    Fs = scaler.transform(F)
    pca = PCA(n_components=min(32, Fs.shape[1]), random_state=0).fit(Fs)
    Zp = pca.transform(Fs)
    km = MiniBatchKMeans(n_clusters=K, random_state=0, batch_size=1024, n_init=10).fit(Zp)
    labels = km.labels_
    print("clustered", len(labels), "K", K, flush=True)

    lattice_mean, lattice_std, medoid_X, medoid_Z = [], [], [], []
    for c in range(K):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            lattice_mean.append(torch.zeros(6)); lattice_std.append(torch.ones(6))
            medoid_X.append(torch.zeros(1, 3)); medoid_Z.append(torch.zeros(1, dtype=torch.long))
            continue
        center = Zp[idx].mean(0)
        med = idx[np.argmin(((Zp[idx] - center) ** 2).sum(1))]
        lattice_mean.append(torch.from_numpy(np.stack([lats[i] for i in idx]).mean(0)))
        lattice_std.append(torch.from_numpy(np.stack([lats[i] for i in idx]).std(0) + 1e-3))
        medoid_X.append(torch.from_numpy(coords[med]))
        medoid_Z.append(torch.from_numpy(types[med]))

    key_entries = {}
    for key, lab, e, L, X, Z in zip(keys, labels, e_hull, lats, coords, types):
        key_entries.setdefault(key, []).append((e, L, X, Z))
    key_best = {k: sorted(v, key=lambda t: t[0])[0] for k, v in key_entries.items()}
    key_alt = {k: sorted(v, key=lambda t: t[0])[1] for k, v in key_entries.items() if len(v) > 1}

    key_counts = {}
    for key, lab in zip(keys, labels):
        if key not in key_counts:
            key_counts[key] = np.zeros(K, dtype=np.float32)
        key_counts[key][lab] += 1
    pi = np.bincount(labels, minlength=K).astype(np.float32)
    pi /= pi.sum()

    bank = {
        "K": K,
        "lattice_mean": torch.stack(lattice_mean),
        "lattice_std": torch.stack(lattice_std),
        "medoid_X": medoid_X,
        "medoid_Z": medoid_Z,
        "pi": torch.from_numpy(pi),
        "key_counts": key_counts,
        "lattice_feat_mean": torch.from_numpy(np.stack(lats).mean(0)),
        "lattice_feat_std": torch.from_numpy(np.stack(lats).std(0) + 1e-6),
        "key_lattice": {k: torch.from_numpy(v[1]) for k, v in key_best.items()},
        "key_X": {k: torch.from_numpy(v[2]) for k, v in key_best.items()},
        "key_Z": {k: torch.from_numpy(v[3]) for k, v in key_best.items()},
        "key_alt_lattice": {k: torch.from_numpy(v[1]) for k, v in key_alt.items()},
        "key_alt_X": {k: torch.from_numpy(v[2]) for k, v in key_alt.items()},
        "key_alt_Z": {k: torch.from_numpy(v[3]) for k, v in key_alt.items()},
        "labels": labels,
        "keys": keys,
        "pca": pca,
        "scaler": scaler,
    }
    torch.save(bank, OUT)
    print("saved", OUT, "K", K, "keys", len(key_counts), flush=True)


if __name__ == "__main__":
    main()

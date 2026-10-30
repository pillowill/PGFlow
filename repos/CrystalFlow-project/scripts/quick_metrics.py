"""Fast subset metrics when generated lattices are extreme.

Usage:
  python quick_metrics.py --root_path RUN --label LABEL --n 512
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from pymatgen.analysis.structure_matcher import StructureMatcher

from eval_utils import get_crystals_list
from compute_metrics import Crystal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root_path', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--n', type=int, default=512)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    root = Path(args.root_path)
    d = torch.load(root / f'eval_diff_{args.label}.pt', map_location='cpu')
    fc = d['frac_coords'][0]
    at = d['atom_types'][0]
    ln = d['lengths'][0]
    an = d['angles'][0]
    na = d['num_atoms'][0]
    pred_arrays = get_crystals_list(fc, at, ln, an, na)

    b = d['input_data_batch']
    gt_arrays = get_crystals_list(b.frac_coords, b.atom_types, b.lengths, b.angles, b.num_atoms)

    n = min(args.n, len(pred_arrays), len(gt_arrays))
    sm = StructureMatcher(ltol=0.3, stol=0.3, angle_tol=10)
    n_valid = n_constructed = n_matched = 0
    rms = []
    lat_mae = []
    invalid_reasons = {}
    for i in range(n):
        a = pred_arrays[i]
        L = np.asarray(a['lengths']); A = np.asarray(a['angles'])
        reason = None
        if not np.isfinite(L).all() or not np.isfinite(A).all():
            reason = 'nan'
        elif L.min() < 0.5 or L.max() > 50.0:
            reason = 'extreme_lattice'
        elif A.min() < 10 or A.max() > 170:
            reason = 'extreme_angle'
        if reason:
            invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
            continue
        n_constructed += 1
        p = Crystal(a, compute_valid=True, compute_fp=False)
        if not p.valid:
            invalid_reasons['invalid'] = invalid_reasons.get('invalid', 0) + 1
            continue
        n_valid += 1
        g = Crystal(gt_arrays[i], compute_valid=True, compute_fp=False)
        try:
            r = sm.get_rms_dist(p.structure, g.structure)
        except Exception:
            r = None
        if r is not None:
            n_matched += 1
            rms.append(float(r[0]))
        try:
            lat_mae.append(float(np.mean(np.abs(L - np.asarray(gt_arrays[i]['lengths'])))))
        except Exception:
            pass

    out = {
        'n': n,
        'constructed_rate': n_constructed / n,
        'valid_rate': n_valid / n,
        'match_rate': n_matched / n,
        'match_rate_valid': n_matched / max(n_valid, 1),
        'rms_dist_matched': float(np.mean(rms)) if rms else None,
        'lattice_mae_valid': float(np.mean(lat_mae)) if lat_mae else None,
        'invalid_reasons': invalid_reasons,
        'note': 'subset; extreme lattice filtered before StructureMatcher',
    }
    out_path = Path(args.out) if args.out else root / f'quick_metrics_{args.label}.json'
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()


"""Template source for CrystalFlow CSP (composition-only at test).

This module is intentionally self-contained: it does not need spacegroup
labels, and it returns the same tensors as CSPFlow's built-in source.

Source:
  L0 = template lattice (polar or matrix) + sigma_l * noise
  X0 = template frac coords reordered to target atom order + sigma_x * noise

Template lookup key:
  (num_atoms, sorted atomic numbers)

If a key is missing, fall back to the original Gaussian/uniform source.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def lattice_polar(L: torch.Tensor) -> torch.Tensor:
    """3x3 lattice matrix -> 6D polar representation used by CrystalFlow."""
    J = L @ L.T
    A, U = torch.linalg.eigh(J)
    S = (U * (0.5 * A.log())) @ U.T
    k0 = S[0, 1]
    k1 = S[0, 2]
    k2 = S[1, 2]
    k3 = (S[0, 0] - S[1, 1]) / 2
    k4 = (S[0, 0] + S[1, 1] - 2 * S[2, 2]) / 6
    k5 = (S[0, 0] + S[1, 1] + S[2, 2]) / 3
    return torch.stack([k0, k1, k2, k3, k4, k5])


def reorder_by_type(Xt: torch.Tensor, Zt: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    """Reorder template coords to target atom order, by element only.

    Uses target *types*, never target coordinates.
    """
    out = torch.empty_like(Xt)
    for z in torch.unique(Z):
        ti = torch.nonzero(Zt == z).flatten()
        gi = torch.nonzero(Z == z).flatten()
        if len(ti) != len(gi):
            raise ValueError(f"type-count mismatch for Z={z}: {len(ti)} vs {len(gi)}")
        out[gi] = Xt[ti]
    return out


def key_of(Z) -> tuple:
    return (len(Z), tuple(sorted(int(z) for z in Z)))


class TemplateSource(nn.Module):
    def __init__(
        self,
        bank=None,
        bank_path: Optional[str] = None,
        lattice_polar_flag: bool = True,
        sigma_l: float = 0.10,
        sigma_x: float = 0.02,
        coord_mode: str = "template",
        fallback: str = "gaussian",
        **kwargs,
    ):
        super().__init__()
        if bank is None:
            if bank_path is None:
                raise ValueError("provide bank or bank_path")
            bank = torch.load(bank_path, map_location="cpu")
        self.bank = bank
        self.lattice_polar_flag = lattice_polar_flag
        self.sigma_l = sigma_l
        self.sigma_x = sigma_x
        self.coord_mode = coord_mode
        self.fallback = fallback

    def sample(self, batch):
        B = batch.num_graphs
        N = batch.frac_coords.shape[0]
        device = batch.frac_coords.device

        if self.lattice_polar_flag:
            l0 = torch.empty(B, 6, device=device)
        else:
            l0 = torch.empty(B, 3, 3, device=device)
        f0 = torch.empty(N, 3, device=device)
        node2graph = getattr(batch, "batch", None)
        if node2graph is None:
            node2graph = torch.repeat_interleave(torch.arange(B), batch.num_atoms.long())

        for i in range(B):
            nodes = torch.nonzero(node2graph == i).flatten()
            Z = batch.atom_types[nodes].cpu()
            k = key_of(Z)
            tmpl = self.bank.get(k, None)

            if tmpl is None:
                if self.lattice_polar_flag:
                    l0[i] = torch.randn(6, device=device)
                    l0[i, 5] += 1.0
                else:
                    l0[i] = torch.randn(3, 3, device=device)
                f0[nodes] = torch.rand(len(nodes), 3, device=device)
                continue

            L = tmpl["L"].to(device)
            Xt = tmpl["X"].to(device)
            Zt = tmpl["Z"].to(device)

            kk = lattice_polar(L) if self.lattice_polar_flag else L
            l0[i] = kk + self.sigma_l * torch.randn_like(kk)

            if self.coord_mode == "template":
                Xt = reorder_by_type(Xt, Zt, Z.to(device))
                f0[nodes] = (Xt + self.sigma_x * torch.randn_like(Xt)) % 1.0
            else:
                f0[nodes] = torch.rand(len(nodes), 3, device=device)

        return l0, f0

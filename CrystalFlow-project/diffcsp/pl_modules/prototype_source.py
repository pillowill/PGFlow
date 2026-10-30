
"""Prototype-mixture source for CrystalFlow CSP.

Source:
  p0(L, X | c) = sum_k q(k | c) p_{0,k}(L, X | c)

- k is a prototype from global structure clustering (not spacegroup).
- q(k|c) is an empirical composition -> prototype distribution with smoothing.
- p_{0,k}: Gaussian in lattice_polar + medoid coordinates with noise.
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import torch
import torch.nn as nn


def reorder_by_type(Xt, Zt, Z):
    out = torch.empty_like(Xt)
    for z in torch.unique(Z):
        ti = torch.nonzero(Zt == z).flatten()
        gi = torch.nonzero(Z == z).flatten()
        if len(ti) != len(gi):
            raise ValueError(f"type count mismatch for Z={z}: {len(ti)} vs {len(gi)}")
        out[gi] = Xt[ti]
    return out


class PrototypeSource(nn.Module):
    def __init__(
        self,
        bank_path: str,
        lattice_polar_flag: bool = True,
        sigma_l: float = 0.05,
        sigma_x: float = 0.02,
        q_smoothing: float = 1.0,
        fallback: str = "gaussian",
        **kwargs,
    ):
        super().__init__()
        bank = torch.load(bank_path, map_location="cpu")
        self.lattice_mean = bank["lattice_mean"].float()     # (K,6)
        self.lattice_std = bank["lattice_std"].float()       # (K,6)
        self.medoid_X = bank["medoid_X"]                     # list of (N,3)
        self.medoid_Z = bank["medoid_Z"]                     # list of (N,)
        self.pi = bank["pi"].float()                         # (K,)
        self.key_counts = bank.get("key_counts", {})         # key -> (K,)
        self.key_lattice = bank.get("key_lattice", {})
        self.key_X = bank.get("key_X", {})
        self.key_Z = bank.get("key_Z", {})
        self.key_alt_lattice = bank.get("key_alt_lattice", {})
        self.key_alt_X = bank.get("key_alt_X", {})
        self.key_alt_Z = bank.get("key_alt_Z", {})
        self.feat_mean = bank.get("lattice_feat_mean", torch.zeros(6)).float()
        self.feat_std = bank.get("lattice_feat_std", torch.ones(6)).float()
        self.lattice_polar_flag = lattice_polar_flag
        self.sigma_l = sigma_l
        self.sigma_x = sigma_x
        self.q_smoothing = q_smoothing
        self.fallback = fallback

    def _key(self, Z):
        return (len(Z), tuple(sorted(int(z) for z in Z)))

    def _q(self, key):
        c = self.key_counts.get(key, None)
        if c is None:
            return self.pi.numpy()
        c = np.asarray(c, dtype=np.float64)
        q = (c + self.q_smoothing * self.pi.numpy()) / (c.sum() + self.q_smoothing)
        return q / q.sum()

    def _norm_lattice(self, raw):
        # bank stores raw lattice_polar; source lives in normalized latent
        return (raw - self.feat_mean.to(raw.device)) / self.feat_std.to(raw.device)

    def sample(self, batch):
        # Keep the per-graph Python work on CPU and transfer once at the end.
        # This avoids dozens of GPU->CPU syncs and per-graph H2D copies.
        B = batch.num_graphs
        N = batch.frac_coords.shape[0]
        device = batch.frac_coords.device

        node2graph = getattr(batch, "batch", None)
        if node2graph is None:
            node2graph = torch.repeat_interleave(torch.arange(B), batch.num_atoms.long())
        node2graph = node2graph.detach().cpu()
        atom_types = batch.atom_types.detach().cpu()
        lattice_mean = self.lattice_mean.float().cpu()

        l0 = torch.empty(B, 6, dtype=torch.float32)
        f0 = torch.empty(N, 3, dtype=torch.float32)

        for i in range(B):
            nodes = torch.nonzero(node2graph == i, as_tuple=False).flatten()
            Z = atom_types[nodes]
            key = self._key(Z)
            q = self._q(key)
            k = int(np.random.choice(len(q), p=q))

            if self.training:
                Xdict, Zdict, Ldict = self.key_alt_X, self.key_alt_Z, self.key_alt_lattice
            else:
                Xdict, Zdict, Ldict = self.key_X, self.key_Z, self.key_lattice

            if key not in Xdict:
                l0[i] = torch.randn(6)
                l0[i, 5] = 1.0
                l0[i] = torch.clamp(l0[i], -4.0, 4.0)
                f0[nodes] = torch.rand(len(nodes), 3)
                continue

            Xt = Xdict[key].float().cpu()
            Zt = Zdict[key].cpu()
            l_raw = 0.5 * (lattice_mean[k] + Ldict[key].float().cpu())
            l_base = self._norm_lattice(l_raw)
            l0[i] = l_base + self.sigma_l * torch.randn(6)
            l0[i, 5] = 1.0
            l0[i] = torch.clamp(l0[i], -4.0, 4.0)
            Xt = reorder_by_type(Xt, Zt, Z)
            f0[nodes] = (Xt + self.sigma_x * torch.randn_like(Xt)) % 1.0

        return l0.to(device), f0.to(device)

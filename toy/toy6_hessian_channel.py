"""Toy-6: Hessian-metric transport through a cooperative migration channel.

The state (x, y) represents a migration coordinate x and a transverse
neighbour-relaxation coordinate y in a fixed periodic-crystal local frame.
Straight paths cross an off-centre, high-curvature exclusion core.  The
Hessian path minimizes an action with G = I + alpha H(E)^T H(E), where H(E)
is known analytically.  It is a mechanism test, not a material PES.
"""
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
K = 16
ALPHA = 0.18
HESSIAN_SCALE = 100.0
OBSTACLE = torch.tensor([0.0, -0.35], device=DEVICE, dtype=DTYPE)
SX, SY, AMP, KY = 0.42, 0.28, 9.0, 0.35


def energy_hessian(q):
    """Energy and Cartesian Hessian for a harmonic channel plus exclusion core."""
    dx, dy = q[..., 0], q[..., 1] - OBSTACLE[1]
    exponent = -0.5 * ((dx / SX) ** 2 + (dy / SY) ** 2)
    core = AMP * torch.exp(exponent)
    energy = 0.5 * KY * q[..., 1] ** 2 + core
    ux, uy = dx / SX**2, dy / SY**2
    hxx = core * (ux * ux - 1.0 / SX**2)
    hxy = core * (ux * uy)
    hyy = KY + core * (uy * uy - 1.0 / SY**2)
    h = torch.stack((torch.stack((hxx, hxy), -1), torch.stack((hxy, hyy), -1)), -2)
    return energy, h


def metric_action(nodes, hessian_metric):
    d = nodes[:, 1:] - nodes[:, :-1]
    mid = 0.5 * (nodes[:, 1:] + nodes[:, :-1])
    if hessian_metric:
        _, h = energy_hessian(mid)
        g = torch.eye(2, device=DEVICE, dtype=DTYPE) + ALPHA * (h.transpose(-1, -2) @ h) / HESSIAN_SCALE
        segment = torch.einsum("bki,bkij,bkj->bk", d, g, d)
    else:
        segment = (d * d).sum(-1)
    return 0.5 * K * segment.sum(-1)


def optimize_hessian_paths(start, target, iterations=240):
    """Batched multi-start path optimisation with fixed endpoints."""
    batch = len(start)
    t = torch.linspace(0, 1, K + 1, device=DEVICE, dtype=DTYPE)[None, :, None]
    line = start[:, None] * (1 - t) + target[:, None] * t
    candidates = []
    for sign in (1.0, -1.0):
        init = line.clone()
        init[..., 1] += sign * 0.85 * torch.sin(math.pi * t[..., 0])
        candidates.append(init)
    internal = torch.nn.Parameter(torch.cat([x[:, None, 1:-1] for x in candidates], 1))
    optim = torch.optim.Adam([internal], lr=0.045)
    for _ in range(iterations):
        nodes = torch.cat((start[:, None, None].expand(-1, 2, -1, -1), internal, target[:, None, None].expand(-1, 2, -1, -1)), dim=2)
        loss = metric_action(nodes.flatten(0, 1), True).mean()
        optim.zero_grad(); loss.backward(); optim.step()
    optimized = torch.cat((start[:, None, None].expand(-1, 2, -1, -1), internal.detach(), target[:, None, None].expand(-1, 2, -1, -1)), dim=2)
    optimized = optimized.flatten(0, 1)
    candidate_actions = metric_action(optimized, True).reshape(batch, 2)
    line_action = metric_action(line, True)
    all_actions = torch.cat((line_action[:, None], candidate_actions), dim=1)
    all_nodes = torch.cat((line[:, None], optimized.reshape(batch, 2, K + 1, 2)), dim=1)
    choice = all_actions.argmin(1)
    return all_nodes[torch.arange(batch, device=DEVICE), choice], line


def interpolate(nodes, t):
    index = torch.clamp((t * K).long(), max=K - 1)
    frac = (t * K - index).unsqueeze(-1)
    row = torch.arange(len(t), device=DEVICE)
    a, b = nodes[row, index], nodes[row, index + 1]
    return a + frac * (b - a), K * (b - a)


def path_metrics(nodes):
    dense_t = torch.linspace(0, 1, 129, device=DEVICE)
    expanded = nodes[:, None].expand(-1, len(dense_t), -1, -1).reshape(-1, K + 1, 2)
    t = dense_t.repeat(len(nodes))
    positions, _ = interpolate(expanded, t)
    positions = positions.reshape(len(nodes), -1, 2)
    energies, _ = energy_hessian(positions)
    distance = torch.sqrt(((positions[..., 0] / SX) ** 2 + ((positions[..., 1] - OBSTACLE[1]) / SY) ** 2))
    d = nodes[:, 1:] - nodes[:, :-1]
    curvature = nodes[:, 2:] - 2 * nodes[:, 1:-1] + nodes[:, :-2]
    return {
        "peak_energy": float(energies.max(1).values.mean().cpu()),
        "integrated_energy": float(energies.mean(1).mean().cpu()),
        "core_entry_fraction": float((distance.min(1).values < 1.0).float().mean().cpu()),
        "path_length": float(torch.linalg.vector_norm(d, dim=-1).sum(-1).mean().cpu()),
        "discrete_curvature": float((curvature.square().sum(-1)).mean().cpu()),
    }


class VelocityMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(3, 32), torch.nn.SiLU(), torch.nn.Linear(32, 32), torch.nn.SiLU(), torch.nn.Linear(32, 2))

    def forward(self, x, t):
        return self.net(torch.cat((x, t[:, None]), dim=1))


def fit_velocity(train_nodes, val_nodes, seed):
    """Identical MLP initialization, time samples and update count per path method."""
    torch.manual_seed(seed)
    model = VelocityMLP().to(DEVICE)
    rng = torch.Generator(device=DEVICE).manual_seed(1000 + seed)
    repeats = 8
    train_nodes = train_nodes.repeat_interleave(repeats, 0)
    val_nodes = val_nodes.repeat_interleave(repeats, 0)
    train_t = torch.rand(len(train_nodes), device=DEVICE, generator=rng)
    val_t = torch.rand(len(val_nodes), device=DEVICE, generator=rng)
    x, u = interpolate(train_nodes, train_t)
    xv, uv = interpolate(val_nodes, val_t)
    optim = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(600):
        prediction = model(x, train_t)
        loss = torch.mean((prediction - u) ** 2)
        optim.zero_grad(); loss.backward(); optim.step()
    with torch.no_grad():
        prediction = model(xv, val_t)
        mse = torch.mean((prediction - uv) ** 2)
        normalized = mse / torch.mean(uv ** 2)
    return model, float(mse.cpu()), float(normalized.cpu())


def sample_source(rng, n, target=False):
    center = np.array([1.75, 0.0]) if target else np.array([-1.75, 0.0])
    noise = rng.normal(size=(n, 2)) * np.array([0.16, 0.14])
    return torch.tensor(center + noise, device=DEVICE, dtype=DTYPE)


def mmd(a, b):
    dist_aa = torch.cdist(a, a).square(); dist_bb = torch.cdist(b, b).square(); dist_ab = torch.cdist(a, b).square()
    return float((torch.exp(-dist_aa / 0.5).mean() + torch.exp(-dist_bb / 0.5).mean() - 2 * torch.exp(-dist_ab / 0.5).mean()).cpu())


def generate(model, source, target, steps):
    x = source.clone(); energies = []; core_hits = []
    with torch.no_grad():
        for i in range(steps):
            t = torch.full((len(x),), (i + 0.5) / steps, device=DEVICE)
            x += model(x, t) / steps
            e, _ = energy_hessian(x)
            radial = torch.sqrt((x[:, 0] / SX) ** 2 + ((x[:, 1] - OBSTACLE[1]) / SY) ** 2)
            energies.append(e); core_hits.append(radial < 1.0)
    energy_stack, hit_stack = torch.stack(energies), torch.stack(core_hits)
    target_np, generated_np = target[:400].cpu().numpy(), x[:400].cpu().numpy()
    cost = cdist(generated_np, target_np, "sqeuclidean")
    row, col = linear_sum_assignment(cost)
    return x, {
        "mmd_biased": mmd(x, target),
        "empirical_w2": float(np.sqrt(cost[row, col].mean())),
        "trajectory_peak_energy": float(energy_stack.max(0).values.mean().cpu()),
        "trajectory_integrated_energy": float(energy_stack.mean().cpu()),
        "trajectory_core_entry_fraction": float(hit_stack.any(0).float().mean().cpu()),
        "endpoint_energy": float(energy_stack[-1].mean().cpu()),
    }


def figure(seed, straight, hessian, model_straight, model_hessian, source):
    xx, yy = np.meshgrid(np.linspace(-2.3, 2.3, 220), np.linspace(-1.3, 1.5, 180))
    grid = torch.tensor(np.stack((xx.ravel(), yy.ravel()), -1), device=DEVICE, dtype=DTYPE)
    energy, _ = energy_hessian(grid)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for ax, paths, name, model in ((axes[0], straight, "Straight path", model_straight), (axes[1], hessian, "Hessian-metric path", model_hessian)):
        ax.contourf(xx, yy, energy.detach().cpu().numpy().reshape(xx.shape), levels=32, cmap="magma_r")
        for path in paths[:30].detach().cpu().numpy(): ax.plot(path[:, 0], path[:, 1], color="cyan", alpha=.32, lw=.8)
        generated, _ = generate(model, source[:30], source[:30] + torch.tensor([3.5, 0.0], device=DEVICE), 20)
        ax.scatter(generated[:, 0].cpu(), generated[:, 1].cpu(), c="white", s=10, label="20-NFE endpoints")
        ax.set(title=name, xlabel="migration coordinate x", ylabel="cooperative relaxation y", xlim=(-2.3, 2.3), ylim=(-1.3, 1.5))
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Toy-6: straight transport crosses the exclusion core; Hessian paths seek the soft channel")
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(f"results/figures/toy6_hessian_channel_seed{seed}.png", dpi=170); plt.close(fig)


def run(seed):
    torch.manual_seed(10 + seed); rng = np.random.default_rng(seed)
    train_start, train_target = sample_source(rng, 128), sample_source(rng, 128, target=True)
    val_start, val_target = sample_source(rng, 64), sample_source(rng, 64, target=True)
    start = time.perf_counter()
    h_train, line_train = optimize_hessian_paths(train_start, train_target)
    h_val, line_val = optimize_hessian_paths(val_start, val_target)
    path_seconds = time.perf_counter() - start
    models, rows = {}, []
    for name, train, val in (("straight", line_train, line_val), ("hessian", h_train, h_val)):
        model, mse, normalized = fit_velocity(train, val, seed=200 + seed)
        models[name] = model
        eval_source, eval_target = sample_source(rng, 800), sample_source(rng, 800, target=True)
        sampling = {}
        for steps in (5, 20, 100):
            _, sampling[str(steps)] = generate(model, eval_source, eval_target, steps)
        rows.append({"method": name, "validation_mse": mse, "normalized_mse": normalized,
                     "path_metrics": path_metrics(val), "sampling": sampling})
    figure(seed, line_val, h_val, models["straight"], models["hessian"], sample_source(rng, 30))
    return {"seed": seed, "config": {"state": "2D migration plus cooperative relaxation", "segments": K, "alpha": ALPHA, "hessian_scale": HESSIAN_SCALE, "energy": "harmonic transverse channel plus off-centre Gaussian exclusion core", "train_pairs": 128, "validation_pairs": 64, "mlp": "3-32-32-2 SiLU, 600 full-batch updates", "source_target": "independent narrow left/right basin samples", "device": str(DEVICE)}, "path_precompute_seconds": path_seconds, "results": rows}


if __name__ == "__main__":
    output = [run(seed) for seed in range(3)]
    Path("results").mkdir(exist_ok=True)
    Path("results/toy6_hessian_channel.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))

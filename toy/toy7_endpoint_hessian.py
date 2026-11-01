"""Toy-7: CurvFM-style endpoint physics on the cooperative-migration toy.

All arms use the same straight conditional paths.  ``force`` and ``hessian``
add a frozen analytic target-well surrogate at the one-step endpoint estimate.
The figure follows the split-flow grammar of the QUBO coupling study, with the
target-well energy background added on the right-hand endpoint side.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
TARGET = torch.tensor([1.75, 0.0], device=DEVICE, dtype=DTYPE)
WELL_WIDTH = torch.tensor([0.42, 0.25], device=DEVICE, dtype=DTYPE)
WELL_DEPTH = 3.0
HESSIAN_SCALE, FORCE_SCALE = 50.0, 10.0
LAMBDA_FORCE, LAMBDA_HESSIAN, TIME_POWER = 0.8, 1.5, 2
SRC, TGT, GEN, TRAJ, OFFSET = "#2f6fae", "#9ca3af", "#f28e2b", "#8b83cf", 3.4


def target_pes(q):
    """Frozen anisotropic Gaussian well: energy, force, and Hessian."""
    delta = q - TARGET
    inv_var = 1.0 / WELL_WIDTH.square()
    z = 0.5 * (delta.square() * inv_var).sum(-1)
    exp = torch.exp(-z)
    energy = WELL_DEPTH * (1.0 - exp)
    grad = WELL_DEPTH * exp[..., None] * delta * inv_var
    force = -grad
    outer = (delta * inv_var)[..., :, None] * (delta * inv_var)[..., None, :]
    diagonal = torch.diag(inv_var)
    hessian = WELL_DEPTH * exp[..., None, None] * (diagonal - outer)
    return energy, force, hessian


def sample_pairs(rng, n):
    source = np.array([-1.75, 0.0]) + rng.normal(size=(n, 2)) * np.array([0.26, 0.22])
    # Reference endpoints remain inside the local basin where curvature is informative.
    target = np.array([1.75, 0.0]) + rng.normal(size=(n, 2)) * np.array([0.10, 0.06])
    return (torch.tensor(source, device=DEVICE, dtype=DTYPE),
            torch.tensor(target, device=DEVICE, dtype=DTYPE))


class VelocityMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, 32), torch.nn.SiLU(),
            torch.nn.Linear(32, 32), torch.nn.SiLU(), torch.nn.Linear(32, 2),
        )

    def forward(self, x, t):
        return self.net(torch.cat((x, t[:, None]), dim=1))


def fit(source, target, val_source, val_target, method, seed):
    """Same model/time draws for every arm; only endpoint residual changes."""
    torch.manual_seed(seed)
    model = VelocityMLP().to(DEVICE)
    generator = torch.Generator(device=DEVICE).manual_seed(7000 + seed)
    repeats = 8
    x0 = source.repeat_interleave(repeats, 0); x1 = target.repeat_interleave(repeats, 0)
    t = torch.rand(len(x0), device=DEVICE, generator=generator)
    xt = (1 - t[:, None]) * x0 + t[:, None] * x1
    velocity = x1 - x0
    with torch.no_grad():
        _, target_force, target_hessian = target_pes(x1)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(350):
        pred = model(xt, t)
        fm = (pred - velocity).square().mean()
        endpoint = xt + (1 - t[:, None]) * pred
        weight = t.pow(TIME_POWER)
        if method == "force":
            _, predicted_force, _ = target_pes(endpoint)
            residual = ((predicted_force - target_force).square().mean(1) / FORCE_SCALE**2 * weight).mean()
            loss = fm + LAMBDA_FORCE * residual
        elif method == "hessian":
            _, _, predicted_hessian = target_pes(endpoint)
            residual = ((predicted_hessian - target_hessian).square().mean((1, 2)) / HESSIAN_SCALE**2 * weight).mean()
            loss = fm + LAMBDA_HESSIAN * residual
        else:
            residual = torch.zeros((), device=DEVICE)
            loss = fm
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    generator_val = torch.Generator(device=DEVICE).manual_seed(8000 + seed)
    xv0 = val_source.repeat_interleave(repeats, 0); xv1 = val_target.repeat_interleave(repeats, 0)
    tv = torch.rand(len(xv0), device=DEVICE, generator=generator_val)
    xv = (1 - tv[:, None]) * xv0 + tv[:, None] * xv1
    uv = xv1 - xv0
    with torch.no_grad():
        pv = model(xv, tv); endpoint = xv + (1 - tv[:, None]) * pv
        _, force_hat, hessian_hat = target_pes(endpoint)
        _, force_ref, hessian_ref = target_pes(xv1)
        mse = (pv - uv).square().mean()
        return model, {
            "validation_mse": float(mse.cpu()),
            "normalized_mse": float((mse / uv.square().mean()).cpu()),
            "one_step_force_residual": float((force_hat - force_ref).square().mean().sqrt().cpu()),
            "one_step_hessian_residual": float((hessian_hat - hessian_ref).square().mean().sqrt().cpu()),
        }


def mmd(a, b):
    aa, bb, ab = torch.cdist(a, a).square(), torch.cdist(b, b).square(), torch.cdist(a, b).square()
    return float((torch.exp(-aa / 0.5).mean() + torch.exp(-bb / 0.5).mean() - 2 * torch.exp(-ab / 0.5).mean()).cpu())


@torch.no_grad()
def rollout(model, source, steps):
    x = source.clone(); path = [x.clone()]
    for i in range(steps):
        t = torch.full((len(x),), (i + 0.5) / steps, device=DEVICE)
        x += model(x, t) / steps
        path.append(x.clone())
    return torch.stack(path)


def evaluate(model, source, target, steps):
    path = rollout(model, source, steps); final = path[-1]
    energy, force, hessian = target_pes(final)
    target_np, final_np = target[:400].cpu().numpy(), final[:400].cpu().numpy()
    cost = cdist(final_np, target_np, "sqeuclidean"); row, col = linear_sum_assignment(cost)
    _, _, target_hessian = target_pes(target)
    return {
        "mmd_biased": mmd(final, target),
        "empirical_w2": float(np.sqrt(cost[row, col].mean())),
        "endpoint_energy": float(energy.mean().cpu()),
        "endpoint_force_rms": float(force.square().sum(1).mean().sqrt().cpu()),
        "endpoint_hessian_to_target_rms": float((hessian - target_hessian).square().mean().sqrt().cpu()),
    }, path


def draw_split_energy(ax, source, target, generated, path, title):
    # Same left-source / middle-purple-trajectory / right-target+generation convention.
    grid_x, grid_y = np.meshgrid(np.linspace(0.4, 3.0, 160), np.linspace(-0.75, 0.75, 130))
    grid = torch.tensor(np.stack((grid_x.ravel(), grid_y.ravel()), -1), device=DEVICE, dtype=DTYPE)
    e, _, _ = target_pes(grid)
    ax.contourf(grid_x + OFFSET, grid_y, e.cpu().numpy().reshape(grid_x.shape), levels=20, cmap="magma_r", alpha=.18, zorder=0)
    ax.scatter(target[:, 0].cpu() + OFFSET, target[:, 1].cpu(), s=7, color=TGT, alpha=.42, edgecolor="none", zorder=1)
    steps = len(path) - 1
    for i in range(len(source)):
        px = [path[k, i, 0].cpu() + (2 * k / steps - 1) * OFFSET for k in range(steps + 1)]
        py = [path[k, i, 1].cpu() for k in range(steps + 1)]
        ax.plot(px, py, color=TRAJ, alpha=.18, lw=.38, zorder=2)
    ax.scatter(generated[:, 0].cpu() + OFFSET, generated[:, 1].cpu(), s=7, color=GEN, alpha=.84, edgecolor="none", zorder=4)
    ax.scatter(source[:, 0].cpu() - OFFSET, source[:, 1].cpu(), s=7, color=SRC, alpha=.9, edgecolor="none", zorder=4)
    ax.set_title(title, fontsize=10, pad=6); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.set_xlim(-6.2, 6.2); ax.set_ylim(-1.15, 1.15)


def figure(seed, source, target, models):
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 9, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0))
    titles = {"fm": "Straight FM", "force": "Endpoint force residual", "hessian": "Endpoint Hessian residual"}
    for ax, method in zip(axes, ("fm", "force", "hessian")):
        _, path = evaluate(models[method], source, target, 8)
        draw_split_energy(ax, source[:100], target[:100], path[-1, :100], path[:, :100], titles[method])
    axes[0].set_ylabel("cooperative relaxation y", fontsize=10)
    fig.text(.06, .12, "source", color=SRC, rotation=90, ha="center", va="center", fontsize=9)
    fig.text(.94, .12, "target energy background", color="#6d5570", rotation=90, ha="center", va="center", fontsize=9)
    fig.subplots_adjust(left=.07, right=.94, top=.84, bottom=.06, wspace=.03)
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(f"results/figures/toy7_endpoint_hessian_seed{seed}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run(seed):
    rng = np.random.default_rng(seed)
    source, target = sample_pairs(rng, 192); val_source, val_target = sample_pairs(rng, 96)
    models, rows = {}, []
    for method in ("fm", "force", "hessian"):
        model, validation = fit(source, target, val_source, val_target, method, seed=100 + seed)
        models[method] = model
        eval_source, eval_target = sample_pairs(rng, 800)
        sampling = {str(nfe): evaluate(model, eval_source, eval_target, nfe)[0] for nfe in (5, 20, 100)}
        rows.append({"method": method, **validation, "sampling": sampling})
    figure(seed, *sample_pairs(rng, 150), models)
    return {"seed": seed, "config": {"path": "straight FM only", "endpoint_estimator": "x_t + (1-t) v_theta", "surrogate": "frozen analytic anisotropic target-well energy/force/Hessian", "time_power": TIME_POWER, "lambda_force": LAMBDA_FORCE, "lambda_hessian": LAMBDA_HESSIAN, "train_pairs": 192, "validation_pairs": 96, "mlp": "3-32-32-2 SiLU, 350 full-batch updates", "device": str(DEVICE)}, "results": rows}


if __name__ == "__main__":
    output = [run(seed) for seed in range(3)]
    Path("results").mkdir(exist_ok=True)
    Path("results/toy7_endpoint_hessian.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))

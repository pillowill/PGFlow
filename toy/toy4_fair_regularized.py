"""Fair paired Toy-4 comparison with path-deviation and smoothness regularizers.

All methods in a seed share endpoints, time samples, random Fourier basis,
ridge budget, and inference source/target samples. Only the path labels differ.
"""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from toy4_control import data, features, mmd

K = 8
BETA, SIGMA = 10.0, 0.5
REG = {"dev": 0.08, "curvature": 0.04, "velocity": 0.04}


def sample_path(nodes, t):
    pos = np.minimum((t * K).astype(int), K - 1)
    frac = (t * K - pos)[:, None]
    rows = np.arange(len(t))
    left, right = nodes[rows, pos], nodes[rows, pos + 1]
    return left + frac * (right - left), K * (right - left)


def scalar_action(nodes):
    d = np.diff(nodes, axis=0)
    mid = (nodes[:-1] + nodes[1:]) / 2
    w = 1.0 + BETA * np.exp(-np.sum(mid * mid, axis=1) / (2 * SIGMA**2))
    return K * 0.5 * float(np.sum(w * np.sum(d * d, axis=1)))


def regularizers(nodes, a, b):
    t = np.linspace(0.0, 1.0, K + 1)[:, None]
    line = a * (1 - t) + b * t
    dev = float(np.mean(np.sum((nodes - line) ** 2, axis=1)))
    curvature = float(np.mean(np.sum((nodes[2:] - 2 * nodes[1:-1] + nodes[:-2]) ** 2, axis=1)))
    velocity = float(np.mean(np.sum((np.diff(nodes, axis=0)[1:] - np.diff(nodes, axis=0)[:-1]) ** 2, axis=1)))
    return {"dev": dev, "curvature": curvature, "velocity": velocity}


def objective(nodes, a, b, lambdas):
    regs = regularizers(nodes, a, b)
    return scalar_action(nodes) + sum(lambdas[k] * regs[k] for k in regs)


def solve_path(a, b, lambdas, optimize=True):
    t = np.linspace(0.0, 1.0, K + 1)[:, None]
    line = a * (1 - t) + b * t
    if not optimize:
        return line, scalar_action(line), regs_for_line(line, a, b)
    direction = b - a
    normal = np.array([-direction[1], direction[0]]) / max(np.linalg.norm(direction), 1e-12)
    best_nodes, best_value = line, objective(line, a, b, lambdas)
    for sign in (-1.0, 1.0):
        initial = line + sign * 0.7 * np.sin(np.pi * t) * normal
        def fun(z):
            nodes = np.vstack((a, z.reshape(K - 1, 2), b))
            return objective(nodes, a, b, lambdas)
        res = minimize(fun, initial[1:-1].ravel(), method="L-BFGS-B",
                       options={"maxiter": 15, "ftol": 1e-6})
        if res.fun < best_value:
            best_value = float(res.fun)
            best_nodes = np.vstack((a, res.x.reshape(K - 1, 2), b))
    return best_nodes, float(best_value), regs_for_line(best_nodes, a, b)


def regs_for_line(nodes, a, b):
    return regularizers(nodes, a, b)


def make_paths(a, b, method):
    lambdas = {"dev": 0.0, "curvature": 0.0, "velocity": 0.0} if method == "straight" else (REG if method == "pg_regularized" else {"dev": 0.0, "curvature": 0.0, "velocity": 0.0})
    rows = [solve_path(x, y, lambdas, optimize=(method != "straight")) for x, y in zip(a, b)]
    return np.array([r[0] for r in rows]), np.array([r[1] for r in rows]), [r[2] for r in rows]


def run(seed):
    rng = np.random.default_rng(seed)
    n, nv, repeats = 20, 10, 4
    a, b = data(rng, n); av, bv = data(rng, nv)
    ts = rng.random(n * repeats); tv = rng.random(nv * repeats)
    w = rng.normal(size=(3, 128)); phase = rng.uniform(0, 2 * np.pi, 128)
    source, target = data(rng, 500)
    results = []
    for method in ("straight", "pg", "pg_regularized"):
        lambdas = {"dev": 0.0, "curvature": 0.0, "velocity": 0.0} if method == "straight" else (REG if method == "pg_regularized" else {"dev": 0.0, "curvature": 0.0, "velocity": 0.0})
        train, action, reg_rows = make_paths(a, b, method if method != "pg" else "pg")
        val, val_action, val_reg_rows = make_paths(av, bv, method if method != "pg" else "pg")
        x, u = sample_path(np.repeat(train, repeats, axis=0), ts)
        xv, uv = sample_path(np.repeat(val, repeats, axis=0), tv)
        # Same velocity-network basis, design, ridge penalty, and budget for every method.
        phi = features(x, ts, w, phase)
        coef = np.linalg.solve(phi.T @ phi + 0.1 * np.eye(phi.shape[1]), phi.T @ u)
        pred = features(xv, tv, w, phase) @ coef
        mse = float(np.mean((pred - uv) ** 2)); moment = float(np.mean(uv * uv))
        generated = source.copy()
        for i in range(50):
            generated += features(generated, np.full(len(generated), (i + 0.5) / 50), w, phase) @ coef / 50
        dist = cdist(generated, target, "sqeuclidean")
        ri, ci = linear_sum_assignment(dist)
        field = features(xv[:200], tv[:200], w, phase) @ coef
        h = 1e-3; grads = []
        for axis in range(2):
            delta = np.eye(2)[axis] * h
            grads.append((features(xv[:200] + delta, tv[:200], w, phase) @ coef - features(xv[:200] - delta, tv[:200], w, phase) @ coef) / (2 * h))
        jacobian = float(np.mean(np.sum(np.stack(grads, axis=-1) ** 2, axis=(1, 2))))
        results.append({"method": method, "validation_mse": mse, "normalized_mse": mse / moment,
                        "mmd_biased": mmd(generated, target), "empirical_w2": float(np.sqrt(dist[ri, ci].mean())),
                        "mean_velocity_norm": float(np.linalg.norm(field, axis=1).mean()),
                        "mean_squared_spatial_jacobian": jacobian, "mean_scalar_action": float(action.mean()),
                        "mean_validation_scalar_action": float(val_action.mean()),
                        "mean_path_regularizers": {k: float(np.mean([r[k] for r in reg_rows])) for k in REG},
                        "mean_validation_path_regularizers": {k: float(np.mean([r[k] for r in val_reg_rows])) for k in REG}})
    return {"seed": seed, "config": {"train_pairs": n, "validation_pairs": nv, "repeats": repeats, "segments": K, "beta": BETA, "sigma": SIGMA, "regularizers": REG}, "results": results}


if __name__ == "__main__":
    rows = [run(seed) for seed in range(2)]
    Path("results").mkdir(exist_ok=True)
    Path("results/toy4_fair_regularized.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))

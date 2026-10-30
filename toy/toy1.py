"""Toy-1: discrete paths in a Gaussian scalar metric."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize


def action(nodes, beta, sigma):
    delta = np.diff(nodes, axis=0)
    mid = (nodes[:-1] + nodes[1:]) / 2
    weight = 1 + beta * np.exp(-np.sum(mid**2, axis=1) / (2 * sigma**2))
    return float(len(delta) * np.sum(weight * np.sum(delta**2, axis=1)) / 2)


def run(beta=10., sigma=.5, k=40):
    t = np.linspace  (0, 1, k + 1)
    straight = np.column_stack((4*t-2, np.zeros_like(t)))
    candidates = [(action(straight, beta, sigma), straight, 'straight', True)]
    for amplitude in (0.5, -0.5, 1., -1.):
        initial = straight.copy()
        initial[:, 1] = amplitude * np.sin(np.pi*t)
        def objective(z):
            nodes = np.vstack((straight[0], z.reshape(-1, 2), straight[-1]))
            return action(nodes, beta, sigma)
        result = minimize(objective, initial[1:-1].ravel(), method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
        nodes = np.vstack((straight[0], result.x.reshape(-1, 2), straight[-1]))
        candidates.append((result.fun, nodes, 'optimized', bool(result.success)))
    value, nodes, kind, success = min(candidates, key=lambda x: x[0])
    return {'beta': beta, 'sigma': sigma, 'segments': k, 'straight_action': action(straight, beta, sigma), 'best_action': float(value), 'candidate': kind, 'solver_success': success, 'max_abs_y': float(np.max(np.abs(nodes[:,1]))), 'path_length': float(np.linalg.norm(np.diff(nodes, axis=0), axis=1).sum()), 'nodes': nodes.tolist()}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/toy1.json')
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = [run(beta=0), run()]
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps([{k:v for k,v in r.items() if k != 'nodes'} for r in result], indent=2))

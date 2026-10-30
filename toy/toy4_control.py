"""Controlled FM path-complexity experiment using random Fourier features."""
import json
from pathlib import Path
import numpy as np


def features(x, t, w, b):
    z = np.column_stack((x, t))
    return np.column_stack((np.ones(len(x)), z, np.sqrt(2 / len(b))*np.cos(z @ w + b)))


def data(rng, n):
    x0 = rng.normal(size=(n, 2))
    centers = np.array([[2, 2], [2, -2], [-2, 2], [-2, -2]])
    x1 = centers[rng.integers(0, 4, n)] + .25*rng.normal(size=(n, 2))
    return x0, x1


def path(x0, x1, t, amplitude):
    d = x1-x0
    normal = np.column_stack((-d[:, 1], d[:, 0]))
    s = np.sin(np.pi*t)[:, None]
    return x0 + t[:, None]*d + amplitude*s*normal, d + amplitude*np.pi*np.cos(np.pi*t)[:, None]*normal


def mmd(x, y):
    def kernel(a,b):
        dist = np.sum((a[:,None,:]-b[None,:,:])**2, axis=2)
        return np.exp(-dist/4)
    return float(kernel(x,x).mean()+kernel(y,y).mean()-2*kernel(x,y).mean())


def experiment(seed, amplitude):
    rng = np.random.default_rng(seed)
    x0,x1 = data(rng, 4000)
    t = rng.random(len(x0))
    x,u = path(x0,x1,t,amplitude)
    w = rng.normal(scale=1., size=(3,128)); b = rng.uniform(0,2*np.pi,128)
    phi = features(x,t,w,b)
    coef = np.linalg.solve(phi.T@phi + .1*np.eye(phi.shape[1]), phi.T@u)
    a,b1 = data(rng, 1500); tv = rng.random(len(a))
    xv,uv = path(a,b1,tv,amplitude)
    pred = features(xv,tv,w,b)@coef
    loss = float(np.mean((pred-uv)**2))
    scale = float(np.mean(uv**2))
    source,target = data(rng, 500)
    generated = source.copy()
    for i in range(50):
        generated += features(generated, np.full(len(generated), (i+.5)/50), w, b)@coef/50
    return {'seed':seed,'amplitude':amplitude,'validation_mse':loss,'target_velocity_second_moment':scale,'normalized_mse':loss/scale,'mmd_biased':mmd(generated,target)}

if __name__ == '__main__':
    results = [experiment(seed, amp) for seed in range(3) for amp in (0., .5)]
    Path('results').mkdir(exist_ok=True)
    Path('results/toy4_control.json').write_text(json.dumps(results, indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2))

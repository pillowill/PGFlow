"""Experimental fixed-target-lattice conditional paths for CrystalFlow.

CPU/SciPy reference implementation, intended for small controlled experiments.
"""
import itertools
import torch
import numpy as np
from scipy.optimize import minimize


def _action(nodes, lattice, images, alpha, cutoff, sigma):
    segments=len(nodes)-1
    total=0.
    for a,b in zip(nodes[:-1],nodes[1:]):
        middle=(a+b)*.5
        velocity=(b-a)@lattice*segments
        total+=np.sum(velocity**2)
        for i in range(len(a)):
            for j in range(i+1,len(a)):
                delta=middle[j]-middle[i]; delta=delta-np.round(delta)
                r=(delta+images)@lattice
                d=np.linalg.norm(r,axis=1); mask=(d<cutoff)&(d>1e-10)
                d=d[mask]; r=r[mask]
                w=np.exp(-d*d/(2*sigma*sigma))*(1-(d/cutoff)**2)**2
                total+=alpha*np.sum(w*(r@(velocity[j]-velocity[i]))**2/d**2)
    return total/(2*segments)


def physical_path(f0, displacement, lattices, num_atoms, times, *, segments=3, alpha=2., cutoff=4., sigma=2., maxiter=40, return_nodes=False):
    """Return positions and exact piecewise-constant velocities, detached from inputs."""
    if segments < 2 or alpha < 0 or cutoff <= 0 or sigma <= 0 or maxiter < 1:
        raise ValueError('Invalid physical path settings')
    f=f0.detach().cpu().double().numpy(); d=displacement.detach().cpu().double().numpy()
    ls=lattices.detach().cpu().double().numpy(); counts=num_atoms.detach().cpu().tolist(); ts=times.detach().cpu().double().numpy()
    positions=[]; velocities=[]; all_nodes=[]; offset=0
    for count,L,t in zip(counts,ls,ts):
        singular=np.linalg.svd(L,compute_uv=False)[-1]
        if not np.isfinite(singular) or singular < 1e-6: raise ValueError('Physical path requires a nonsingular target lattice')
        radius=int(np.ceil(cutoff/singular+.5))
        if radius>12: raise ValueError('Physical path image bound too large; check lattice')
        images=np.array(list(itertools.product(range(-radius,radius+1),repeat=3)))
        start=f[offset:offset+count]; end=start+d[offset:offset+count]
        grid=np.linspace(0,1,segments+1)[:,None,None]; line=start[None]*(1-grid)+end[None]*grid
        def objective(z):
            nodes=np.concatenate((start[None],z.reshape(segments-1,count,3),end[None]))
            return _action(nodes,L,images,alpha,cutoff,sigma)
        baseline=objective(line[1:-1].ravel())
        result=minimize(objective,line[1:-1].ravel(),method='L-BFGS-B',options={'maxiter':maxiter})
        nodes=np.concatenate((start[None],result.x.reshape(segments-1,count,3),end[None])) if np.isfinite(result.fun) and result.fun<baseline else line
        all_nodes.append(torch.as_tensor(nodes, device=f0.device, dtype=f0.dtype))
        index=min(int(t*segments),segments-1); fraction=t*segments-index
        positions.append(nodes[index]*(1-fraction)+nodes[index+1]*fraction)
        velocities.append(segments*(nodes[index+1]-nodes[index]))
        offset+=count
    if offset != len(f): raise ValueError('num_atoms does not match coordinate count')
    result=(torch.as_tensor(np.concatenate(positions),device=f0.device,dtype=f0.dtype), torch.as_tensor(np.concatenate(velocities),device=f0.device,dtype=f0.dtype))
    return (*result, all_nodes) if return_nodes else result

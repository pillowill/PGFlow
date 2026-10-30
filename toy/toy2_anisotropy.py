"""Toy-2: frozen versus position-dependent central-force stiffness."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

K=20; eps=.2
start=np.array([-.9,.5]); end=np.array([.9,.5]); t=np.linspace(0,1,K+1)[:,None]; line=start*(1-t)+end*t

def metric(x,alpha,dynamic):
    y=x if dynamic else np.array([1.,0.]); u=y/np.linalg.norm(y)
    weight=np.exp(-np.dot(y,y)/2) if dynamic else 1.
    return eps*np.eye(2)+alpha*weight*np.outer(u,u)

def action(nodes,alpha,dynamic):
    mid=(nodes[1:]+nodes[:-1])/2; v=np.diff(nodes,axis=0)*K
    return float(sum(z@metric(x,alpha,dynamic)@z for x,z in zip(mid,v))/(2*K))

def solve(alpha,dynamic):
    def f(z): return action(np.vstack((start,z.reshape(K-1,2),end)),alpha,dynamic)
    r=minimize(f,line[1:-1].ravel(),method='L-BFGS-B',options={'maxiter':500})
    path=np.vstack((start,r.x.reshape(K-1,2),end)); value=f(r.x); baseline=action(line,alpha,dynamic)
    if value>baseline: path=line; value=baseline
    return {'alpha':alpha,'dynamic':dynamic,'straight_action':baseline,'optimized_action':value,'max_deviation':float(np.linalg.norm(path-line,axis=1).max()),'distance_range':[float(np.linalg.norm(path,axis=1).min()),float(np.linalg.norm(path,axis=1).max())],'solver_success':bool(r.success)}

rows=[solve(a,d) for a in (0.,1.,5.) for d in (False,True)]
g=metric(np.array([1.,0.]),5.,False)
assert np.allclose(np.linalg.eigvalsh(g),[eps,eps+5])
assert all(x['optimized_action']<=x['straight_action']+1e-8 for x in rows)
assert all(x['max_deviation']<1e-5 for x in rows if not x['dynamic'])
Path('results/toy2_anisotropy.json').write_text(json.dumps(rows,indent=2),encoding='utf8'); print(json.dumps(rows,indent=2))

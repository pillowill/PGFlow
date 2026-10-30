"""Strength scan and nearest-neighbor conditional variance proxy."""
import json, time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import toy4_metric as m
from toy4_control import data, features, mmd


def proxy(x,t,u,k=30):
    z=np.column_stack((x,t*2))
    _,ind=cKDTree(z).query(z,k=k+1)
    neighbor=u[ind[:,1:]]
    return float(np.mean(np.var(neighbor,axis=1)))


def run(seed,beta):
    m.BETA=beta; rng=np.random.default_rng(seed); n=500; nv=200; rep=4
    a,b=data(rng,n); av,bv=data(rng,nv)
    start=time.perf_counter(); paths,st=m.make_paths(a,b); val,sv=m.make_paths(av,bv)
    elapsed=time.perf_counter()-start
    ts=rng.random(n*rep); tv=rng.random(nv*rep)
    x,u=m.sample(np.repeat(paths,rep,axis=0),ts); xv,uv=m.sample(np.repeat(val,rep,axis=0),tv)
    w=rng.normal(size=(3,128)); phase=rng.uniform(0,2*np.pi,128)
    phi=features(x,ts,w,phase); coef=np.linalg.solve(phi.T@phi+.1*np.eye(phi.shape[1]),phi.T@u)
    loss=float(np.mean((features(xv,tv,w,phase)@coef-uv)**2)); moment=float(np.mean(uv**2))
    source,target=data(rng,500); generated=source.copy()
    for i in range(50): generated += features(generated,np.full(500,i/50),w,phase)@coef/50
    return {'seed':seed,'beta':beta,'mean_action':float(st[:,0].mean()),'straight_action':float(st[:,1].mean()),'validation_mse':loss,'normalized_mse':loss/moment,'knn_velocity_variance_proxy':proxy(xv,tv,uv),'mmd_biased':mmd(generated,target),'path_seconds':elapsed}

if __name__=='__main__':
    result=[run(s,b) for s in range(3) for b in (0.,1.,3.,10.)]
    Path('results/toy4_strength_scan.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

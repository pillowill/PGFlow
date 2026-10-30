"""Paired scalar-metric path optimization and 2-D flow matching experiment."""
import json, time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from toy4_control import data, features, mmd

BETA, SIGMA, K = 10., .5, 12


def action_grad(nodes):
    d = np.diff(nodes, axis=0); m = (nodes[1:] + nodes[:-1])/2
    e = BETA*np.exp(-np.sum(m*m, axis=1)/(2*SIGMA**2))
    w = 1+e
    grad_m = -e[:, None]*m/SIGMA**2
    sq = np.sum(d*d, axis=1)
    left = -K*w[:, None]*d + K*.25*sq[:, None]*grad_m
    right = K*w[:, None]*d + K*.25*sq[:, None]*grad_m
    grad = np.zeros_like(nodes); grad[:-1] += left; grad[1:] += right
    return float(K*.5*np.sum(w*sq)), grad


def solve(a,b):
    t = np.linspace(0,1,K+1)[:,None]
    line = a*(1-t)+b*t
    best, _ = action_grad(line); nodes_best = line; success = True
    d = b-a; norm = np.linalg.norm(d)
    normal = np.array([-d[1], d[0]])/max(norm, 1e-12)
    for sign in (-1,1):
        init = line + sign*.7*np.sin(np.pi*t)*normal
        def objective(z):
            nodes = np.vstack((a,z.reshape(K-1,2),b))
            v,g = action_grad(nodes)
            return v,g[1:-1].ravel()
        res = minimize(objective,init[1:-1].ravel(),jac=True,method='L-BFGS-B',options={'maxiter':150,'ftol':1e-9})
        if res.fun < best:
            best, nodes_best, success = float(res.fun), np.vstack((a,res.x.reshape(K-1,2),b)), bool(res.success)
    return nodes_best, best, action_grad(line)[0], success


def sample(nodes, t):
    pos = np.minimum((t*K).astype(int),K-1); frac = (t*K-pos)[:,None]
    rows = np.arange(len(t)); a=nodes[rows,pos]; b=nodes[rows,pos+1]
    return a+frac*(b-a), K*(b-a)


def make_paths(a,b):
    result = [solve(x,y) for x,y in zip(a,b)]
    return np.array([r[0] for r in result]), np.array([[r[1],r[2],r[3]] for r in result])


def run(seed):
    rng=np.random.default_rng(seed); n=500; nv=200; repeats=4
    a,b=data(rng,n); av,bv=data(rng,nv)
    start=time.perf_counter(); opt,stats=make_paths(a,b); optv,statsv=make_paths(av,bv); elapsed=time.perf_counter()-start
    grid=np.linspace(0,1,K+1)[None,:,None]
    line=a[:,None,:]*(1-grid)+b[:,None,:]*grid
    linev=av[:,None,:]*(1-grid)+bv[:,None,:]*grid
    ts=rng.random(n*repeats); tv=rng.random(nv*repeats)
    w=rng.normal(size=(3,128)); phase=rng.uniform(0,2*np.pi,128)
    source,target=data(rng,500)
    results=[]
    for name,train,val in [('straight',line,linev),('metric',opt,optv)]:
        x,u=sample(np.repeat(train,repeats,axis=0),ts)
        phi=features(x,ts,w,phase); coef=np.linalg.solve(phi.T@phi+.1*np.eye(phi.shape[1]),phi.T@u)
        xv,uv=sample(np.repeat(val,repeats,axis=0),tv); pred=features(xv,tv,w,phase)@coef
        loss=float(np.mean((pred-uv)**2)); moment=float(np.mean(uv**2))
        scores={}
        for steps in (10,50,100):
            generated=source.copy()
            for i in range(steps):
                generated += features(generated,np.full(len(source),i/steps),w,phase)@coef/steps
            scores[str(steps)]=mmd(generated,target)
        results.append({'method':name,'validation_mse':loss,'velocity_second_moment':moment,'normalized_mse':loss/moment,'mmd_biased':scores})
    return {'seed':seed,'train_pairs':n,'validation_pairs':nv,'samples_per_pair':repeats,'segments':K,'beta':BETA,'sigma':SIGMA,'path_seconds':elapsed,'mean_metric_action':float(stats[:,0].mean()),'mean_straight_action':float(stats[:,1].mean()),'fraction_optimized':float(np.mean(stats[:,0] < stats[:,1]-1e-8)),'selected_solver_success_fraction':float(stats[:,2].mean()),'validation_mean_metric_action':float(statsv[:,0].mean()),'validation_mean_straight_action':float(statsv[:,1].mean()),'results':results}

if __name__ == '__main__':
    results=[run(i) for i in range(3)]
    Path('results').mkdir(exist_ok=True)
    Path('results/toy4_metric.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2))

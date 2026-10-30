"""Toy-4: anisotropic metric and explicit potential path ablations."""
import json,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment
from scipy.spatial.distance import cdist
from toy4_control import data,features,mmd
from toy4_metric import sample

K=12; BETA=10.; SIGMA=.5

def objective(nodes,mode):
    d=np.diff(nodes,axis=0); mid=(nodes[:-1]+nodes[1:])/2
    e=np.exp(-np.sum(mid**2,axis=1)/(2*SIGMA**2)); ge=-e[:,None]*mid/SIGMA**2
    if mode=='anisotropic':
        sq=d[:,0]**2; grad_d=d.copy(); grad_d[:,0]+=BETA*e*d[:,0]
    else:
        sq=np.sum(d*d,axis=1); grad_d=(1+BETA*e)[:,None]*d
    val=K*.5*np.sum(np.sum(d*d,axis=1)+BETA*e*sq) if mode=='anisotropic' else K*.5*np.sum((1+BETA*e)*sq)
    gm=K*.5*BETA*sq[:,None]*ge
    if mode=='repulsion': val+=3*np.mean(e); gm+=3/K*ge
    gd=K*grad_d
    g=np.zeros_like(nodes); g[:-1]+=-gd+.5*gm; g[1:]+=gd+.5*gm
    return float(val),g

def solve(a,b,mode):
    t=np.linspace(0,1,K+1)[:,None]; line=a*(1-t)+b*t
    best=objective(line,mode)[0]; selected=line
    if mode=='straight': return line,best,best
    direction=b-a; normal=np.array([-direction[1],direction[0]])/max(np.linalg.norm(direction),1e-9)
    for sign in (-1,1):
        initial=line+sign*.7*np.sin(np.pi*t)*normal
        def fun(z):
            x=np.vstack((a,z.reshape(K-1,2),b)); v,g=objective(x,mode); return v,g[1:-1].ravel()
        r=minimize(fun,initial[1:-1].ravel(),jac=True,method='L-BFGS-B',options={'maxiter':100,'ftol':1e-8})
        if r.fun<best: best=float(r.fun); selected=np.vstack((a,r.x.reshape(K-1,2),b))
    return selected,best,objective(line,mode)[0]

def paths(a,b,mode):
    result=[solve(x,y,mode) for x,y in zip(a,b)]
    return np.array([r[0] for r in result]),np.array([[r[1],r[2]] for r in result])

def experiment(seed,mode):
    rng=np.random.default_rng(seed); n=300; nv=120; repeat=4
    a,b=data(rng,n); av,bv=data(rng,nv); start=time.perf_counter(); x,stats=paths(a,b,mode); xv,sv=paths(av,bv,mode); seconds=time.perf_counter()-start
    ts=rng.random(n*repeat); tv=rng.random(nv*repeat); w=rng.normal(size=(3,128)); phase=rng.uniform(0,2*np.pi,128)
    xt,ut=sample(np.repeat(x,repeat,axis=0),ts); phi=features(xt,ts,w,phase); coef=np.linalg.solve(phi.T@phi+.1*np.eye(phi.shape[1]),phi.T@ut)
    z,u=sample(np.repeat(xv,repeat,axis=0),tv); pred=features(z,tv,w,phase)@coef; loss=float(np.mean((pred-u)**2)); moment=float(np.mean(u*u))
    source,target=data(rng,500); scores={}
    for steps in (10,50,100):
        gen=source.copy(); potential=[]
        for i in range(steps):
            potential.append(np.exp(-np.sum(gen*gen,axis=1)/(2*SIGMA**2)).mean()); gen+=features(gen,np.full(len(gen),i/steps),w,phase)@coef/steps
        signs=(gen>0).astype(int); counts=np.bincount(2*signs[:,0]+signs[:,1],minlength=4)
        distances=cdist(gen,target,'sqeuclidean'); ri,ci=linear_sum_assignment(distances)
        scores[str(steps)]={'empirical_w2':float(np.sqrt(distances[ri,ci].mean())),'mmd_biased':mmd(gen,target),'quadrant_fractions':(counts/len(gen)).tolist(),'trajectory_gaussian_potential':float(np.mean(potential))}
    eval_x=z[:200]; eval_t=tv[:200]; h=1e-3
    field=features(eval_x,eval_t,w,phase)@coef
    derivatives=[(features(eval_x+np.eye(2)[i]*h,eval_t,w,phase)@coef-features(eval_x-np.eye(2)[i]*h,eval_t,w,phase)@coef)/(2*h) for i in range(2)]
    field_norm=float(np.mean(np.linalg.norm(field,axis=1)))
    smoothness=float(np.mean(np.sum(np.stack(derivatives,axis=-1)**2,axis=(1,2))))
    return {'mean_velocity_norm':field_norm,'mean_squared_spatial_jacobian':smoothness,'seed':seed,'method':mode,'mean_objective':float(stats[:,0].mean()),'mean_straight_objective':float(stats[:,1].mean()),'validation_objective':float(sv[:,0].mean()),'validation_mse':loss,'normalized_mse':loss/moment,'path_seconds':seconds,'sampling':scores}

if __name__=='__main__':
    from scipy.optimize import check_grad
    rng=np.random.default_rng(8); x=rng.normal(size=(K+1,2))
    for mode in ('anisotropic','repulsion'):
        err=check_grad(lambda z:objective(z.reshape(-1,2),mode)[0],lambda z:objective(z.reshape(-1,2),mode)[1].ravel(),x.ravel())
        assert err<1e-3,(mode,err)
    result=[experiment(seed,mode) for seed in range(3) for mode in ('straight','scalar','anisotropic','repulsion')]
    Path('results/toy4_ablation.json').write_text(json.dumps(result,indent=2),encoding='utf8'); print(json.dumps(result,indent=2))

"""Exploratory fixed-lattice MP-20 path experiment; target-dependent source."""
import argparse, csv, hashlib, json, time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from pymatgen.core import Structure

IMAGES=np.array([(i,j,k) for i in (-1,0,1) for j in (-1,0,1) for k in (-1,0,1)])
K=3; ALPHA=2.; SIGMA=2.

def metric_action(nodes,L):
    total=0.
    for x,y in zip(nodes[:-1],nodes[1:]):
        mid=(x+y)/2; v=(y-x)@L*K
        total+=np.sum(v*v)
        for i in range(len(x)):
            for j in range(i+1,len(x)):
                r=(mid[j]-mid[i]+IMAGES)@L
                d2=np.sum(r*r,axis=1); safe=np.maximum(d2,1e-12)
                weight=np.exp(-d2/(2*SIGMA**2))
                projections=r@(v[j]-v[i])
                total+=ALPHA*np.sum(weight*projections**2/safe)
    return float(total/(2*K))

def minimum_distance(nodes,L):
    closest=np.inf
    for x in nodes:
        for i in range(len(x)):
            for j in range(i+1,len(x)):
                closest=min(closest,float(np.linalg.norm((x[j]-x[i]+IMAGES)@L,axis=1).min()))
    return closest

def run(path,count,seed):
    rng=np.random.default_rng(seed); records=[]; start=time.perf_counter()
    with open(path,encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f):
            s=Structure.from_str(row['cif'],fmt='cif'); n=len(s)
            if not 2<=n<=5: continue
            L=s.lattice.matrix; target=s.frac_coords
            # Cartesian perturbation; lift ends exactly at the observed target.
            source=target+rng.normal(0,.25,size=(n,3))@np.linalg.inv(L)
            t=np.linspace(0,1,K+1)[:,None,None]; line=source[None]*(1-t)+target[None]*t
            def obj(z): return metric_action(np.concatenate((source[None],z.reshape(K-1,n,3),target[None])),L)
            initial=line[1:-1].ravel(); baseline=obj(initial)
            result=minimize(obj,initial,method='L-BFGS-B',options={'maxiter':80,'ftol':1e-9})
            optimized=np.concatenate((source[None],result.x.reshape(K-1,n,3),target[None])) if result.fun<baseline else line
            records.append({'material_id':row['material_id'],'formula':row['pretty_formula'],'n_atoms':n,'straight_action':baseline,'optimized_action':metric_action(optimized,L),'straight_min_distance_angstrom':minimum_distance(line,L),'optimized_min_distance_angstrom':minimum_distance(optimized,L),'solver_success':bool(result.success),'solver_message':str(result.message),'endpoint_error':float(np.max(np.abs(optimized[-1]-target)))})
            if len(records)>=count: break
    digest=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return {'dataset':str(path),'sha256':digest,'split':'val','selection':'first eligible rows, 2–5 atoms','seed':seed,'source_cartesian_noise_std_angstrom':.25,'segments':K,'alpha':ALPHA,'sigma_angstrom':SIGMA,'elapsed_seconds':time.perf_counter()-start,'records':records}

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--data',required=True); parser.add_argument('--count',type=int,default=15); parser.add_argument('--seed',type=int,default=0); parser.add_argument('--output',default='results/mp20_path_pilot.json'); args=parser.parse_args()
    result=run(args.data,args.count,args.seed); Path(args.output).write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result,indent=2))

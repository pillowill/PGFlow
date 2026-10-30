"""Periodic MP-20 path diagnostic with bounded, smooth image sums."""
import argparse, csv, hashlib, itertools, json, time, warnings
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from pymatgen.core import Structure

K=3; CUTOFF=4.; ALPHA=2.; SIGMA=2.

def images_for(L, radius):
    bound=int(np.ceil(radius/np.linalg.svd(L,compute_uv=False)[-1]+.5))
    if bound>12: raise ValueError('cell too ill-conditioned for image enumeration')
    return np.array(list(itertools.product(range(-bound,bound+1),repeat=3)))

def vectors(delta,L,images):
    # This lift selection permutes a complete cutoff-bounded image sum.
    return (delta-np.round(delta)+images)@L

def action(nodes,L,images):
    result=0.
    for x,y in zip(nodes[:-1],nodes[1:]):
        mid=(x+y)/2; v=(y-x)@L*K
        result += np.sum(v*v)
        for i in range(len(x)):
            for j in range(i+1,len(x)):
                r=vectors(mid[j]-mid[i],L,images); d=np.linalg.norm(r,axis=1)
                inside=(d<CUTOFF)&(d>1e-10); d=d[inside]; r=r[inside]
                w=np.exp(-d*d/(2*SIGMA**2))*(1-(d/CUTOFF)**2)**2
                result+=ALPHA*np.sum(w*(r@(v[j]-v[i]))**2/d**2)
    return float(result/(2*K))

def min_distance(nodes,L,images):
    minimum=float('inf'); self_vectors=images@L
    self_d=np.linalg.norm(self_vectors,axis=1); minimum=float(self_d[self_d>1e-10].min())
    for a,b in zip(nodes[:-1],nodes[1:]):
        for frac in np.linspace(0,1,11):
            x=a*(1-frac)+b*frac
            for i in range(len(x)):
                for j in range(i+1,len(x)):
                    minimum=min(minimum,float(np.linalg.norm(vectors(x[j]-x[i],L,images),axis=1).min()))
    return minimum

def load(path,n,seed):
    with open(path,encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f))
    order=np.random.default_rng(seed).permutation(len(rows)); found=[]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for idx in order:
            row=rows[idx]
            try: s=Structure.from_str(row['cif'],fmt='cif')
            except Exception: continue
            if not 2<=len(s)<=5: continue
            try: images_for(s.lattice.matrix,CUTOFF)
            except ValueError: continue
            found.append((row,s))
            if len(found)==n: break
    return found

def run(path,n):
    structures=load(path,n,2026); out=[]; start=time.perf_counter()
    for seed in (0,1,2):
        for noise in (.25,.75):
            rng=np.random.default_rng(seed)
            for row,s in structures:
                L=s.lattice.matrix; target=s.frac_coords; atoms=len(s); images=images_for(L,max(CUTOFF, np.linalg.norm(L,axis=1).sum()/2))
                source=target+rng.normal(0,noise,size=(atoms,3))@np.linalg.inv(L)
                t=np.linspace(0,1,K+1)[:,None,None]; line=source[None]*(1-t)+target[None]*t
                def obj(z): return action(np.concatenate((source[None],z.reshape(K-1,atoms,3),target[None])),L,images)
                base=obj(line[1:-1].ravel()); opt=minimize(obj,line[1:-1].ravel(),method='L-BFGS-B',options={'maxiter':80,'ftol':1e-9})
                nodes=np.concatenate((source[None],opt.x.reshape(K-1,atoms,3),target[None])) if opt.fun<base else line
                out.append({'id':row['material_id'],'n':atoms,'seed':seed,'noise_angstrom':noise,'straight_action':base,'optimized_action':action(nodes,L,images),'straight_min_distance':min_distance(line,L,images),'optimized_min_distance':min_distance(nodes,L,images),'solver_success':bool(opt.success),'message':str(opt.message)})
    return {'data':str(path),'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'split':'val','selection':'random permutation seed 2026, eligible 2-5 atoms','cutoff_angstrom':CUTOFF,'elapsed_seconds':time.perf_counter()-start,'records':out}

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--count',type=int,default=15); p.add_argument('--output',default='results/mp20_path_scan.json'); args=p.parse_args()
    result=run(args.data,args.count); Path(args.output).write_text(json.dumps(result,indent=2),encoding='utf-8'); print('completed',len(result['records']),'cases in',result['elapsed_seconds'],'seconds')


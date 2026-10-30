"""Compare CPU SciPy and GPU Torch paths on identical MP-20 validation CIFs."""
import csv, json, os, sys, time
from pathlib import Path
root=Path(__file__).resolve().parents[1]; project=root/'repos'/'CrystalFlow-project'; sys.path.insert(0,str(project)); os.environ['PROJECT_ROOT']=str(project)
import numpy as np
import torch
from pymatgen.core import Structure
from diffcsp.pl_modules.physical_path import physical_path, _action
from diffcsp.pl_modules.physical_path_torch import physical_path_torch, torch_action
import itertools, math

def load():
    cases=[]
    with (project/'data/mp_20/val.csv').open(encoding='utf8',newline='') as f:
        for row in csv.DictReader(f):
            s=Structure.from_str(row['cif'],fmt='cif')
            if 2<=len(s)<=5:
                cases.append((row['material_id'],s))
                if len(cases)==4: break
    return cases

def main():
    assert torch.cuda.is_available()
    dev='cuda'; torch.manual_seed(123); rows=[]
    # Warm the CUDA allocator and optimizer outside the timed region.
    for id,s in load():
        f1=torch.tensor(s.frac_coords,device=dev,dtype=torch.float64); f0=torch.rand_like(f1)
        d=(f1-f0-.5).remainder(1)-.5
        L=torch.tensor(s.lattice.matrix,device=dev,dtype=torch.float64); n=torch.tensor([len(s)],device=dev); times=torch.tensor([.41],device=dev,dtype=torch.float64)
        r=math.ceil(4/torch.linalg.svdvals(L).min().item()+.5)
        images=torch.tensor(list(itertools.product(range(-r,r+1),repeat=3)),device=dev,dtype=torch.float64)
        grid=torch.linspace(0,1,4,device=dev,dtype=torch.float64)[:,None,None]; line=f0[None]+grid*d[None]
        cpu=_action(line.cpu().numpy(),L.cpu().numpy(),images.cpu().numpy(),2,4,2)
        gpu=torch_action(line,L,images,2,4,2).item()
        assert np.isclose(cpu,gpu,rtol=1e-10,atol=1e-10)
        physical_path_torch(f0,d,L[None],n,times,segments=3,maxiter=1)
        torch.cuda.synchronize()
        results={}
        for name,fn,iterations in [('scipy',physical_path,3),('torch',physical_path_torch,20)]:
            torch.cuda.synchronize(); start=time.perf_counter(); x,v,nodes=fn(f0,d,L[None],n,times,segments=3,maxiter=iterations,return_nodes=True); torch.cuda.synchronize(); elapsed=time.perf_counter()-start
            assert torch.isfinite(x).all() and torch.isfinite(v).all()
            results[name]={'seconds':elapsed,'position_norm':float(x.norm()),'velocity_norm':float(v.norm()),'action':float(torch_action(nodes[0],L,images,2,4,2))}
        rows.append({'id':id,'atoms':len(s),'straight_action':cpu,'results':results})
    out={'gpu':torch.cuda.get_device_name(), 'dtype':'float64','scipy_maxiter':3,'torch_steps':20,'note':'different optimizers and convergence; timing alone does not establish equal solution quality','cases':rows}
    (root/'results/pgflow_gpu_benchmark.json').write_text(json.dumps(out,indent=2),encoding='utf8'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()

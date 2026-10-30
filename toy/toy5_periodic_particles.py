"""Toy-5: periodic particle stiffness paths vs straight paths."""
import sys,time,json,itertools,math
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'repos'/'CrystalFlow-project'))
from diffcsp.pl_modules.physical_path_torch import physical_path_torch,torch_action


def metrics(nodes,L,bonds,reference):
    samples=torch.stack([a*(1-t)+b*t for a,b in zip(nodes[:-1],nodes[1:]) for t in torch.linspace(0,1,11,device=L.device)])
    delta=samples[:,:,None,:]-samples[:,None,:,:]; delta=delta-delta.round()
    dist=torch.linalg.vector_norm(delta@L,dim=-1)
    ij=torch.triu_indices(nodes.shape[1],nodes.shape[1],1,device=L.device)
    pair=dist[:,ij[0],ij[1]]
    rep=(1.1-pair).clamp_min(0).square().sum(dim=1)
    bd=dist[:,bonds[0],bonds[1]]
    strain=(bd-reference).abs().mean(dim=1)
    energy=rep+0.5*((bd-reference)**2).sum(dim=1)
    return {'min_distance':float(pair.min()),'mean_bond_strain':float(strain.mean()),'peak_toy_energy':float(energy.max())}


def run(N,seed,device):
    torch.manual_seed(seed); side=math.ceil(math.sqrt(N)); box=side*1.6
    L=torch.eye(3,device=device,dtype=torch.float64)*box
    target=torch.tensor([[(i%side+.5)/side,(i//side+.5)/side,.5] for i in range(N)],device=device,dtype=torch.float64)
    source=(target+torch.randn_like(target)*(.55/box))%1
    d=(target-source+.5).remainder(1)-.5
    t=torch.tensor([.3],device=device,dtype=torch.float64); count=torch.tensor([N],device=device)
    grid=torch.linspace(0,1,4,device=device,dtype=torch.float64)[:,None,None]; line=source[None]+grid*d[None]
    ij=torch.triu_indices(N,N,1,device=device); dd=target[ij[1]]-target[ij[0]]; dd=dd-dd.round(); targetdist=torch.linalg.vector_norm(dd@L,dim=1)
    bonds=ij[:,targetdist<1.7]; ref=targetdist[targetdist<1.7]
    radius=math.ceil(4/box+.5); images=torch.tensor(list(itertools.product(range(-radius,radius+1),repeat=3)),device=device,dtype=torch.float64)
    if device=='cuda': torch.cuda.synchronize()
    start=time.perf_counter(); _,_,paths=physical_path_torch(source,d,L[None],count,t,segments=3,maxiter=100,lr=.003,return_nodes=True)
    if device=='cuda': torch.cuda.synchronize()
    elapsed=time.perf_counter()-start; opt=paths[0]
    A=float(torch_action(line,L,images,2,4,2)); B=float(torch_action(opt,L,images,2,4,2)); assert B <= A+1e-8
    return {'n':N,'seed':seed,'seconds':elapsed,'n_bonds':len(ref),'straight_action':A,'optimized_action':B,'straight':metrics(line,L,bonds,ref),'optimized':metrics(opt,L,bonds,ref)}

if __name__=='__main__':
    device='cuda' if torch.cuda.is_available() else 'cpu'
    run(8,99,device)
    rows=[run(n,seed,device) for n in (8,16,32) for seed in range(3)]
    out={'device':device,'description':'Fixed square periodic cell; source is target plus Gaussian displacement; target-dependent diagnostic, not a generative task. Bond energy is a toy proxy, not DFT. Distances sampled at 11 points/segment.','results':rows}
    Path('results/toy5_periodic_particles.json').write_text(json.dumps(out,indent=2),encoding='utf8'); print(json.dumps(out,indent=2))

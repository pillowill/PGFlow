"""Invariants for the torch physical path implementation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'repos'/'CrystalFlow-project'))
import torch
from diffcsp.pl_modules.physical_path_torch import physical_path_torch, torch_action
from diffcsp.pl_modules.physical_path import _action
import numpy as np
from scipy.optimize import check_grad
import itertools

def check(device):
 f=torch.tensor([[.1,.2,.3],[.7,.6,.8]],device=device,dtype=torch.float64)
 d=torch.tensor([[.3,-.1,.2],[-.2,.2,-.3]],device=device,dtype=torch.float64)
 L=torch.tensor([[[4.,0,0],[1.,4.,0],[0,0,4.]]],device=device,dtype=torch.float64); n=torch.tensor([2],device=device)
 def run(t,alpha=2.):
  with torch.inference_mode(): return physical_path_torch(f,d,L,n,torch.tensor([t],device=device,dtype=torch.float64),segments=3,alpha=alpha,maxiter=5)
 x0,_=run(0); x1,_=run(1); h=1e-5; x,v=run(.4); a,_=run(.4-h); b,_=run(.4+h)
 assert torch.allclose(x0,f,atol=1e-8) and torch.allclose(x1,f+d,atol=1e-8)
 assert torch.allclose((b-a)/(2*h),v,atol=1e-7)
 x,v=run(.4,0); assert torch.allclose(x,f+.4*d,atol=1e-8) and torch.allclose(v,d,atol=1e-8)
 print(device,'endpoint, exact velocity, alpha=0 and inference-mode checks passed')
check('cpu')
if torch.cuda.is_available(): check('cuda')

L=np.array([[3.,0.,0.],[1.,4.,0.],[0.,0.,4.]])
im=np.array(list(itertools.product(range(-3,4),repeat=3)))
rng=np.random.default_rng(10); z=rng.uniform(.1,.9,size=(4,3,3))
def fun(x): return _action(x.reshape(z.shape),L,im,2.,4.,2.)
def grad(x):
 q=torch.tensor(x.reshape(z.shape),dtype=torch.float64,requires_grad=True)
 value=torch_action(q,torch.tensor(L),torch.tensor(im,dtype=torch.float64),2.,4.,2.)
 return torch.autograd.grad(value,q)[0].detach().numpy().ravel()
error=check_grad(fun,grad,z.ravel()); assert error<1e-3,error
print('NumPy action vs torch gradient finite difference error:',error)

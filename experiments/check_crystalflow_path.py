import sys
sys.path.insert(0,'CrystalFlow-project')
import torch
from diffcsp.pl_modules.physical_path import physical_path
f=torch.tensor([[.1,.2,.3],[.7,.6,.8]],dtype=torch.float64)
d=torch.tensor([[.3,-.1,.2],[-.2,.2,-.3]],dtype=torch.float64)
L=torch.tensor([[[4.,0,0],[1.,4.,0],[0,0,4.]]],dtype=torch.float64); n=torch.tensor([2])
kw=dict(segments=3,alpha=2.,maxiter=60)
x0,_=physical_path(f,d,L,n,torch.tensor([0.]),**kw)
x1,_=physical_path(f,d,L,n,torch.tensor([1.]),**kw)
t=torch.tensor([.4],dtype=torch.float64); h=1e-5
x,v=physical_path(f,d,L,n,t,**kw); a,_=physical_path(f,d,L,n,t-h,**kw); b,_=physical_path(f,d,L,n,t+h,**kw)
assert torch.allclose(x0,f,atol=1e-8)
assert torch.allclose(x1,f+d,atol=1e-8)
assert torch.allclose((b-a)/(2*h),v,atol=1e-7)
x,v=physical_path(f,d,L,n,t,segments=3,alpha=0.,maxiter=60)
assert torch.allclose(x,f+t*d,atol=1e-6)
assert torch.allclose(v,d,atol=1e-6)
print('endpoint, exact velocity, and alpha=0 checks passed')


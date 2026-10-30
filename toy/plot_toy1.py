"""Render Toy-1 paths and their exact segment speeds."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from toy1 import run,action
beta=10.; sigma=.5; k=40
nodes=np.array(run(beta,sigma,k)['nodes']); t=np.linspace(0,1,k+1); line=np.column_stack((4*t-2,np.zeros_like(t)))
def sine(a): return line+np.column_stack((np.zeros_like(t),a*np.sin(np.pi*t)))
r=minimize_scalar(lambda a:action(sine(a),beta,sigma),bounds=(0,3),method='bounded'); curved=sine(r.x)
q=np.linspace(-2.5,2.5,200); x,y=np.meshgrid(q,q); z=1+beta*np.exp(-(x*x+y*y)/(2*sigma*sigma))
fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True); ax=axes[0]; im=ax.pcolormesh(x,y,z,shading='auto',cmap='magma_r')
for path,label,style in ((line,'straight','--'),(curved,'sine',':'),(nodes,'optimized','-')):
 ax.plot(path[:,0],path[:,1],style,label=label); axes[1].plot((t[:-1]+t[1:])/2,np.linalg.norm(np.diff(path,axis=0)*k,axis=1),style,label=label)
ax.set_aspect('equal'); ax.set(xlabel='x',ylabel='y',title='Metric and paths'); ax.legend(); fig.colorbar(im,ax=ax,label='metric multiplier'); axes[1].set(xlabel='time',ylabel='segment speed',title='Exact piecewise velocity'); axes[1].legend(); Path('results/figures').mkdir(exist_ok=True); fig.savefig('results/figures/toy1_metric_path.png',dpi=160)

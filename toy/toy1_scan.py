"""Toy-1 grid scan with straight, sine, and free-node candidates."""
import json,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from toy1 import action,run

rows=[]
for beta in (0.,3.,10.):
 for sigma in (.35,.7):
  for k in (16,32):
   start=time.perf_counter(); t=np.linspace(0,1,k+1); line=np.column_stack((4*t-2,np.zeros(k+1)))
   def path(a): return line+np.column_stack((np.zeros(k+1),a*np.sin(np.pi*t)))
   sine=minimize_scalar(lambda a: action(path(a),beta,sigma),bounds=(0,3),method='bounded')
   r=run(beta,sigma,k); nodes=np.asarray(r.pop('nodes')); r.update({'sine_action':min(action(line,beta,sigma),float(sine.fun)),'elapsed_seconds':time.perf_counter()-start,'endpoint_error':float(max(np.linalg.norm(nodes[0]-line[0]),np.linalg.norm(nodes[-1]-line[-1]))),'velocity_max_norm':float(np.linalg.norm(np.diff(nodes,axis=0)*k,axis=1).max())})
   assert r['best_action']<=r['straight_action']+1e-8 and r['endpoint_error']<1e-12
   if beta==0: assert np.allclose(nodes,line,atol=1e-7)
   rows.append(r)
Path('results/toy1_scan.json').write_text(json.dumps(rows,indent=2),encoding='utf8'); print(json.dumps(rows,indent=2))

"""Two particles on a square torus: stiffness quadratic form and null modes."""
import json
from pathlib import Path
import numpy as np


def stiffness(f, epsilon=.1, alpha=2.):
    delta=f[1]-f[0]; delta-=np.round(delta)
    n=delta/np.linalg.norm(delta)
    B=np.r_[-n,n]
    return epsilon*np.eye(4)+alpha*np.outer(B,B),delta

f=np.array([[.95,.5],[.05,.5]])
g,delta=stiffness(f)
translation=np.array([1.,0.,1.,0.]); radial=np.array([-1.,0.,1.,0.]); tangent=np.array([0.,-1.,0.,1.])
result={'fractional_positions':f.tolist(),'minimum_image':delta.tolist(),'eigenvalues':np.linalg.eigvalsh(g).tolist(),'translation_quadratic':float(translation@g@translation),'radial_quadratic':float(radial@g@radial),'tangent_quadratic':float(tangent@g@tangent),'epsilon':.1,'alpha':2.}
assert np.allclose(delta,[.1,0])
assert np.allclose(np.linalg.eigvalsh(g),[.1,.1,.1,4.1])
assert np.isclose(result['radial_quadratic'],8.2)
assert np.isclose(result['tangent_quadratic'],.2)
Path('results/toy2_periodic.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))

"""Toy-3: periodic nearest images and same-species assignments."""
import itertools, json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment


def nearest(delta, lattice, radius=8):
    images=np.array(list(itertools.product(range(-radius,radius+1),repeat=len(delta))))
    candidates=delta+images
    lengths=np.linalg.norm(candidates@lattice,axis=1)
    k=int(np.argmin(lengths))
    assert np.all(np.abs(images[k])<radius), 'enlarge image search'
    return candidates[k],float(lengths[k]),images[k]


def match(source,target,lattice,types):
    costs=np.zeros((len(source),len(target)))
    for i in range(len(source)):
        for j in range(len(target)):
            costs[i,j]=nearest(target[j]-source[i],lattice)[1]**2 if types[i]==types[j] else 1e12
    i,j=linear_sum_assignment(costs)
    return j,float(costs[i,j].sum())

L=np.array([[1.,0.],[.9,.2]])
d=np.array([.49,.49]); exact,length,image=nearest(d,L); wrapped=d-np.round(d)
s=np.array([[.1,.1],[.8,.1],[.4,.7]])
t=np.array([[.79,.1],[.11,.1],[.42,.69]])
assignment,cost=match(s,t,np.eye(2),['A','A','B'])
identity=sum(nearest(t[i]-s[i],np.eye(2))[1]**2 for i in range(3))
cross=nearest(np.array([.05-.95]),np.ones((1,1)))
assert np.isclose(cross[0][0],.1)
assert length < np.linalg.norm(wrapped@L)
assert assignment.tolist()==[1,0,2] and cost<identity
result={'cross_boundary_displacement':cross[0].tolist(),'skew_lattice':L.tolist(),'skew_fractional_delta':d.tolist(),'component_wrap_length':float(np.linalg.norm(wrapped@L)),'nearest_delta':exact.tolist(),'nearest_image':image.tolist(),'nearest_length':length,'assignment':assignment.tolist(),'matched_cost':cost,'identity_cost':identity,'limitations':'Fixed translation and atom types; finite image enumeration with interior optimum check, no full lattice-basis or origin alignment.'}
Path('results/toy3_periodic_matching.json').write_text(json.dumps(result,indent=2),encoding='utf8'); print(json.dumps(result,indent=2))

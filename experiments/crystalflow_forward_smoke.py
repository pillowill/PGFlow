"""Small real-CIF CrystalFlow forward/backward integration check."""
import csv, os, sys, time
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'CrystalFlow-project'
os.environ['PROJECT_ROOT']=str(root); sys.path.insert(0,str(root))
import torch
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate
from pymatgen.core import Structure
from torch_geometric.data import Data, Batch
from diffcsp.common.data_utils import lattice_polar_decompose

with initialize_config_dir(config_dir=str(root/'conf'),version_base=None):
    cfg=compose(config_name='default',overrides=['data=mp_20','model=flow_polar','model.decoder.hidden_dim=32','model.decoder.num_layers=2','model.decoder.num_freqs=16','model.time_dim=32'])
rows=[]
with (root/'data/mp_20/val.csv').open(encoding='utf-8',newline='') as f:
    for row in csv.DictReader(f):
        s=Structure.from_str(row['cif'],fmt='cif')
        if 2<=len(s)<=5:
            rows.append((row['material_id'],s))
            if len(rows)==2: break
items=[]
for name,s in rows:
    items.append(Data(frac_coords=torch.tensor(s.frac_coords,dtype=torch.float32),atom_types=torch.tensor([x.specie.Z for x in s],dtype=torch.long),lengths=torch.tensor([s.lattice.abc],dtype=torch.float32),angles=torch.tensor([s.lattice.angles],dtype=torch.float32),lattice_polar=torch.tensor([lattice_polar_decompose(s.lattice.matrix)],dtype=torch.float32),num_atoms=len(s),num_nodes=len(s)))
device = "cuda" if "--cuda" in sys.argv else "cpu"
batch=Batch.from_data_list(items).to(device)
for enabled in (False,True):
    torch.manual_seed(19)
    cfg.model.physical_path_enabled=enabled
    cfg.model.physical_path_backend="torch" if "--torch-path" in sys.argv else "scipy"
    model=instantiate(cfg.model,_recursive_=False).to(device)
    torch.manual_seed(42); start=time.perf_counter()
    out=model(batch); out['loss'].backward()
    assert torch.isfinite(out['loss'])
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    print('enabled',enabled,'ids',[x[0] for x in rows],'loss',float(out['loss'].detach()),'coord',float(out['loss_coord'].detach()),'seconds',round(time.perf_counter()-start,3),flush=True)


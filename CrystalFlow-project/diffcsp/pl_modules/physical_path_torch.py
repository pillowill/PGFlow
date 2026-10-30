"""Torch optimizer for the periodic fixed-lattice stiffness action."""
import itertools
import torch


def torch_action(nodes, lattice, images, alpha, cutoff, sigma):
    k = nodes.shape[0] - 1
    mid = (nodes[:-1] + nodes[1:]) * .5
    velocity = (nodes[1:] - nodes[:-1]) @ lattice * k
    total = velocity.square().sum()
    pairs = torch.triu_indices(nodes.shape[1], nodes.shape[1], 1, device=nodes.device)
    if pairs.shape[1]:
        delta = mid[:, pairs[1]] - mid[:, pairs[0]]
        delta = delta - delta.round()
        r = (delta[:, :, None, :] + images[None, None, :, :]) @ lattice
        d2 = r.square().sum(-1).clamp_min(1e-12)
        w = torch.exp(-d2 / (2*sigma*sigma)) * (1-d2/(cutoff*cutoff)).clamp_min(0).square()
        dv = velocity[:, pairs[1]] - velocity[:, pairs[0]]
        dot = (r * dv[:, :, None, :]).sum(-1)
        total = total + alpha*(w*dot.square()/d2).sum()
    return total/(2*k)


def physical_path_torch(f0, displacement, lattices, num_atoms, times, *, segments=3, alpha=2., cutoff=4., sigma=2., maxiter=40, lr=.03, return_nodes=False):
    if segments<2 or alpha<0 or cutoff<=0 or sigma<=0 or maxiter<1 or lr<=0:
        raise ValueError('Invalid physical path settings')
    counts=num_atoms.detach().cpu().tolist()
    if sum(counts)!=len(f0): raise ValueError('num_atoms does not match coordinate count')
    positions=[]; velocities=[]; all_nodes=[]; offset=0
    # Lightning validation can enable inference_mode; local optimization needs autograd.
    with torch.inference_mode(False), torch.enable_grad():
        for g,count in enumerate(counts):
            start=f0[offset:offset+count].detach().clone()
            end=start+displacement[offset:offset+count].detach().clone()
            L=lattices[g].detach().clone()
            singular=torch.linalg.svdvals(L).min().item()
            if not 1e-6 < singular: raise ValueError('Physical path requires nonsingular lattice')
            radius=int(__import__('math').ceil(cutoff/singular+.5))
            if radius>12: raise ValueError('Physical path image bound too large')
            images=torch.tensor(list(itertools.product(range(-radius,radius+1),repeat=3)),device=start.device,dtype=start.dtype)
            grid=torch.linspace(0,1,segments+1,device=start.device,dtype=start.dtype)[:,None,None]
            line=start[None]*(1-grid)+end[None]*grid
            interior=torch.nn.Parameter(line[1:-1].clone())
            opt=torch.optim.Adam([interior],lr=lr)
            best=line.detach().clone()
            best_value=torch_action(best,L,images,alpha,cutoff,sigma).detach()
            for _ in range(maxiter):
                opt.zero_grad(set_to_none=True)
                nodes=torch.cat((start[None],interior,end[None]))
                value=torch_action(nodes,L,images,alpha,cutoff,sigma)
                better=torch.isfinite(value)&(value<best_value)
                best=torch.where(better,nodes.detach(),best)
                best_value=torch.where(better,value.detach(),best_value)
                value.backward(); opt.step()
            nodes=torch.cat((start[None],interior,end[None])).detach()
            value=torch_action(nodes,L,images,alpha,cutoff,sigma).detach()
            best=torch.where(torch.isfinite(value)&(value<best_value),nodes,best)
            all_nodes.append(best.detach())
            t=times[g].detach().clone()
            index=torch.clamp((t*segments).long(),0,segments-1)
            fraction=t*segments-index
            positions.append(best[index]*(1-fraction)+best[index+1]*fraction)
            velocities.append(segments*(best[index+1]-best[index]))
            offset+=count
    result=(torch.cat(positions).detach(),torch.cat(velocities).detach())
    return (*result, all_nodes) if return_nodes else result

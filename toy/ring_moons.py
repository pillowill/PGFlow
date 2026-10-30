"""Paired 8-ring and two-moons conditional path experiments."""
import json,time
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from toy4_control import features,mmd
from toy4_metric import sample
from toy4_ablation import paths,K,SIGMA

CENTERS=2*np.column_stack((np.cos(np.arange(8)*np.pi/4),np.sin(np.arange(8)*np.pi/4)))

def data(rng,n,dataset):
    x0=rng.normal(size=(n,2))
    if dataset=='ring':
        x1=CENTERS[rng.integers(0,8,n)]+.1*rng.normal(size=(n,2))
    else:
        angle=rng.uniform(0,np.pi,n); side=rng.integers(0,2,n)
        x=np.where(side==0,np.cos(angle),1-np.cos(angle))
        y=np.where(side==0,np.sin(angle),.5-np.sin(angle))
        x1=1.4*(np.column_stack((x,y))-np.array([.5,.25]))+.08*rng.normal(size=(n,2))
    return x0,x1

def distribution_metrics(x,y,dataset):
    d=cdist(x,y,'sqeuclidean'); i,j=linear_sum_assignment(d)
    r={'mmd_biased':mmd(x,y),'empirical_w2':float(np.sqrt(d[i,j].mean()))}
    if dataset=='ring':
        dist=cdist(x,CENTERS); idx=dist.argmin(axis=1); counts=np.bincount(idx,minlength=8)
        r.update({'mode_fractions':(counts/len(x)).tolist(),'mode_coverage_min_1pct':int((counts>=len(x)*.01).sum()),'outside_radius_0_3':float((dist.min(axis=1)>.3).mean())})
    else:
        # Approximate distance to the two noiseless arcs on a dense fixed grid.
        a=np.linspace(0,np.pi,500); upper=np.column_stack((np.cos(a),np.sin(a))); lower=np.column_stack((1-np.cos(a),.5-np.sin(a)))
        arcs=1.4*(np.vstack((upper,lower))-np.array([.5,.25])); d_arc=cdist(x,arcs)
        r.update({'outside_arc_0_2':float((d_arc.min(axis=1)>.2).mean()),'arc_side_fractions':[float((d_arc.argmin(axis=1)<500).mean()),float((d_arc.argmin(axis=1)>=500).mean())],'central_gap_fraction':float(((np.abs(x[:,0])<.25)&(np.abs(x[:,1])<.15)).mean())})
    return r

def experiment(dataset,seed,mode,save_plot):
    rng=np.random.default_rng(seed); n=250; nv=100; repeat=4
    a,b=data(rng,n,dataset); av,bv=data(rng,nv,dataset); start=time.perf_counter()
    actual='straight' if mode=='zero' else mode
    train,stat=paths(a,b,actual); val,sv=paths(av,bv,actual); elapsed=time.perf_counter()-start
    assert np.allclose(train[:,0],a) and np.allclose(train[:,-1],b)
    ts=rng.random(n*repeat); tv=rng.random(nv*repeat); w=rng.normal(size=(3,128)); phase=rng.uniform(0,2*np.pi,128)
    x,u=sample(np.repeat(train,repeat,axis=0),ts); phi=features(x,ts,w,phase)
    coef=np.linalg.solve(phi.T@phi+.1*np.eye(phi.shape[1]),phi.T@u)
    z,uv=sample(np.repeat(val,repeat,axis=0),tv); pred=features(z,tv,w,phase)@coef
    loss=float(np.mean((pred-uv)**2)); moment=float(np.mean(uv**2))
    source,target=data(rng,400,dataset); scores={}; generated50=None
    for steps in (10,50,100):
        gen=source.copy(); potentials=[]
        for i in range(steps):
            potentials.append(np.exp(-np.sum(gen*gen,axis=1)/(2*SIGMA**2)).mean())
            gen+=features(gen,np.full(len(gen),i/steps),w,phase)@coef/steps
        scores[str(steps)]=distribution_metrics(gen,target,dataset)
        scores[str(steps)]['trajectory_gaussian_potential']=float(np.mean(potentials))
        if steps==50: generated50=gen.copy()
    h=1e-3; eval_x=z[:200]; eval_t=tv[:200]; field=features(eval_x,eval_t,w,phase)@coef
    grads=[(features(eval_x+np.eye(2)[i]*h,eval_t,w,phase)@coef-features(eval_x-np.eye(2)[i]*h,eval_t,w,phase)@coef)/(2*h) for i in range(2)]
    result={'dataset':dataset,'seed':seed,'method':mode,'mean_training_path_objective':float(stat[:,0].mean()),'mean_straight_objective_same_metric':float(stat[:,1].mean()),'validation_mse':loss,'normalized_mse':loss/moment,'velocity_second_moment':moment,'mean_velocity_norm':float(np.linalg.norm(field,axis=1).mean()),'mean_squared_spatial_jacobian':float(np.mean(np.sum(np.stack(grads,axis=-1)**2,axis=(1,2)))),'path_seconds':elapsed,'sampling':scores}
    if save_plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axs=plt.subplots(1,3,figsize=(12,3.7),constrained_layout=True)
        axs[0].scatter(b[:,0],b[:,1],s=3,alpha=.4); axs[0].set_title('Training targets')
        for p in train[:30]: axs[1].plot(p[:,0],p[:,1],alpha=.4,linewidth=.6)
        axs[1].set_title('Conditional paths')
        axs[2].scatter(generated50[:,0],generated50[:,1],s=3,alpha=.5)
        gx,gy=np.meshgrid(np.linspace(-2.5,2.5,12),np.linspace(-2.5,2.5,12)); points=np.column_stack((gx.ravel(),gy.ravel())); v=features(points,np.full(len(points),.5),w,phase)@coef
        axs[2].quiver(points[:,0],points[:,1],v[:,0],v[:,1],alpha=.4,scale=35); axs[2].set_title('Samples and field, t=0.5')
        for ax in axs: ax.set_aspect('equal'); ax.set_xlim(-3,3); ax.set_ylim(-3,3)
        out=Path('results/figures');out.mkdir(parents=True,exist_ok=True);fig.savefig(out/f'{dataset}_{mode}_seed0.png',dpi=140);plt.close(fig)
    return result

if __name__=='__main__':
    import toy4_ablation as path_engine
    old_beta=path_engine.BETA; path_engine.BETA=0.
    check_a=np.array([[-1.,.2],[.3,-.4]]); check_b=np.array([[1.,-.1],[-.4,.8]])
    zero_nodes,_=path_engine.paths(check_a,check_b,'scalar')
    tt=np.linspace(0,1,K+1)[None,:,None]; expected=check_a[:,None,:]*(1-tt)+check_b[:,None,:]*tt
    assert np.allclose(zero_nodes,expected,atol=1e-6)
    path_engine.BETA=old_beta
    rows=[]
    for dataset in ('ring','moons'):
        for seed in range(5):
            for mode in ('straight','zero','scalar','anisotropic','repulsion'):
                row=experiment(dataset,seed,mode,seed==0 and mode!='zero'); rows.append(row)
                print(dataset,seed,mode,'loss',round(row['normalized_mse'],4),'mmd50',round(row['sampling']['50']['mmd_biased'],5),flush=True)
            pair=rows[-5:]; assert abs(pair[0]['normalized_mse']-pair[1]['normalized_mse'])<1e-12
            assert abs(pair[0]['sampling']['50']['mmd_biased']-pair[1]['sampling']['50']['mmd_biased'])<1e-12
    result={'config':{'source':'standard normal','pairing':'independent','train_pairs':250,'validation_pairs':100,'time_samples_per_pair':4,'features':128,'ridge':.1,'seeds':5,'ring_radius':2.,'ring_noise':.1,'moons_noise':.08,'metric':'fixed Gaussian center prior; anisotropy fixed x axis; not a learned normal field','note':'zero path explicitly uses straight route; finite-sample metrics and local optimization; elapsed includes validation path computation'},'results':rows}
    Path('results/ring_moons.json').write_text(json.dumps(result,indent=2),encoding='utf8')

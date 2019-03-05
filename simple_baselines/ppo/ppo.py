from .policy import build_policy
from .runer import Runner
from simple_baselines.utils import constfn,set_global_seeds

import time
import numpy as np
import os

def learn(network,env,total_timesteps,nsteps=2048,
          lr=1e-4,gamma=0.99,lam=0.95,
          max_grad_norm=0.5,cliprange=0.2,
          ent_coef=0.0,vf_coef=0.5,
          nminibatches=4,noptepochs=4,
          seed=None,
          model_fn=None,
          load_path=None,
          save_interval=0,
          log_interval=10,
          epsilon_start=0,
          epsilon_final=0.002,
          epsilon_decay=140,
          **netword_kargs):#value_network

    if isinstance(lr,float):lr=constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange,float):cliprange=constfn(cliprange)
    else:assert callable(cliprange)
    set_global_seeds(seed)

    total_timesteps=int(total_timesteps)
    nenvs=env.num_envs
    ob_space=env.observation_space
    ac_space=env.action_space
    nbatch=nenvs*nsteps
    nbatch_train=nbatch//nminibatches
    assert  nbatch%nminibatches==0

    if model_fn is None:
        from .model import Model
        model_fn=Model

    policy=build_policy(env,network,**netword_kargs)
    model=model_fn(policy=policy,ob_space=ob_space,ac_space=ac_space,nbatch_act=nenvs,nbatch_train=nbatch_train,nsteps=nsteps,
                   ent_coef=ent_coef,vf_coef=vf_coef,max_grad_norm=max_grad_norm)
    if load_path is not None:
        model.load(load_path)

    runner=Runner(env=env,model=model,nsteps=nsteps,gamma=gamma,lam=lam,epsilon_start=epsilon_start,epsilon_final=epsilon_final,epsilon_decay=epsilon_decay,)

    nupdates=total_timesteps//nbatch
    tfirststart = time.time()

    for update in range(1,nupdates+1):
        tstart=time.time()
        frac=1.0-(update-1.0)/nupdates
        lrnow=lr(frac)
        cliprangenow=cliprange(frac)
        obs,returns,masks,actions,values,neglogps,states=runner.run(update)


        mblossvals = []
        if states is None:
            inds=np.arange(nbatch)

            for j in range(noptepochs):
                np.random.shuffle(inds)
                for start in range(0,nbatch,nbatch_train):
                    end=start+nbatch_train
                    mbinds=inds[start:end]
                    slices=[attr[mbinds] for attr in (obs,returns,masks,actions,values,neglogps)]
                    mblossvals.append(model.train(lrnow,cliprangenow,*slices))

        else:
            assert nenvs%nminibatches==0
            envsperbatch=nenvs//nminibatches
            envinds=np.arange(nenvs)
            flatinds=np.arange(nenvs*nsteps).reshape(nenvs,nsteps)
            envsperbatch=nbatch_train//nsteps
            for _ in range(noptepochs):
                np.random.shuffle(envinds)
                for start in range(0,nenvs,envsperbatch):
                    end=start+envsperbatch
                    mbenvinds=envinds[start:end]
                    mbflatinds=flatinds[mbenvinds].ravel()
                    slices=(arr[mbflatinds] for arr in (obs,returns,masks,actions,values,neglogps))
                    mbstates=states[mbenvinds]
                    mblossvals.append(model.train(lrnow,cliprangenow,*slices,mbstates))
            
        lossvals=np.mean(mblossvals,axis=0)
        tnow=time.time()
        fps=int(nbatch/(tnow-tstart))

        if update%log_interval==0:
            print("*******status********")
            print("total_timesteps:",update*nbatch/10000)
            print("fps",fps)
            print("update",update)
            print("time_cost:",tnow-tfirststart)
            for (loss_name,loss_val) in zip(model.loss_name,lossvals):
                print("%s:"%loss_name,loss_val)
            print("********************")
        if save_interval and update%save_interval==0:
            save_path=os.path.join(os.getcwd(),"%s"%update)
            model.save(save_path)
            print('Saving to', save_path)

    return model



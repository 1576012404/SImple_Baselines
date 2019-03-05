import numpy as np
from abc import ABC, abstractmethod
import random
import math

class AbstractEnvRunner(ABC):
    def __init__(self, *, env, model, nsteps):
        self.env = env
        self.model = model
        self.nenv = nenv = env.num_envs if hasattr(env, 'num_envs') else 1
        self.batch_ob_shape = (nenv*nsteps,) + env.observation_space.shape
        self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
        self.obs[:] = env.reset()
        self.nsteps = nsteps
        self.states=model.initial_state
        self.dones = [False for _ in range(nenv)]

    @abstractmethod
    def run(self):
        raise NotImplementedError

class Runner(AbstractEnvRunner):
    def __init__(self,env,model,nsteps,gamma,lam,epsilon_start=0.9,epsilon_final=0.002,epsilon_decay=140,):
        super().__init__(env=env,model=model,nsteps=nsteps, )
        self.lam=lam
        self.gamma=gamma
        self.epsilon_start=epsilon_start
        self.epsilon_final=epsilon_final
        self.epsilon_decay=epsilon_decay

    def run(self,update):
        mb_obs,mb_actions,mb_rewards,mb_values,mb_dones,mb_neglogps=[],[],[],[],[],[]
        mb_states=self.states
        if self.epsilon_start>0:
            epsilon=self.epsilon_final+(self.epsilon_start-self.epsilon_final)*math.exp(-1*update/self.epsilon_decay)
        else:
            epsilon=0

        for i in range(self.nsteps):
            actions,values,self.states,neglogps=self.model.step(self.obs,S=self.states,M=self.dones)
            # if i==4:
            # 
            #     print("###################")
            #     print("values", values)
            #     print("actions", actions)
            #     print("done", self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogps.append(neglogps)
            mb_dones.append(self.dones)

            if epsilon and random.random()<epsilon:
                actions=[]
                unwapped=self.env.unwrapped
                for env in unwapped.envs:
                    actions.append(env.action_space.sample())

            self.obs[:],rewards,self.dones,infos=self.env.step(actions)#step
            mb_rewards.append(rewards)

        mb_obs=np.asarray(mb_obs,dtype=self.obs.dtype)
        mb_rewards=np.asarray(mb_rewards,dtype=np.float32)
        mb_actions=np.asarray(mb_actions)
        mb_values=np.asarray(mb_values,dtype=np.float32)
        mb_neglogps=np.asarray(mb_neglogps,dtype=np.float32)
        mb_dones=np.asarray(mb_dones,dtype=np.bool)
        last_values=self.model.value(self.obs,S=self.states,M=self.dones)

        mb_returns=np.zeros_like(mb_rewards)
        mb_advs=np.zeros_like(mb_rewards)
        lastgaelam=0

        for t in reversed(range(self.nsteps)):
            if t==self.nsteps-1:
                next_none_terminal=1.0-self.dones
                next_values=last_values
            else:
                next_none_terminal=1.0-mb_dones[t+1]
                next_values=mb_values[t+1]
            # print("delta0",mb_rewards[t].shape,next_values.shape,next_none_terminal.shape)
            delta=mb_rewards[t]+self.gamma*next_values*next_none_terminal-mb_values[t]
            # print("delta",delta.shape,next_none_terminal.shape,lastgaelam)
            mb_advs[t]=lastgaelam=delta+self.gamma*self.lam*next_none_terminal*lastgaelam
        mb_returns=mb_advs+mb_values
        return (*map (flatten,(mb_obs,mb_returns,mb_dones,mb_actions,mb_values,mb_neglogps)),mb_states)







def flatten(attr):
    s=attr.shape
    return attr.swapaxes(0,1).reshape(s[0]*s[1],*s[2:])








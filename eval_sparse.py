import gym
import numpy as np
from simple_baselines.ppo import ppo

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
class MyReward(gym.Wrapper):
    def __init__(self, env):
        super(MyReward, self).__init__(env)
        self.m_RwardList = []
        self.m_step=0
        self.m_count = 0

    def step(self, action):
        self.m_step+=1
        obs, reward, done, info = self.env.step(action)
        reward=0
        iMaxStep=self.unwrapped.spec.max_episode_steps
        if done and self.m_step<iMaxStep:
            reward=10
        self.m_count += 1
        if self.m_count%60000==0:
            print("frame",self.m_count)
        self.m_RwardList.append(reward)
        if done:
            iMeanReward = np.sum(self.m_RwardList)
            print("mean_reward", iMeanReward)
            self.m_RwardList = []
        return obs, reward, done, info

    def reset(self, **kwargs):
        self.m_step=0
        return super(MyReward,self).reset(**kwargs)


def EnvFunc(iSeed):
    def _EnvFunc_():
        oEnv = gym.make("MountainCar-v0")
        oEnv.seed(iSeed)
        oEnv = MyReward(oEnv)
        return oEnv
    return _EnvFunc_


def Test():
    env=DummyVecEnv([EnvFunc(i) for i in range(1)])
    env = VecNormalize(env,ob=True, ret=True)
    act=ppo.learn(
        network="mlp",
        env=env,
        # lr=3e-4,
        nsteps=256,
        nminibatches=8,
        # lam=0.94,
        total_timesteps=0,
        log_interval=100,
        epsilon_start=0.9,
        epsilon_final=0.002,
        epsilon_decay=140,
        # save_interval=500,
         load_path="/home/duoyi/MyGit/simple_baselines/300",
        num_layers=3,
        num_hidden=256,
        value_network="copy"
    )
    obs=env.reset()
    iFrame=0

    while True:
        action,_,_,_=act.step(obs)
        obs,reward,done,info=env.step(action)
        iFrame+=1
        if done[0]:
            print("total Frame",iFrame)
            iFrame=0
        env.render()




if __name__=="__main__":
    Test()
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
        self.m_Print=0

    def step(self, action):
        self.m_step+=1
        obs, reward, done, info = self.env.step(action)
        reward=0
        iMaxStep=self.unwrapped.spec.max_episode_steps
        if done and self.m_step<iMaxStep:
            reward=10
        self.m_count += 1
        self.m_RwardList.append(reward)
        if done:
            iMeanReward = np.sum(self.m_RwardList)
            if self.m_count>self.m_Print:
                print("mean_reward", iMeanReward)
                self.m_Print+=3000
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


def Train():
    env=DummyVecEnv([EnvFunc(i) for i in range(8)])
    env = VecNormalize(env,ob=True, ret=True)
    act=ppo.learn(
        network="mlp",
        env=env,
        # lr=3e-4,
        nsteps=256,
        nminibatches=8,
        # lam=0.94,
        total_timesteps=6000000,
        log_interval=100,
        epsilon_start=0.9,
        epsilon_final=0.002,
        epsilon_decay=140,
        save_interval=100,
        num_layers=3,
        num_hidden=256,
        value_network="copy"
    )
    act.save("ppo2_model/model")



if __name__=="__main__":
    Train()
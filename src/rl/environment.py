import torch
import pandas as pd
import numpy as np

from datetime import datetime, timedelta

from slurm_load.workload import WorkLoad

from app.utils import vectorize_job

date_format = "%Y-%m-%dT%H:%M:%S"

class SlurmEnv(object):
    def __init__(self):
        self.location = '/una/directory/a/caso'
        pass
    
    def step(action:torch.Tensor):
        observation = None
        reward = 0.
        terminated = True
        
        #inserirre l'avvio del docker

        return observation, reward, terminated, False
    
    def reward_calculation(self, loc:str='C:\\Users\\david\\Uni\\Tesi\\slurm_ai_sched\\src\\rl\\smt.csv') -> float:
        executed_time = 0
        wait_time = 0

        data = pd.read_csv(loc, sep='|', header=0)
        eligible = data['Eligible'].apply(lambda x : datetime.strptime(x, date_format))
        start = data['Start'].apply(lambda x : datetime.strptime(x, date_format))
        wait_time = start - eligible
        wait_time = wait_time.apply(lambda x : x.total_seconds()).mean()

        return wait_time

class DummyEnv(object):
    def __init__(self, workload:WorkLoad, job_num:int = 25):
        self.workload = workload
        self.job_num = job_num
        self.reset()

    def step(self, action:torch.Tensor) -> torch.Tensor:
        actor_value = action / 2
        real_value = self.obs[:,2] / 2
        reward = torch.sqrt((actor_value - real_value)**2).sum()
        self.reset()
        return -reward / (real_value.sum())


    def observation(self) -> torch.Tensor:
        return self.obs
    
    def reset(self):
        self.jobs = self.workload.generate_workload(self.job_num)
        self.obs = vectorize_job(self.jobs).float()

if __name__ == '__main__':
    env = SlurmEnv()
    wait_time = env.reward_calculation()
    print(wait_time)
import torch
import pandas as pd
import numpy as np
import docker
import os
import shutil

from pathlib import Path

from datetime import datetime, timedelta

from slurm_load.workload import WorkLoad

from app.utils import vectorize_job

from app.slurm_topology import TopologyApp

from app.env_gen import SlurmEnvGen

from slurm_topo.topology import TopologyPrinter

from rl.docker_utils import DockerSched

from app.utils import read_account, node_extract, topology_extract
from slurm_load.utils import read_users_sim
from slurm_load.job import JobGenerator

date_format = "%Y-%m-%dT%H:%M:%S"
dir_name = "slurm_env_dir"

def print_workload(path, jobs):
    with open(path / 'first_job.events', 'w') as f:
        for td, job in jobs:
            f.write(f"-dt {td} -e submit_batch_job | {job}\n")

class SlurmEnv(object):
    def __init__(self, save_path:str|Path, num_env=1):
        self.save_path = save_path if isinstance(save_path, Path) else Path(save_path)
        self.num_env = num_env
        self.docker_sched = []
        self.sl_env = SlurmEnvGen()
        self.batch = 10
        

        if not self.save_path.is_dir():
            raise ValueError("arg save_path is not an existing directory")

    def dir_setup(self, force=False):
        ori_loc = f"{dir_name}0"
        
        if not (self.save_path / ori_loc).exists():
            os.mkdir(self.save_path / ori_loc)    
            self.slurm_setup()
        else:
            p = self.save_path / ori_loc
            users = read_users_sim(p / 'etc/users.sim')
            accounts = read_account(p / 'etc/sacctmgr.script')
            nodes = node_extract(p / 'etc/slurm.conf')
            topo = topology_extract(p / 'etc/topology.conf', nodes)
            job_gen = JobGenerator(topology=topo)
            self.workload_gen = WorkLoad(users[1:], accounts, job_gen)
            self.obs = self.workload_gen.generate_workload(self.batch)
            print_workload(p / 'workload', self.obs)
        
        self.docker_sched.append(DockerSched("slurm_env_0", self.save_path / f"{dir_name}0", "/home/slurm/mount"))

        for i in range(1, self.num_env):
            loc = f"{dir_name}{i}"
            if (self.save_path / loc).exists():
                shutil.rmtree(self.save_path / loc)
            shutil.copytree(self.save_path / ori_loc, self.save_path / loc, dirs_exist_ok=True)

    def slurm_setup(self):
        topology_app = TopologyApp(self.save_path / f"{dir_name}0", Path("/home/slurm/mount"))
        topology_app.print_topology(self.sl_env.topology)
        topology_app.print_acc_manager(self.sl_env.accounts)
        topology_app.print_gres(self.sl_env.topology.nodes)
        topology_app.print_slurm_conf(self.sl_env.topology.nodes)
        topology_app.print_slurmDB()
        topology_app.print_users_sim(self.sl_env.users)
        topology_app.print_sim_conf()
        topology_app.copy_cert()
        topology_app.print_workload(self.sl_env.generate_workload(self.batch))
        self.workload_gen = self.sl_env.workload

    
    def step(self, action:torch.Tensor=None):
        rewards = []
        for i in range(self.num_env):
            exit_code, log = self.docker_sched[i].execute()
            print(exit_code)
            rew = self.reward_calculation(self.save_path /  f"{dir_name}{i}/results/slurm_acct.out")
            rewards.append(rew)
            self.obs = self.workload_gen.generate_workload(self.batch)
            print_workload(self.save_path /  f"{dir_name}{i}/workload", self.obs)
        return rewards
            
    
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
    #env = SlurmEnv()
    #wait_time = env.reward_calculation()
    #print(wait_time)

    env = SlurmEnv("/home/ago/tesi/slurm_ai_sched/src/tests/env")
    env.dir_setup()
    for i in range(10):
        rewards = env.step()
        print(rewards)
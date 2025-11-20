from typing import Any
from slurm_topo.topology import Topology
import numpy.random as rnd
from datetime import time, timedelta

class Job(object):
    '''
    Class Job: class for organizing all jobs in
    '''

    def __init__(self, job_id:str, sim_walltime:int, uid:str, flags:list[str]):
        self.job_id = job_id
        self.sim_walltime = sim_walltime
        self.uid = uid
        self.flags = flags

    def __str__(self):
        base_job_str = f"-J {self.job_id} -sim-walltime {self.sim_walltime} --uid={self.uid}"
        for option in self.flags:
            base_job_str = base_job_str + ' ' + option
        return base_job_str + ' pseudo.job'

class JobGenerator:
    '''
    Class JobGenerator: a random job generator
    '''

    def __init__(self, topology:Topology, max_long_job=24, max_short_job=60, min_mem = 1000, **kwargs):
        self.max_long_job = max_long_job
        self.max_short_job = max_short_job
        self.topology = topology
        self.min_mem = min_mem

    def set_seed(seed:int):
        rnd.seed(seed)

    def generate_time(self, minute_b:bool=False, hour_b=False) -> time:
        hour = 0 if not hour_b or self.max_long_job <= 1 else rnd.randint(0, 24)
        if minute_b:
            minute = 1 if self.max_short_job <= 1 else rnd.randint(1, self.max_short_job)
        else:
            minute = rnd.choice([0, 30]) if hour != 0 else rnd.randint(1, 60)

        return time(
            hour=hour,
            minute=minute,
        )
    
    def generate_job_time(self, long:bool=False):
        req_time = self.generate_time(minute_b=not long, hour_b=long)
        req_time_s = timedelta(hours=req_time.hour, minutes=req_time.minute).total_seconds()
        delta_time = rnd.randint(-1, min(req_time_s, 3600))
        sim_walltime = -1 if delta_time == -1 or delta_time == 0 else int(req_time_s - delta_time)
        return req_time, sim_walltime

    def generate_job_tasks(self, max_task, args={}):
        n_task_nodes = rnd.randint(1, max_task + 1)
        num_nodes = self.topology.count_nodes(procs=n_task_nodes, **args)
        if num_nodes == 0 or num_nodes == 1:
            return 0, 0
        n_tasks = rnd.randint(1, num_nodes) * n_task_nodes
        return n_task_nodes, n_tasks
    
    def stringfy_flags(self, req_time:time, n_tasks, tasks_nodes, account=None, constr=None, mem=0, gres=0) -> list[str]:
        flags = []
        req_time = req_time.strftime('%H:%M:%S')
        flags.append(f'-t {req_time}')
        flags.append(f'-n {n_tasks}')
        flags.append(f'--ntasks-per-node={tasks_nodes}')
        if account is not None:
            flags.append(f'-A {account}')
        flags.append(f'-p normal')
        flags.append(f'-q normal')
        if constr is not None:
            flags.append(f'--constraint={constr}')
        if mem > 0:
            flags.append(f'--mem={mem}')
        if gres > 0:
            flags.append(f'--gres=gpu:{gres}')
        
        return flags
    
    def generate_classic_job(self, job_id:str, user_id:str, account:str, long:bool=True, feat='DEFAULT'):
        req_time, sim_walltime = self.generate_job_time(long)
        n_task_nodes, n_tasks = self.generate_job_tasks(self.topology.max_tasks_node[feat])
        if n_task_nodes == 0:
            return None
        flags = self.stringfy_flags(req_time, n_tasks, n_task_nodes, account)
        return Job(job_id, sim_walltime, user_id, flags)
    
    def generate_gres(self, feat):
        if self.topology.max_gres[feat] == 0:
            return rnd.randint(1, self.topology.max_gres['ALL'] + 1) if self.topology.max_gres['ALL'] != 0 else 0
        else:
            return rnd.randint(1, self.topology.max_gres[feat] + 1)

    def generate_gpu_job(self, job_id:str, user_id:str, account:str, long:bool=True, feat='DEFAULT'):
        req_time, sim_walltime = self.generate_job_time(long)
        gres = self.generate_gres(feat)
        n_task_nodes, n_tasks = self.generate_job_tasks(self.topology.max_tasks_node[feat], {'gres':gres})
        if n_task_nodes == 0:
            return None
        num_nodes = n_tasks // n_task_nodes
        if num_nodes < self.topology.count_nodes(procs=n_task_nodes, gres=gres):
            return None
        flags = self.stringfy_flags(req_time, n_tasks, n_task_nodes, account, gres=gres)
        return Job(job_id, sim_walltime, user_id, flags)
    
    def generate_mem(self, feat):
        mem = rnd.randint(self.min_mem, int(self.topology.max_mem[feat] * 4 / 5))
        mem -= mem % 1000
        return mem

    def generate_generic_job(self, job_id:str, user_id:str, account:str, long:bool=False, feat='DEFAULT'):
        req_time, sim_walltime = self.generate_job_time(long)
        param = {}
        if rnd.choice([True, False]):
            gres = self.generate_gres(feat)
            param['gres'] = gres
        if rnd.choice([True, False]):
            mem = self.generate_mem(feat)
            param['mem'] = mem
        n_task_nodes, n_tasks = self.generate_job_tasks(self.topology.max_tasks_node[feat], param)
        if n_task_nodes == 0:
            return None
        flags = self.stringfy_flags(req_time, n_tasks, n_task_nodes, account, **param)
        return Job(job_id, sim_walltime, user_id, flags)

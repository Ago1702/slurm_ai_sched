import numpy.random as rnd

from slurm_load.job import Job, JobGenerator
from slurm_load.user import User
from slurm_load.utils import read_users_sim, print_users_sim



class WorkLoad(object):
    '''
    Class WorkLoad: Simple class for generating workload
    '''

    def __init__(self, users:list[User], accounts:dict, job_gen:JobGenerator, 
                 dt_range:int|list[int] = [1800, 6000], probability:list = [0.7, 0.9], retry = 5, **kwargs):
        self.users = users
        self.accounts = accounts
        self.job_gen = job_gen

        self.ts = 0
        self.jb_id = 1
        if isinstance(dt_range, int):
            self.dt_range = [1, dt_range]
        else:
            self.dt_range = dt_range
        self.probability = probability
        self.retry = retry
        
    
    def set_seed(self, seed:int):
        rnd.seed(seed)
    
    def reset(self):
        '''
        Reset generator internal state
        '''
        self.ts = 0
        self.jb_id = 1

    def generate_job(self) -> tuple[int, Job]:
        td = self.ts
        user:User = rnd.choice(self.users)
        user_id = user.usr
        account = self.accounts[user_id]
        job_id = f"jobid_{self.jb_id + 1000}"
        val = rnd.rand()
        if val < self.probability[0]:
            gen = self.job_gen.generate_classic_job
        elif val < self.probability[1]:
            gen = self.job_gen.generate_gpu_job
        else:
            gen = self.job_gen.generate_generic_job
        
        for i in range(self.retry):
            job = gen(job_id, user_id, account)
            if job is not None:
                break
        else:
            while job is None:
                job = self.job_gen.generate_classic_job(job_id, user_id, account)
        
        self.ts += rnd.randint(low=self.dt_range[0], high=self.dt_range[1])
        self.jb_id += 1
        return td, job
    
    def generate_workload(self, num:int, reset:bool=True) -> list[tuple]:
        """
        Generate a workload with @num jobs

        Args:
            num (int): Jobs number
            reset (bool, optional): Reset the internal state of the machine at the beginning. Defaults to True.

        Returns:
            list[tuple]: a list of couple: submission time (sec.) and the Job submitted
        """
        if reset: self.reset()
        workload = []
        for i in range(num):
            td, job = self.generate_job()
            workload.append((td, job))
        return workload


    def reset(self, seed:int=None):
        self.ts = 0
        self.jb_id = 1
        if seed is not None:
            self.set_seed(seed)



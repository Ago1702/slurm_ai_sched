from typing import Any

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
            base_job_str = base_job_str + option + ' '
        return base_job_str + 'pseudo.job'
    

if __name__ == '__main__':
    flags = [
        '-t 00:01:00',
        '-n 12',
        '--ntasks-per-node=12',
        '-A account2',
        '-p normal',
        '-q normal'
    ]

    job = Job('jobid_1001', '0', 'user5', flags=flags)
    print(job)
        
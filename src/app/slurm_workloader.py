from slurm_load.job import Job, JobGenerator
from slurm_load.user import User
from slurm_load.utils import read_users_sim
from slurm_load.workload import WorkLoad

from app.utils import read_account, node_extract, topology_extract

from pathlib import Path

import sys

if __name__ == '__main__':
    p = 'C:/Users/david/Uni/Tesi/slurm_ai_sched/src/tests/topology_gen'
    p = Path(p)

    workload_dir = p / 'workload'

    users = read_users_sim(p / 'etc/users.sim')
    accounts = read_account(p / 'etc/sacctmgr.script')
    nodes = node_extract(p / 'etc/slurm.conf')
    topo = topology_extract(p / 'etc/topology.conf', nodes)

    job_gen = JobGenerator(topology=topo)
    workload_gen = WorkLoad(users[1:], accounts, job_gen)
    jobs = workload_gen.generate_workload(50)

    with open(workload_dir / 'first_job.events', 'w') as f:
        for td, job in jobs:
            f.write(f"-dt {td} -e submit_batch_job | {job}\n")

    sys.exit(0)

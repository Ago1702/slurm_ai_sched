from slurm_topo.topology import TopologyGenerator, Topology, TopologyPrinter
from slurm_topo.node import NodeGenerator, Node
from slurm_load.user import UserGenerator, User
from slurm_load.utils import print_users_sim
from slurm_load.job import JobGenerator
from slurm_load.workload import WorkLoad

from pathlib import Path

from app.utils import eprint, load_template

import numpy.random as rnd

import shutil
import sys
import os
import step
import argparse

GRES_TEMP = load_template('gres.conf.step')
SL_CONF_TEMP =load_template('slurm.conf.step')
ACC_MNG_TEMP = load_template('sacctmgr.script.step')
SL_DB_TMP = load_template('slurmdbd.conf.step')
SIM_CONF = load_template('sim.conf.step')

def print_gres(filename, nodes:list[Node]):
    gres_nodes = [node for node in nodes if node.gres != 0]
    with open(filename, 'wb') as f:
        GRES_TEMP.stream(f, nodes=gres_nodes,)

def print_slurm_conf(filename, nodes:list[Node], path='lol', name='micro'):
    with open(filename, 'wb') as f:
        SL_CONF_TEMP.stream(f, nodes=nodes, path=path, name=name)

def print_acc_manager(filename, users:dict, max_job=1000):
    with open(filename, 'wb') as f:
        accounts = set(users.values())
        ACC_MNG_TEMP.stream(f, accounts=accounts, users=users, max_job=max_job)

def print_slurmDB(filename, path):
    with open(filename, 'wb') as f:
        SL_DB_TMP.stream(f, path=path)

def print_sim_conf(filename, path):
    with open(filename, 'wb') as f:
        SIM_CONF.stream(f, path=path)




if __name__ == '__main__':
    p = 'C:/Users/david/Uni/Tesi/slurm_ai_sched/src/tests/topology_gen'
    p = Path(p)

    etc_dir = p / 'etc'
    etc_dir.mkdir(exist_ok=True)

    workload_dir = p / 'workload'
    workload_dir.mkdir(exist_ok=True)

    

    node_gen = NodeGenerator()
    topo_gen = TopologyGenerator(2, 12, node_gen)
    topo = topo_gen.generate_topology(2)
    printer = TopologyPrinter()

    printer.print_topology(f"{etc_dir.absolute()}/topology.conf", topo.topo)
    print_gres(f"{etc_dir.absolute()}/gres.conf", topo.nodes)
    print_slurm_conf(f"{etc_dir.absolute()}/slurm.conf", topo.nodes, path=p.as_posix())

    user_gen = UserGenerator(6, 7, 3, 4)
    users, num_group = user_gen.generate_users()
    users = list(users)
    users.sort()
    print_users_sim(f"{etc_dir.absolute()}/users.sim", users)

    account_n = rnd.randint(1, len(users) + 1)
    accounts = {}
    for i, user in enumerate(users[1:]):
        accounts[user.usr] = f"account{rnd.randint(1, account_n + 1)}"
    
    print_acc_manager(f"{etc_dir.absolute()}/sacctmgr.script", accounts)
    print_slurmDB(f"{etc_dir.absolute()}/slurmdbd.conf", path=p.as_posix())
    print_sim_conf(f"{etc_dir.absolute()}/sim.conf", path=p.as_posix())
    shutil.copy('templates/slurm.cert', (etc_dir / 'slurm.cert').as_posix())
    shutil.copy('templates/slurm.key', (etc_dir / 'slurm.key').as_posix())
    job_gen = JobGenerator(topology=topo)
    workload_gen = WorkLoad(users[1:], accounts, job_gen)
    jobs = workload_gen.generate_workload(30)
    with open(workload_dir / 'first_job.events', 'w') as f:
        for td, job in jobs:
            f.write(f"-dt {td} -e submit_batch_job | {job}\n")
            td += rnd.randint(1, 30)

    sys.exit(0)
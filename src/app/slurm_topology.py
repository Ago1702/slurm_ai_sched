from slurm_topo.topology import TopologyGenerator, Topology
from slurm_topo.node import NodeGenerator, Node
from slurm_load.user import UserGenerator, User

from app.utils import eprint, load_template

import sys
import step
import argparse

GRES_TEMP = load_template('gres.conf.step')
SL_CONF_TEMP =load_template('slurm.conf.step')
ACC_MNG_TEMP = load_template('sacctmgr.script.step')

def print_gres(filename, nodes:list[Node]):
    gres_nodes = [node for node in nodes if node.gres != 0]
    with open(filename, 'wb') as f:
        GRES_TEMP.stream(f, nodes=gres_nodes,)

def print_slurm_conf(filename, nodes:list[Node]):
    with open(filename, 'wb') as f:
        SL_CONF_TEMP.stream(f, nodes=nodes,)

def print_acc_manager(filename, users:dict, max_job=1000):
    with open(filename, 'wb') as f:
        accounts = set(users.values())
        ACC_MNG_TEMP.stream(f, accounts=accounts, users=users, max_job=max_job)




if __name__ == '__main__':

    sys.exit(0)
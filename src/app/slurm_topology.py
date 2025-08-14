from slurm_topo.topology import TopologyGenerator, Topology
from slurm_topo.node import NodeGenerator, Node
from slurm_load.user import UserGenerator, User

from app.utils import eprint, load_template

import sys
import step
import argparse

GRES_TEMP = load_template('gres.conf.step')

def print_gres(filename, nodes:list[Node]):
    gres_nodes = [node for node in nodes if node.gres != 0]
    with open(filename, 'wb') as f:
        GRES_TEMP.stream(f, nodes=gres_nodes,)



if __name__ == '__main__':

    sys.exit(0)
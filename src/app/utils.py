import sys
import step
import re

from slurm_load.utils import print_users_sim

from slurm_topo.node import Node, read_node
from slurm_topo.topology import Topology, read_topology

TEMPLATE_LOC = "./templates/"

def load_template(filename:str) -> step.Template:
    try:
        with open(file=TEMPLATE_LOC + filename, mode='r') as f:
            return step.Template(f.read())
    except Exception as e:
        eprint(f'File {filename} not found')
        sys.exit(1)

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def read_account(p) -> dict[str, str]:
    rx = re.compile(r'^add user name=(.*) DefaultAccount=(.*) MaxSubmitJobs=\d+', flags=re.MULTILINE)
    with open(p, mode='r') as f:
        text = f.read()
        accounts = rx.findall(text)
        accounts = {account[0]: account[1] for account in accounts}
        accounts.pop('admin')
        return accounts

def node_extract(p) -> list[Node]|None:
    rx = re.compile(r'^NodeName=.* (.*=.*)*')
    with open(p, mode='r') as f:
        lines = f.readlines()
        lines = [line.rstrip('\n') for line in lines if rx.match(line)]
    if len(lines) == 0:
        return None
    return read_node(lines)

def topology_extract(p, nodes) -> Topology:
    rx = re.compile(r'^SwitchName=.* (.*=.*)*')
    with open(p, mode='r') as f:
        lines = f.readlines()
        lines = [line.rstrip('\n') for line in lines if rx.match(line)]
    if len(lines) == 0:
        return None
    return Topology(read_topology(nodes, lines))
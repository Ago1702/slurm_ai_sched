import sys
import step
import re
import torch

from datetime import timedelta

from slurm_load.utils import print_users_sim

from slurm_topo.node import Node, read_node
from slurm_topo.topology import Topology, read_topology
from slurm_load.job import Job

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

# Questo è inguardabile, è da sistemare
def extract_flags_value(flags:list[str]):
    values = [0] * 5
    for flag in flags:
        if flag.startswith('--'):
            name, value = flag.split('=')
        else:
            name, value = flag.split(' ')
        if name == '-n':
            values[0] = int(value)
        if name == '--ntasks-per-node':
            values[1] = int(value)
        if name == '-t':
            value = value.split(':')
            values[2] = int(timedelta(hours=int(value[0]), minutes=int(value[1]), seconds=int(value[2])).total_seconds())
        if name == '--mem':
            values[3] = int(value)
        if name == '--gres':
            values[4] = int(value.split(':')[1])
    return values 

def vectorize_job(jobs:list[tuple[int, Job]]) -> torch.Tensor:
    jobs_data = []
    for td, job in jobs:
        flags = job.flags
        values = extract_flags_value(flags)
        jobs_data.append(values)
    t_job = torch.tensor(jobs_data)
    return t_job

        
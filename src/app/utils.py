import sys
import step

from slurm_load.utils import print_users_sim

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
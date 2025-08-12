import numpy.random as rnd
import re

FEATURES = ['BigMem', 'ManyCores']

class Node(object):
    '''
    Class Node: Virtual node representation
    '''

    def __init__(self,name:str, procs:int, sockets:int, num:int|tuple[int]=0, memory:int=0, thread_core:int=1, gres:int=0, features:list[str]=[]):
        self.name = name
        self.num = num
        self.memory = memory
        self.procs = procs
        self.sockets = sockets
        self.thread_core = thread_core
        self.gres = gres
        self.features = features
        if isinstance(num, list) and num[0] == 1:
            self.num = num[1]
        if isinstance(num, list):
            cum_number = num[1] - num[0] + 1
        else:
            cum_number = num
        self.cum_number = cum_number

    def node_name(self) -> str:
        if isinstance(self.num, tuple):
            slurm_format += f"{self.name}[{self.num[0]}-{self.num[1]}]" if self.num[0] != self.num[1] else f"{self.name}{self.num[0]}"
        elif self.num == 0:
            slurm_format += f"{self.name}"
        elif self.num == 1:
            slurm_format += f"{self.name}1"
        else:
            slurm_format += f"{self.name}[1-{self.num}]"
        return slurm_format

    def slurm_formatting(self):
        slurm_format = f'NodeName='
        if isinstance(self.num, tuple):
            slurm_format += f"{self.name}[{self.num[0]}-{self.num[1]}]" if self.num[0] != self.num[1] else f"{self.name}{self.num[0]}"
        elif self.num == 0:
            slurm_format += f"{self.name}"
        elif self.num == 1:
            slurm_format += f"{self.name}1"
        else:
            slurm_format += f"{self.name}[1-{self.num}]"
        slurm_format += ' '
        
        if self.memory != 0:
            slurm_format += f'RealMemory={self.memory} '
        
        slurm_format += f"Procs={self.procs} "
        slurm_format += f"Sockets={self.sockets} "
        slurm_format += f"CoresPerSocket={self.procs//self.sockets} "
        slurm_format += f"ThreadsPerCore={self.thread_core} "

        if self.gres > 0:
            slurm_format += f"Gres=gpu:{self.gres}"
        if len(self.features) == 0:
            return slurm_format
        slurm_format += ' Feature='
        feat = ','.join(self.features)
        slurm_format += feat
        return slurm_format

class NodeGenerator(object):
    '''
    Class NodeGenerator: Random Computing Node generator
    '''
    def __init__(self, min_proc:int=0, max_proc:int=7, min_mem:int=6, max_mem:int=33, min_sock:int=1, max_sock:int=3, min_gres:int=1, max_gres:int=9):
        self.min_proc = min_proc
        self.max_proc = max_proc
        self.min_mem = min_mem
        self.max_mem = max_mem
        self.min_sock = min_sock
        self.max_sock = max_sock
        self.min_gres = min_gres
        self.max_gres = max_gres

        self.many_min_proc = max_proc * 2
        self.many_max_proc = max_proc * 8
        self.big_min_mem = max_mem * 4
        self.big_max_mem = max_mem * 16


    def set_seed(self, seed:int):
        rnd.seed(seed)

    def generate_sockets(self, bot, top, num):
        ret = rnd.randint(bot, top)
        ret -= (num % ret)
        return ret

    def generate_node(self, name:str, num:int|tuple[int], features:list=[], big_mem:bool=False, gpu:bool=False, many_cores:bool=False):
        if many_cores:
            procs = rnd.randint(self.many_min_proc, self.many_max_proc)
            features.append('ManyCores')
        else:
            procs = rnd.randint(self.min_proc, self.max_proc)
        procs = 1 if procs == 0 else procs * 4
        sockets = self.generate_sockets(self.min_sock, self.max_sock, procs)
        if big_mem:
            memory = rnd.randint(self.big_min_mem, self.big_max_mem)
            features.append('BigMem')
        else:
            memory = rnd.randint(self.min_mem, self.max_mem)
        memory *= 4000
        if not gpu:
            return Node(name, procs, sockets, num, memory, features=features)
        gres = rnd.randint(self.min_gres, self.max_gres)
        return Node(name, procs, sockets, num, memory, gres=gres, features=features)
    
def node_parser(line:str) -> dict:
    props = line.split()
    args = {}
    for prop in props:
        prop, val = prop.split('=')
        match prop:
            case 'NodeName':
                rx = re.compile(r'\d+')
                num = rx.findall(val)
                for i, _ in enumerate(num):
                    num[i] = int(num[i])
                if len(num) == 1:
                    args['num'] = num[0]
                elif len(num) != 0:
                    args['num'] = num
                args['name'] = re.sub(r'[^a-zA-Z]','',val)

            case 'RealMemory':
                args['memory'] = int(val)

            case 'Procs':
                args['procs'] = int(val)

            case 'Sockets':
                args['sockets'] = int(val)

            case 'ThreadsPerCore':
                args['thread_core'] = int(val)

            case 'Feature':
                args['features'] = val.split(',')
            
            case 'Gres':
                rx = re.compile(r'gpu:\d+')
                num = rx.findall(val)
                if len(num) == 1:
                    args['gres'] = int(re.sub(r'[^0-9]', '', num[0]))
    return args

def node_reader(lines:list[str]) -> list[Node]:
    node_dicts = []
    default = {}
    for line in lines:
        node_dict = node_parser(line)

        if node_dict['name'] == 'DEFAULT':
            default = node_dict
        else:
            node_dicts.append(node_dict)
    
    nodes = []
    for node_dict in node_dicts:
        for k in set(default.keys()).difference(node_dict.keys()):
            node_dict[k] = default[k]
        nodes.append(Node(**node_dict))
    return nodes
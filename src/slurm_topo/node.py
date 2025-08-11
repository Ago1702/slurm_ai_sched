import numpy.random as rnd

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
        
        if isinstance(num, list):
            cum_number = num[1] - num[0]
        else:
            cum_number = num
        self.cum_number = cum_number

    def slurm_formatting(self):
        slurm_format = f'NodeName='
        if isinstance(self.num, tuple):
            slurm_format += f"{self.name}[{self.num[0]}-{self.num[1]}]"
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

class Topology:

    def __init__(self, topology_list:list[Node|list]):
        self.topo = topology_list
        self.nodes:list[Node] = self.concatenate_list(topology_list)
        num_tasks = {'DEFAULT': 0}
        max_tasks_node = {'DEFAULT': 0}
        max_mem = {'DEFAULT': 0}
        max_gres = {'DEFAULT': 0}
        num_nodes = 0
        self.features = set(FEATURES)
        for k in self.features:
            num_tasks[k] = 0
            max_tasks_node[k] = 0
            max_mem[k] = 0
            max_gres[k] = 0
        for node in self.nodes:
            num_nodes += node.cum_number
            if len(node.features)==1:
                num_tasks['DEFAULT'] += node.procs
                max_tasks_node['DEFAULT'] = max(max_tasks_node['DEFAULT'], node.procs)
                max_mem['DEFAULT'] = max(max_mem['DEFAULT'], node.memory)
                max_gres['DEFAULT'] = max(max_gres['DEFAULT'], node.gres)
            else:
                node_feat = set(node.features)
                for feat in node_feat.intersection(self.features):
                    num_tasks[feat] += node.procs
                    max_tasks_node[feat] = max(max_tasks_node[feat], node.procs)
                    max_mem[feat] = max(max_mem[feat], node.memory)
                    max_gres[feat] = max(max_gres[feat], node.gres)
        self.num_nodes = num_nodes
        self.num_tasks = num_tasks
        self.max_tasks_node = max_tasks_node
        self.max_mem = max_mem
        self.max_gres = max_gres
        self.partitions = []
        self.qos = []

    def concatenate_list(self, topo_list:list):
        res = []
        for e in topo_list:
            if isinstance(e, list):
                res.extend(self.concatenate_list(e))
            else:
                res.append(e)
        return res
    
    def count_nodes(self, gres=0, procs=0, mem=0, costraints:set={}):
        node_number = 0
        for node in self.nodes:
            if node.gres < gres or node.procs < procs or node.memory < mem:
                continue
            prop = set(node.features)
            if len(costraints) != 0 and not costraints.issubset(prop):
                continue
            node_number += node.cum_number
        return node_number

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
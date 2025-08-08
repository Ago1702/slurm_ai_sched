from abc import ABC, abstractmethod
from enum import Enum
import numpy.random as rnd

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

class TopologyGenerator(object):
    '''
    Class TopologyGenerator: A simple topology generator
    '''

    def __init__(self, min_size, max_size):
        #Add some probability
        self.node_gen = NodeGenerator()
        self.min_size = min_size
        self.max_size = max_size
        self.gpu = 0.3
        self.many_cores = 0.5
        self.mixed = 0.6
        
        self.c = 'a'
        pass

    def increment_char(self):
        if self.c == 'z':
            self.c = 'a'
        else:
            self.c = chr(ord(self.c) + 1)
    
    def group_gen(self):
        node_number = rnd.randint(self.min_size, self.max_size)
        p = rnd.rand()
        if p < self.gpu:
            return self.node_gen.generate_node(name=self.c, num=node_number, features=[f'CPU-{self.c.capitalize()}'], gpu=True)
        elif p < self.many_cores:
            return self.node_gen.generate_node(name=self.c, num=node_number, features=[f'CPU-{self.c.capitalize()}'], many_cores=True)
        else:
            return self.node_gen.generate_node(name=self.c, num=node_number, features=[f'CPU-{self.c.capitalize()}'])
        


    def generate_topology(self, group_num) -> list[Node|list]:
        topo = []
        for i in range(group_num):
            group = self.group_gen()
            topo.append(group)
            self.increment_char()
        return topo


    
if __name__ == '__main__':
        default = Node('DEFAULT', 12, 2, memory=48000)
        nodes_n = Node('n', 12, 2, 4, features=['IB','CPU-N'])
        nodes_m = Node('m', 12, 2, 4, features=['IB','CPU-M'])
        g = Node('g', 12, 2, 1, gres=2, features=['IB','CPU-G'])
        mem = Node('b', 12, 2, 1, memory=512000, gres=2, features=['IB','CPU-G', 'BigMem'])
        nodelist = [default, nodes_n, nodes_m, g, mem]
        gen = NodeGenerator()
        #gen.set_seed(224)
        #rnd.seed(224)
        node_a = gen.generate_node('a', 4, ['IB', 'CPU-A'])
        nodelist.append(node_a)
        node_s = gen.generate_node('s', 8, ['IB', 'CPU-S'], big_mem=True)
        nodelist.append(node_s)
        node_k = gen.generate_node('k', 4, ['IB', 'CPU-K'], gpu=True)
        nodelist.append(node_k)
        node_mc = gen.generate_node('f', 4, ['IB', 'CPU-F'], many_cores=True)
        nodelist.append(node_mc)
        gen.max_sock = 48
        node_gen = gen.generate_node('i', 2, ['IB', 'CPU-I'], big_mem=True, gpu=True, many_cores=True)
        nodelist.append(node_gen)
        for node in nodelist:
            print(node.slurm_formatting())
        topo_gen = TopologyGenerator(2, 10)
        topo = topo_gen.generate_topology(6)
        for node in topo:
            print(node.slurm_formatting())
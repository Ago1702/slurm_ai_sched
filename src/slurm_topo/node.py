from abc import ABC, abstractmethod

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
    


if __name__ == '__main__':
        default = Node('DEFAULT', 12, 2, memory=48000)
        nodes_n = Node('n', 12, 2, 4, features=['IB','CPU-N'])
        nodes_m = Node('m', 12, 2, 4, features=['IB','CPU-M'])
        g = Node('g', 12, 2, 1, gres=2, features=['IB','CPU-G'])
        mem = Node('b', 12, 2, 1, memory=512000, gres=2, features=['IB','CPU-G', 'BigMem'])
        nodelist = [default, nodes_n, nodes_m, g, mem]
        for node in nodelist:
            print(node.slurm_formatting())
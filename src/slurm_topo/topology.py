from slurm_topo.node import Node, NodeGenerator
import numpy.random as rnd

FEATURES = ['BigMem', 'ManyCores']

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
        self.param_dict = {
                    'gpu': False,
                    'many_cores': False,
                    'big_mem': False
                }
        
        self.c = 'a'
        pass

    def increment_char_(self):
        if self.c == 'z':
            self.c = 'a'
        else:
            self.c = chr(ord(self.c) + 1)
    
    def increment_char(self, c):
        if c == 'z':
            return 'a'
        else:
            return chr(ord(c) + 1)

    def randomize_param(self):
        for k in self.param_dict.keys():
            self.param_dict[k] = rnd.choice([True, False])
    
    def group_gen(self):
        node_number = rnd.randint(self.min_size, self.max_size)
        p = rnd.rand()
        if p < self.gpu:
            return self.node_gen.generate_node(name=self.c, num=node_number, features=[f'CPU-{self.c.capitalize()}'], gpu=True)
        elif p < self.many_cores:
            return self.node_gen.generate_node(name=self.c, num=node_number, features=[f'CPU-{self.c.capitalize()}'], many_cores=True)
        elif p < self.mixed:
            nodes = []
            sub_name = 'a'
            for i in range(node_number):
                self.randomize_param()
                node = self.node_gen.generate_node(name=f"{self.c}{sub_name}", num=1, features=[f'CPU-{self.c.capitalize()}'], **self.param_dict)
                nodes.append(node)
                sub_name = self.increment_char(sub_name)
            return nodes
        else:
            return self.node_gen.generate_node(name=self.c, num=node_number, features=[f'CPU-{self.c.capitalize()}'])
        


    def generate_topology(self, group_num) -> Topology:
        topo = []
        for i in range(group_num):
            group = self.group_gen()
            topo.append(group)
            self.increment_char_()
        return Topology(topo)

from slurm_topo.topology import TopologyGenerator, Topology
from slurm_topo.node import NodeGenerator, Node

from slurm_load.user import UserGenerator

from slurm_load.job import JobGenerator

from slurm_load.workload import WorkLoad

import numpy.random as rnd

class SlurmEnvGen:
    """
    Class NodeGenerator: Huge class for handling structure generation.

    parameters:
            max_account: max account number. Default 5.
    
        Node Generator:
            min_proc: Node minimum number * 4 of proc. Default 0.
            max_proc: Node maximum number * 4 of proc. Default 7.
            min_mem: min memry size * 4000. Default 6.
            max_mem: max memry size * 4000. Default 33.
            min_sock: min socket number. Default 1.
            max_mem: max socket number. Default 3.
            min_gres: min number of gpu in gpu nodes. Default 1.
            max_gres: max number of gpu in gpu nodes. Default 9.
        
        Topology Generator:
            min_size: Minimum size of a node group. Default 4.
            max_size: Maximum size of a node group. Default 8.

        User Generator:
            min_usr: Minimum number of user. Default 4.
            max_usr: Maximum number of user. Default 8.
            min_group: Minimum number of group. Default 2.
            min_group: Maximum number of group. Default 4.

        Job Generator:
            max_long_jon: Maximum time for long jobs in hour. default 24.
            max_short_jon: Maximum time for long jobs in minute. Default 60.
            min_mem: min memory required for jobs. Default is 1000.
    """

    def __init__(self, clust_size=6, **args):
        self.clust_size = clust_size
        self.node_gen = NodeGenerator(**args)
        self.topo_gen = TopologyGenerator(node_gen=self.node_gen, **args)
        self.generate_topology()
        self.user_gen = UserGenerator(**args)
        self.generate_users()
        self.job_generator = JobGenerator(topology=self.topology, **args)
        self.workload = WorkLoad(self.users[1:], self.accounts, self.job_generator)


    def generate_topology(self):
        self.topology = self.topo_gen.generate_topology(self.clust_size)
    
    def generate_users(self):
        users, num_group = self.user_gen.generate_users()
        users = list(users)
        users.sort()
        self.users = users
        self.num_group = num_group

        account_n = rnd.randint(1, len(users) + 1)
        accounts = {}
        for i, user in enumerate(users[1:]):
            accounts[user.usr] = f"account{rnd.randint(1, account_n + 1)}"
        
        self.accounts = accounts

    def generate_workload(self, num:int):
        return self.workload.generate_workload(num)




if __name__ == '__main__':
    env = SlurmEnvGen(min_proc=1)
    
import numpy.random as rnd

class User(object):
    '''
    Class User: a simple class for keeping user's info
    '''

    def __init__(self, usr:str='admin', usr_id:int=1000, group:str='admin', group_id:int=1000):
        self.usr = usr
        self.usr_id = usr_id
        self.group = group
        self.group_id = group_id

    def __repr__(self):
        return f"{self.usr}:{self.usr_id}:{self.group}:{self.group_id}"
    
    def __str__(self):
        return f"User {self.usr}:{self.usr_id} Group {self.group}:{self.group_id}"
    
    def __lt__(self, other):
        if isinstance(other, User):
            return self.usr_id < other.usr_id
        raise NotImplementedError('Not Implemented Comparison') 
   
class UserGenerator(object):
    '''
    Class UserGenerator: A simple generator for users
    '''

    def __init__(self, min_usr:int, max_usr:int, min_group:int, max_group:int):
        if min_usr > max_usr:
            raise ValueError("min_usr cannot be greater than max_usr")
        if min_group > max_group:
            raise ValueError("min_group cannot be greater than max_group")
        self.min_usr = min_usr
        self.max_usr = max_usr
        self.min_group = min_group
        self.max_group = max_group 
        
    def set_seed(self, seed:int):
        rnd.seed(seed)

    def generate_users(self) -> tuple[set[User], int]:
        n_group = rnd.randint(self.min_group, self.max_group)
        users = set([User()])
        n_account = rnd.randint(self.min_usr, self.max_usr)
        for i in range(1, n_account):
            gr_i = rnd.randint(1, n_group)
            usr = f'user{i}'
            usr_id = 1000 + i
            group = f'group{gr_i}'
            group_id = 1000 + gr_i
            users.add(User(usr, usr_id, group, group_id))
        return users, n_group
    
def account_gen(users:User, n_account:int):
    accounts = {}
    for i, user in enumerate(users):
        accounts[user.usr] = f"account{rnd.randint(1, n_account + 1)}"
    return accounts
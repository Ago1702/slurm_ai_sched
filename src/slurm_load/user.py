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
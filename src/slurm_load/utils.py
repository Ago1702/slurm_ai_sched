from slurm_load.user import User

def read_users(p='tests/users.sim') -> list[User]:
    user_list = []
    with open(p, mode='r') as f:
        while line := f.readline():
            data = line.split(':')
            if len(data) != 4:
                return []
            user_list.append(User(*data))
    return user_list
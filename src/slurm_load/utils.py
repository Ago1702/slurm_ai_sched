from slurm_load.user import User

def read_users_sim(p='tests/users.sim') -> list[User]:
    user_list = []
    with open(p, mode='r') as f:
        while line := f.readline():
            line = line[:-1]
            data = line.split(':')
            if len(data) != 4:
                return []
            data[1] = int(data[1])
            data[3] = int(data[3])
            user_list.append(User(*data))
    return user_list

def print_users_sim(p, users:set[User]):
    with open(p, mode='w') as f:
        lines = [repr(user) + '\n' for user in users]
        f.writelines(lines)
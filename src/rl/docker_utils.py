import docker
from time import sleep

PORTS  = {
    "8888/tcp":8888,
    '8787/tcp':8787,
}

def mount_stringify(local, virtual):
    return {
        local: {
            'bind': virtual,
            'mode':'rw',
        }
    }

class DockerSched:
    
    def __init__(self, name:str, local_dir, virtual_path):
        self.name = name
        self.client = docker.from_env()
        self.mount = mount_stringify(local_dir, virtual_path)
        self.wdr = virtual_path
    
    def execute(self, user='slurm', cmd='./start.sh'):
        container = self.client.containers.run(
            image="nsimakov/slurm_sim:v3.0",
            name=self.name,
            hostname="slurmsim",
            volumes=self.mount,
            detach=True,
            working_dir=self.wdr,
            #ports=PORTS
        )

        sleep(0.5)

        exit_code, output = container.exec_run(
            cmd=cmd,
            user=user,
            workdir=self.wdr,
        )

        container.stop()
        container.remove()

        return exit_code, output

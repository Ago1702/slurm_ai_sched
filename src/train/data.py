import torch
from torch.utils.data import Dataset
from pathlib import Path

import pandas as pd
from datetime import datetime, timedelta

from app.utils import vectorize_job

import os
import re


date_format = "%Y-%m-%dT%H:%M:%S"

def time_second(s:str):
    hour, minute, second = s.split(':')
    return int(hour) * 3600 + int(minute) * 60 + int(second)

def extract_flags(job:str):
    patterns = {
    "-t": r"-t\s+([0-9:]+)",
    "-n": r"-n\s+(\d+)",
    "--ntasks-per-node": r"--ntasks-per-node[=\s]+(\d+)",
    "--gres": r"--gres[=\s]gpu:+([^\s]+)",
    "--mem": r"--mem[=\s]+(\d+)"
    }

    index = {
    "-t": 4,
    "-n": 0,
    "--ntasks-per-node": 1,
    "--gres": 3,
    "--mem": 2
    }

    results = [0] * 5

    for key, pat in patterns.items():
        match = re.search(pat, job)
        i = index[key]
        if match:
            if key == '-t':
                results[i] = time_second(match.group(1)) / 3600
            else:
                results[i] = int(match.group(1))
        else:
            results[i] = 0   # default when missing
    return results

def reward_calculation(loc:str="/home/ago/tesi/slurm_ai_sched/src/tests/saved/slurm_dataset/env_0/results/slurm_acct.out", p=False):
    executed_time = 0
    wait_time = 0
    data = pd.read_csv(loc, sep='|', header=0)
    eligible = data['Eligible'].apply(lambda x : datetime.strptime(x, date_format))
    start = data['Start'].apply(lambda x : datetime.strptime(x, date_format))
    wait_time = start - eligible
    if p:
        print(eligible)
        print(start)
        print(wait_time)
    wait_time = wait_time.apply(lambda x : x.total_seconds() / (3600)).mean()
    return torch.tensor(wait_time, dtype=torch.float32)

class SlurmDataset(Dataset):
    def __init__(self, path:str|Path, max_i=-1):
        super().__init__()
        self.path = path if isinstance(path, Path) else Path(path)
        self.elements = os.listdir(path)
        self.elements.sort()
        self.labels = self.extract_label()
        self.max_i = max_i 
    
    def extract_label(self):
        labels = []
        for dir_name in self.elements:
            dir_name = Path(dir_name) / "results/slurm_acct.out"
            loc = self.path / dir_name
            label = reward_calculation(loc)
            labels.append(label)
        return labels
    
    def __getitem__(self, index):
        location = Path(self.elements[index]) / "workload/first_job.events"
        location = self.path / location
        with open(location) as f:
            data = [extract_flags(line) for line in f]
        label = self.labels[index]
        data = torch.tensor(data, dtype=torch.float32)
        return data, label
    
    def __len__(self):
        return len(self.elements) if self.max_i == -1 else self.max_i

    
if __name__ == '__main__':
    data = SlurmDataset("/home/ago/tesi/slurm_ai_sched/src/tests/saved/slurm_dataset")
    for i in range(5):
        print(data.elements[i])
        print(data[i])
            
        
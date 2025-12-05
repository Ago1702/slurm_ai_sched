import torch
from train.data import SlurmDataset
from torch.utils.data import DataLoader

from models.rl_models import QNet, SlurmNet

import sys

import wandb

WANDB = True


if WANDB:
    wandb.login()

def init_modules(module):
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight, torch.nn.init.calculate_gain("leaky_relu"))

TRAIN_STEP = 10000
BATCH_SIZE = 64
L_RATE = 0.00005
EPOCHS = 100
HIDDEN_DIM = 2048
DEPTH = 2
DROP = 0.25

project = "Slurm_Sched"


config = {
    'epochs': EPOCHS,
    'lr': L_RATE,
    'hidden_dim':HIDDEN_DIM,
    'depth':DEPTH,
    'dropout':DROP

}

if WANDB:
    run = wandb.init(project=project, config=config)

device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

data = SlurmDataset("/home/ago/tesi/slurm_ai_sched/src/tests/saved/slurm_dataset")
print(len(data))

train_set, val_set = torch.utils.data.random_split(data,[0.8, 0.2])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

model = QNet(
    input_dim=5,
    output_dim=1,
    depth=1,
    #num_heads=16,
    #embedded_dim=512
)

model = SlurmNet(
    input_dim=5,
    mid_dim=16,
    output_dim=1,
    hidden_dim=HIDDEN_DIM,
    depth_first=DEPTH,
    depth_second=DEPTH,
    dropout=DROP
)

model = model.apply(init_modules)
model = model.to(device)

loss_fn = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=L_RATE)

scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=1, total_iters=160)
plateu = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

def train_one_epoch(epoch_index):
    running_loss = 0.
    last_loss = 0.
    for i, data in enumerate(train_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        labels = labels.unsqueeze(-1)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()
        scheduler.step()
        if epoch_index > 10:
            plateu.step(loss.detach())

        # Gather data and report
        running_loss += loss.item()

    return running_loss / (i + 1)

def validate(epoch_index):
    running_loss = 0.
    last_loss = 0.
    model_ = model.eval()
    for i, data in enumerate(val_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        labels = labels.unsqueeze(-1)
        with torch.no_grad():
            outputs = model_(inputs)
            loss = loss_fn(outputs, labels)

        # Gather data and report
        running_loss += loss.item()

    return running_loss / (i + 1)

for epoch in range(EPOCHS):
    loss = train_one_epoch(epoch)
    print(f"Epoch {epoch}: {loss}")
    if WANDB:
        run.log({"loss":loss}, step=epoch)
    if epoch % 5 == 4:
        val_loss = validate(epoch)
        print(f"Validation: {val_loss}")
        if WANDB:
            run.log({"val_loss":val_loss}, step=epoch)

for i, data in enumerate(train_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        labels = labels.unsqueeze(-1)
        

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()
        print(outputs)
        print(labels)
        print(labels - outputs)
        break
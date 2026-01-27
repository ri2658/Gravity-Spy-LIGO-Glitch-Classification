import torch
import torch.nn as nn
import torch.nn.functional as F

from graspy_resnet18 import GraSpyResNet18
from data_preprocessing import preprocessData

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

dataset_path = "../gravity-spy-dataset/"

training_loader, validation_loader, testing_loader = preprocessData(dataset_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GraSpyResNet18().to(device)

# -------- PHASE 1: Train classifier only --------
for param in model.resnet.parameters():
    param.requires_grad = False

for param in model.resnet.fc.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3,                 # higher LR for classifier
    weight_decay=1e-4
)

criterion = nn.CrossEntropyLoss()

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    factor=0.5,
    patience=3
)

def train_one_epoch(epoch_index, tb_writer, optimizer):
    model.train()
    running_loss = 0.0

    for inputs, labels in training_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(training_loader)
    tb_writer.add_scalar('Loss/train', avg_loss, epoch_index)
    return avg_loss

# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/graspy_resnet18_{}'.format(timestamp))
epoch_number = 0

best_vloss = 1_000_000.

PHASE1_EPOCHS = 10
TOTAL_EPOCHS = 30

best_vloss = float("inf")

for epoch in range(PHASE1_EPOCHS):
    print(f'PHASE 1 — EPOCH {epoch + 1}')

    avg_loss = train_one_epoch(epoch, writer, optimizer)

    running_vloss = 0.0
    model.eval()
    with torch.no_grad():
        for vinputs, vlabels in validation_loader:
            vinputs = vinputs.to(device)
            vlabels = vlabels.to(device)
            voutputs = model(vinputs)
            running_vloss += criterion(voutputs, vlabels).item()

    avg_vloss = running_vloss / len(validation_loader)
    print(f'LOSS train {avg_loss:.4f} valid {avg_vloss:.4f}')

    scheduler.step(avg_vloss)

    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        torch.save(model.state_dict(), "best_resnet18.pt")
    
for param in model.resnet.layer4.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=3e-5,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    factor=0.5,
    patience=3
)

for epoch in range(PHASE1_EPOCHS, TOTAL_EPOCHS):
    print(f'PHASE 2 — EPOCH {epoch + 1}')

    avg_loss = train_one_epoch(epoch, writer, optimizer)

    running_vloss = 0.0
    model.eval()
    with torch.no_grad():
        for vinputs, vlabels in validation_loader:
            vinputs = vinputs.to(device)
            vlabels = vlabels.to(device)
            voutputs = model(vinputs)
            running_vloss += criterion(voutputs, vlabels).item()

    avg_vloss = running_vloss / len(validation_loader)
    print(f'LOSS train {avg_loss:.4f} valid {avg_vloss:.4f}')

    scheduler.step(avg_vloss)

    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        torch.save(model.state_dict(), "best_resnet18.pt")
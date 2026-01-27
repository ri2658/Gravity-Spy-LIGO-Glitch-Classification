import torch
import torch.nn as nn
import torch.nn.functional as F

from zevin_coughlin_net import ZevinCoughlinNet
from data_preprocessing import preprocessData

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

dataset_path = "../gravity-spy-dataset/"

training_loader, validation_loader, testing_loader = preprocessData(dataset_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ZevinCoughlinNet().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)

criterion = nn.CrossEntropyLoss()

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=40,
    gamma=0.5
)

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.0
    model.train()

    for inputs, labels in training_loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

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
writer = SummaryWriter('runs/zevin_coughlin_net_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 115

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train()
    avg_loss = train_one_epoch(epoch_number, writer)


    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()
    
    with torch.no_grad():
        for vinputs, vlabels in validation_loader:
            vinputs = vinputs.to(device)
            vlabels = vlabels.to(device)

            voutputs = model(vinputs)
            vloss = criterion(voutputs, vlabels)
            running_vloss += vloss.item()

    avg_vloss = running_vloss / len(validation_loader)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()
    
    scheduler.step()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    if (epoch_number + 1) % 20 == 0:
        torch.save(model.state_dict(), f"checkpoint_epoch_{epoch_number}.pt")
    
    epoch_number += 1
import torch
import numpy as np
from sklearn.metrics import classification_report, f1_score
from resnet_18.graspy_resnet18 import GravitySpyResNet18
from zevin_coughlin_net.data_preprocessing import preprocessData

num_classes = 22
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights_folder = 'ResNet18'
weight_name = 'best_resnet18.pt'
dataset_path = './gravity-spy-dataset/'

_, _, test_loader = preprocessData(dataset_path)
model = GravitySpyResNet18(num_classes=num_classes)
model.load_state_dict(torch.load(weights_folder + '/' + weight_name, map_location=device))
model = model.to(device).eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

macro_f1 = f1_score(all_labels, all_preds, average='macro')
weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
acc = (all_preds == all_labels).mean()
print('accuracy', acc)
print('macro_f1', macro_f1)
print('weighted_f1', weighted_f1)
print(classification_report(all_labels, all_preds, target_names=test_loader.dataset.classes, digits=3))

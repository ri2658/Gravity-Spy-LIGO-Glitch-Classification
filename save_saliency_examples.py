import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from resnet_18.graspy_resnet18 import GraSpyResNet18
from heatmap_algos.AblationCAM import AblationCAM

num_classes = 22
weights_folder = 'ResNet18'
weight_name = 'best_resnet18.pt'
dataset_path = './gravity-spy-dataset/'
out_dir = './saliency_outputs'
os.makedirs(out_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((94, 114)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

test_dataset = datasets.ImageFolder(root=dataset_path + 'test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

model = GraSpyResNet18(num_classes=num_classes)
model.load_state_dict(torch.load(weights_folder + '/' + weight_name, map_location=device))
model = model.to(device).eval()

target_layer = model.resnet.layer4[-1]
cam_generator = AblationCAM(model, target_layer)

# Pick a few representative classes with distinctive morphology
selected_classes = ['Blip', 'Koi_Fish', 'Power_Line', 'Violin_Mode', 'Whistle', 'Helix']
class_to_indices = {c: [] for c in selected_classes}
for idx, (img, label) in enumerate(test_dataset):
    name = test_dataset.classes[label]
    if name in class_to_indices:
        class_to_indices[name].append(idx)

for cls_name in selected_classes:
    idx = class_to_indices[cls_name][0]
    img, label = test_dataset[idx]
    img = img.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img)
        pred = int(torch.argmax(logits, dim=1).item())
    cam = cam_generator.generate(img, class_idx=pred)
    image = img[0, 0].detach().cpu().numpy().astype('float32')

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title(f'True: {test_dataset.classes[label]}\nPred: {test_dataset.classes[pred]}')
    axes[0].axis('off')

    axes[1].imshow(image, cmap='gray')
    axes[1].imshow(cam, cmap='jet', alpha=0.5)
    axes[1].set_title('AblationCAM overlay')
    axes[1].axis('off')

    plt.tight_layout()
    out_path = os.path.join(out_dir, f'{cls_name}_saliency.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path}')

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def preprocessData(path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((94, 114)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = datasets.ImageFolder(
        root=path + "train",
        transform=transform
    )

    val_dataset = datasets.ImageFolder(
        root=path + "validation",
        transform=transform
    )

    test_dataset = datasets.ImageFolder(
        root=path + "test",
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=30, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=30, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=30, shuffle=False)
    
    return train_loader, val_loader, test_loader
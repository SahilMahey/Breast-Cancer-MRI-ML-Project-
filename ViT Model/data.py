# Import libraries
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.long)
        if self.transform:
            image = self.transform(image)
        return image, label

def prepare_data(batch_size=64,x_train=None,y_train=None,x_test=None,y_test=None,x_val=None,y_val=None):
    transform = transforms.Compose(
        [transforms.ToTensor(),])
    # Assuming x_train, x_test, x_val, y_train, y_test, y_val are numpy arrays
    train_dataset = CustomDataset(x_train, y_train, transform=transform)  # Add any required transformations
    print("\nTraining Dataset Customised\n")
    test_dataset = CustomDataset(x_test, y_test, transform=transform)
    print("\nTesting Dataset Customised\n")
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=2)
    print("\nTraining Dataset Loaded\n")
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=2)
    print("\nTesting Dataset Loaded\n")
    if x_val is not None and y_val is not None:
        val_dataset = CustomDataset(x_val, y_val, transform=transform)
        print("\nValidation Dataset Customised\n")
        valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=2)
        print("\nValidation Dataset Loaded\n")
        return trainloader, testloader, valloader
    else:
        return trainloader, testloader
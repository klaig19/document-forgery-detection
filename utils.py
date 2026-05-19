import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

class CustomDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Traverse the directories for real and fake images
        for label, subdir in enumerate(['real', 'fake']):
            subdir_path = os.path.join(image_dir, subdir)
            for img_name in os.listdir(subdir_path):
                if img_name.endswith('.jpg') or img_name.endswith('.png'):
                    self.images.append(os.path.join(subdir_path, img_name))
                    self.labels.append(label)  # 0 for real, 1 for fake

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

def get_data_loaders(image_dir, batch_size=16):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = CustomDataset(image_dir, transform=transform)
    
    # Handle data imbalance with weighted sampling
    class_counts = [len([1 for label in dataset.labels if label == 0]),
                    len([1 for label in dataset.labels if label == 1])]
    class_weights = [sum(class_counts) / c for c in class_counts]
    sample_weights = [class_weights[label] for label in dataset.labels]
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # DataLoader
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
    
    return train_loader
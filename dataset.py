
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class Num16Dataset(Dataset):
    def __init__(self, images_dir, labels_file, transform=None):
        self.images_dir = images_dir
        self.labels = pd.read_csv(labels_file)  # expects columns: 'img', 'label'
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        img_path = os.path.join(self.images_dir, row['img'])
        image = Image.open(img_path).convert('L')
        label = int(row['label'])
        if self.transform:
            image = self.transform(image)
        return image, label

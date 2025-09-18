import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class SiameseGentrificationDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = self._load_image_paths()
        
        # Filter out missing images
        self.data = self.data[self.data.apply(
            lambda row: row['pid_1'] in self.image_paths and row['pid_2'] in self.image_paths, axis=1)
        ].reset_index(drop=True)

    def _load_image_paths(self):
        return {file.replace('.jpg', ''): os.path.join(self.image_dir, file) 
                for file in os.listdir(self.image_dir) if file.endswith('.jpg')}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pid_1, pid_2 = str(self.data.iloc[idx]['pid_1']), str(self.data.iloc[idx]['pid_2'])
        
        early_image = Image.open(self.image_paths[pid_1]).convert('RGB')
        late_image = Image.open(self.image_paths[pid_2]).convert('RGB')

        if self.transform:
            early_image, late_image = self.transform(early_image), self.transform(late_image)

        label = self.data.iloc[idx]['Label']
        return early_image, late_image, torch.tensor(label, dtype=torch.float32)

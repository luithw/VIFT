import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class LatentDinoVectorDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.dino_dir = root_dir.replace('kitti_latent_data', 'kitti_dino_features')
        self.latent_files = os.listdir(root_dir)

        # Verify DINO features directory exists
        if not os.path.exists(self.dino_dir):
            raise ValueError(f"DINO features directory not found at: {self.dino_dir}")

    def __len__(self):
        return int(len(self.latent_files) / 4)

    def __getitem__(self, idx):
        # Load original latent vectors and metadata
        latent_vector = np.load(os.path.join(self.root_dir, f"{idx}.npy"))
        gt = np.load(os.path.join(self.root_dir, f"{idx}_gt.npy"))
        rot = np.load(os.path.join(self.root_dir, f"{idx}_rot.npy"))
        w = np.load(os.path.join(self.root_dir, f"{idx}_w.npy"))

        # Load corresponding DINO features
        dino_features = np.load(os.path.join(self.dino_dir, f"{idx}_features.npy"))

        return (
            torch.from_numpy(latent_vector).to(torch.float),
            torch.from_numpy(dino_features).to(torch.float),
            torch.from_numpy(rot),
            torch.from_numpy(w),
        ), torch.from_numpy(gt).to(torch.float).squeeze()

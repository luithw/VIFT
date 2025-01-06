# we will use many functions inside the parent folder
import sys

sys.path.insert(0, '../')

import math
import os
import torch
import timm
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

# Import the same KITTI dataset and transforms
from src.data.components.KITTI_dataset import KITTI
from src.utils import custom_transform


class FeatureExtractor(torch.nn.Module):
    def __init__(self, model, layers):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        # Register hooks for the specified layers
        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id):
        def fn(_, __, output):
            self._features[layer_id] = output

        return fn

    def forward(self, x):
        _ = self.model(x)  # Run the model to trigger hooks

        # Stack features from desired layers
        all_features = []
        for layer in self.layers:
            feat = self._features[layer]
            all_features.append(feat)
        stacked_features = torch.cat(all_features, dim=2)
        return stacked_features


def main():
    # Set up transforms
    transform_train = [
        custom_transform.ToTensor(),
        custom_transform.Resize((224, 224))  # DINO ViT expects 224x224
    ]
    transform_train = custom_transform.Compose(transform_train)

    # Create dataset and dataloader
    dataset = KITTI("kitti_data",
                    train_seqs=['05', '07', '10'],  # Validation sequences
                    transform=transform_train,
                    sequence_length=11)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Create save directory
    save_dir = "kitti_dino_features/val"
    os.makedirs(save_dir, exist_ok=True)

    # Load DINO ViT model
    model = timm.create_model('vit_base_patch8_224.dino', pretrained=True)

    # Define layers to extract (3,7,11)
    extract_layers = ['blocks.3', 'blocks.7', 'blocks.11']

    # Create feature extractor wrapper
    feature_extractor = FeatureExtractor(model, extract_layers)
    feature_extractor.eval()
    feature_extractor.to("cuda")

    print("Starting validation feature extraction...")
    with torch.no_grad():
        for i, ((imgs, imus, rot, w), gts) in tqdm(enumerate(loader), total=len(loader)):
            # Skip the first image because it is the starting point
            imgs = imgs[:, 1:]

            # imgs shape: [1, 10, 3, 224, 224]
            B, S, C, H, W = imgs.shape

            # Reshape for batch processing
            imgs = imgs.view(-1, C, H, W)  # [10, 3, 224, 224]
            imgs = imgs.to("cuda")

            # Extract features
            features = feature_extractor(imgs)  # [10, num_tokens, feature_dim*3]

            # Save features and metadata
            np.save(os.path.join(save_dir, f"{i}_features.npy"),
                    features.cpu().numpy())

if __name__ == "__main__":
    main()
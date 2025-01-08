import sys

sys.path.insert(0, '../')

import math
import os
import torch
import timm
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

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


def check_missing_sequences(save_dir, total_sequences):
    """Check which sequence indices are missing from the save directory."""
    existing_files = set()
    for filename in os.listdir(save_dir):
        if filename.endswith('_features.npy'):
            idx = int(filename.split('_')[0])
            existing_files.add(idx)

    missing_indices = []
    for i in range(total_sequences):
        if i not in existing_files:
            missing_indices.append(i)

    return missing_indices


def main():
    # Set up transforms
    transform_train = [
        custom_transform.ToTensor(),
        custom_transform.Resize((224, 224))  # DINO ViT expects 224x224
    ]
    transform_train = custom_transform.Compose(transform_train)

    # Create dataset
    dataset = KITTI("kitti_data",
                    train_seqs=['00', '01', '02', '04', '06', '08', '09'],
                    transform=transform_train,
                    sequence_length=11)

    save_dir = "kitti_dino_features/train_10"
    os.makedirs(save_dir, exist_ok=True)

    print(f"total dino sequences: {len(dataset)}")
    breakpoint()

    # Check which sequences are missing
    missing_indices = check_missing_sequences(save_dir, len(dataset))
    print(f"Found {len(missing_indices)} missing sequences")

    if not missing_indices:
        print("No missing sequences found. Exiting...")
        return

    # Create a subset dataset with only the missing sequences
    subset_dataset = Subset(dataset, missing_indices)
    loader = DataLoader(subset_dataset, batch_size=1, shuffle=False)

    # Load DINO ViT model
    model = timm.create_model('vit_base_patch8_224.dino', pretrained=True)

    # Define layers to extract (3,7,11)
    extract_layers = ['blocks.3', 'blocks.7', 'blocks.11']

    # Create feature extractor wrapper
    feature_extractor = FeatureExtractor(model, extract_layers)
    feature_extractor.eval()
    feature_extractor.to("cuda")

    print("Starting feature extraction for missing sequences...")
    with torch.no_grad():
        for i, ((imgs, imus, rot, w), gts) in tqdm(enumerate(loader), total=len(loader)):
            # Get the original index from missing_indices
            original_idx = missing_indices[i]

            # Skip the first image because it is the starting point
            imgs = imgs[:, 1:]

            # imgs shape: [1, 10, 3, 224, 224]
            B, S, C, H, W = imgs.shape

            # Reshape for batch processing
            imgs = imgs.view(-1, C, H, W)
            imgs = imgs.to("cuda")

            # Extract features
            features = feature_extractor(imgs)

            # Save features using the original index
            np.save(os.path.join(save_dir, f"{original_idx}_features.npy"),
                    features.cpu().numpy())

    print("Feature extraction complete!")


if __name__ == "__main__":
    main()
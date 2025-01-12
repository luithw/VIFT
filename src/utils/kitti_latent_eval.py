# import dependencies
import os
from tqdm import tqdm
import numpy as np

import glob
import scipy.io as sio
import torch

from src.models.components.pose_mvformer import PoseMVFormer
from src.utils import custom_transform
from src.utils.kitti_utils import read_pose_from_text, saveSequence
from src.utils.kitti_eval import plotPath_2D, kitti_eval, data_partition
from src.models.components.vsvio import Encoder

from natsort import natsorted


# wrap the model with an encoder for testing
class WrapperModel(torch.nn.Module):
    def __init__(self, params):
        super(WrapperModel, self).__init__()
        self.Feature_net = Encoder(params)

    def forward(self, imgs, imus):
        feat_v, feat_i = self.Feature_net(imgs, imus)
        memory = torch.cat((feat_v, feat_i), 2)
        return memory


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


class KITTI_tester_latent():
    def __init__(self, args, wrapper_weights_path, use_history_in_eval=False):
        super(KITTI_tester_latent, self).__init__()

        # generate data loader for each path
        self.dataloader = []
        for seq in args.val_seq:
            self.dataloader.append(data_partition(args, seq))
        self.args = args

        # Initialize and load pretrained weights for the wrapper model
        self.wrapper_model = WrapperModel(args)
        self.load_wrapper_weights(wrapper_weights_path)
        self.wrapper_model.eval()
        self.wrapper_model.to(self.args.device)
        self.use_history_in_eval = use_history_in_eval
        self.dino_extractor = None

    def load_wrapper_weights(self, weights_path):
        if os.path.exists(weights_path):
            pretrained_w = torch.load(weights_path, map_location='cpu')

            model_dict = self.wrapper_model.state_dict()
            update_dict = {k: v for k, v in pretrained_w.items() if k in model_dict}

            # Check if update dict is equal to model dict
            assert len(update_dict.keys()) == len(
                self.wrapper_model.Feature_net.state_dict().keys()), "Some weights are not loaded"

            self.wrapper_model.load_state_dict(update_dict)
            print(f"Loaded wrapper model weights from {weights_path}")
        else:
            print(f"Warning: Wrapper model weights not found at {weights_path}")

    def test_one_path(self, net, df, num_gpu=1):
        pose_list = []
        self.hist = None

        # Initialize DINO model only if needed
        if isinstance(net, PoseMVFormer) and self.dino_extractor is None:
            import timm
            model = timm.create_model('vit_base_patch8_224.dino', pretrained=True)
            extract_layers = ['blocks.3', 'blocks.7', 'blocks.11']

            self.dino_extractor = FeatureExtractor(model, extract_layers)
            self.dino_extractor.eval()
            self.dino_extractor.to(self.args.device)

        for i, (image_seq, imu_seq, gt_seq) in tqdm(enumerate(df), total=len(df), smoothing=0.9):
            x_in = image_seq.unsqueeze(0).repeat(num_gpu, 1, 1, 1, 1).to(self.args.device)
            i_in = imu_seq.unsqueeze(0).repeat(num_gpu, 1, 1).to(self.args.device)

            with torch.inference_mode():
                # Generate latent representations
                latents = self.wrapper_model(x_in, i_in)

                if isinstance(net, PoseMVFormer):
                    # Extract DINO features for each frame
                    B, S, C, H, W = x_in.shape
                    x_in_reshaped = x_in.view(-1, C, H, W)  # Reshape to process all frames at once

                    # Resize to 224x224 for DINO
                    x_in_resized = torch.nn.functional.interpolate(
                        x_in_reshaped,
                        size=(224, 224),
                        mode='bilinear',
                        align_corners=False
                    )

                    dino_features = self.dino_extractor(x_in_resized)

                    # Skip the first image because it is the starting point
                    dino_features = dino_features[1:]

                    # Reshape DINO features back to sequence format
                    _, num_tokens, feature_dim = dino_features.shape
                    dino_features = dino_features.view(B, S - 1, num_tokens, feature_dim)

                    # Create batch tuple expected by PoseMVFormer
                    batch = (latents, dino_features, None, None)
                else:
                    batch = (latents, None, None)

                # Accumulate poses using appropriate batch format
                if (self.hist is not None) and self.use_history_in_eval:
                    results = torch.zeros(latents.shape[0], latents.shape[1], 6)
                    for idx in range(latents.shape[1]):
                        self.hist = torch.roll(self.hist, -1, 1)
                        self.hist[:, -1, :] = latents[:, idx, :]
                        if isinstance(net, PoseMVFormer):
                            x = (self.hist, dino_features, None, None)
                        else:
                            x = (self.hist, None, None)
                        result = net(x, gt_seq)
                        results[:, idx, :] = result[:, -1, :]
                    pose = results
                else:
                    self.hist = latents
                    pose = net(batch, gt_seq)

            pose_list.append(pose[0, :, :].detach().cpu().numpy())
        pose_est = np.vstack(pose_list)
        return pose_est

    def eval(self, net, selection=None, num_gpu=1, p=0.5):
        self.errors = []
        self.est = []
        for i, seq in enumerate(self.args.val_seq):
            print(f'testing sequence {seq}')
            pose_est = self.test_one_path(net, self.dataloader[i], num_gpu=num_gpu)
            pose_gt = self.dataloader[i].poses_rel
            pose_est_global, pose_gt_global, t_rel, r_rel, t_rmse, r_rmse, speed = kitti_eval(pose_est, self.dataloader[
                i].poses_rel)

            self.est.append({'pose_est': pose_est, 'pose_gt': pose_gt, 'pose_est_global': pose_est_global,
                             'pose_gt_global': pose_gt_global, 'speed': speed})
            self.errors.append({'t_rel': t_rel, 'r_rel': r_rel, 't_rmse': t_rmse, 'r_rmse': r_rmse})

        return self.errors

    def generate_plots(self, save_dir, window_size):
        for i, seq in enumerate(self.args.val_seq):
            plotPath_2D(seq,
                        self.est[i]['pose_gt_global'],
                        self.est[i]['pose_est_global'],
                        save_dir,
                        self.est[i]['speed'],
                        window_size)

    def save_text(self, save_dir):
        for i, seq in enumerate(self.args.val_seq):
            path = save_dir / '{}_pred.txt'.format(seq)
            saveSequence(self.est[i]['pose_est_global'], path)
            print('Seq {} saved'.format(seq))


class KITTI_tester_latent_tokenized():
    def __init__(self, args, wrapper_weights_path, use_history_in_eval=False):
        super().__init__()

        # generate data loader for each path
        self.dataloader = []
        for seq in args.val_seq:
            self.dataloader.append(data_partition(args, seq))
        self.args = args

        # Initialize and load pretrained weights for the wrapper model
        self.wrapper_model = WrapperModel(args)
        self.load_wrapper_weights(wrapper_weights_path)
        self.wrapper_model.eval()
        self.wrapper_model.to(self.args.device)
        self.use_history_in_eval = use_history_in_eval

    def load_wrapper_weights(self, weights_path):
        if os.path.exists(weights_path):
            pretrained_w = torch.load(weights_path, map_location='cpu')

            model_dict = self.wrapper_model.state_dict()
            update_dict = {k: v for k, v in pretrained_w.items() if k in model_dict}

            # Check if update dict is equal to model dict
            assert len(update_dict.keys()) == len(
                self.wrapper_model.Feature_net.state_dict().keys()), "Some weights are not loaded"

            self.wrapper_model.load_state_dict(update_dict)
            print(f"Loaded wrapper model weights from {weights_path}")
        else:
            print(f"Warning: Wrapper model weights not found at {weights_path}")

    def test_one_path(self, net, df, num_gpu=1):
        pose_list = []
        self.hist = None
        for i, (image_seq, imu_seq, gt_seq) in tqdm(enumerate(df), total=len(df), smoothing=0.9):
            x_in = image_seq.unsqueeze(0).repeat(num_gpu, 1, 1, 1, 1).to(self.args.device)
            i_in = imu_seq.unsqueeze(0).repeat(num_gpu, 1, 1).to(self.args.device)

            with torch.inference_mode():
                # Generate latent representations
                latents = self.wrapper_model(x_in, i_in)
                # accumulate poses by passing latents to the main model

                if (self.hist is not None) and self.use_history_in_eval:
                    results = torch.zeros(latents.shape[0], latents.shape[1], 6)
                    for idx in range(latents.shape[1]):
                        self.hist = torch.roll(self.hist, -1,
                                               1)  # shift so that index 0 becomes last one, shift in seq dim
                        self.hist[:, -1, :] = latents[:, idx, :]
                        x = (self.hist, None, None)
                        result, _ = net(x, gt_seq)  # batch_size, seq_len, 6
                        results[:, idx, :] = result[:, -1, :]
                    pose = results
                else:
                    self.hist = latents
                    pose, _ = net((latents, None, None), gt_seq)
            pose_list.append(pose[0, :, :].detach().cpu().numpy())
        pose_est = np.vstack(pose_list)
        return pose_est

    def eval(self, net, selection=None, num_gpu=1, p=0.5):
        self.errors = []
        self.est = []
        for i, seq in enumerate(self.args.val_seq):
            print(f'testing sequence {seq}')
            pose_est = self.test_one_path(net, self.dataloader[i], num_gpu=num_gpu)
            pose_est_global, pose_gt_global, t_rel, r_rel, t_rmse, r_rmse, speed = kitti_eval(pose_est, self.dataloader[
                i].poses_rel)

            self.est.append({'pose_est_global': pose_est_global, 'pose_gt_global': pose_gt_global, 'speed': speed})
            self.errors.append({'t_rel': t_rel, 'r_rel': r_rel, 't_rmse': t_rmse, 'r_rmse': r_rmse})

        return self.errors

    def generate_plots(self, save_dir, window_size):
        for i, seq in enumerate(self.args.val_seq):
            plotPath_2D(seq,
                        self.est[i]['pose_gt_global'],
                        self.est[i]['pose_est_global'],
                        save_dir,
                        self.est[i]['speed'],
                        window_size)

    def save_text(self, save_dir):
        for i, seq in enumerate(self.args.val_seq):
            path = save_dir / '{}_pred.txt'.format(seq)
            saveSequence(self.est[i]['pose_est_global'], path)
            print('Seq {} saved'.format(seq))

import os

import torch
import torch.nn as nn
import math


class PoseMVFormer(nn.Module):

    def __init__(
            self,
            input_dim=768,  # visual_inertial_features dim
            dino_dim=2304,  # dino_features dim
            embedding_dim=384,  # Common dimension after projection
            num_layers=2,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            num_entities=3,
            smart_final='one'  # Options: 'max', 'one', 'avg', 'lin'
    ):
        super().__init__()
        assert smart_final in ['max', 'one', 'avg',
                               'lin'], f"smart_final must be one of ['max', 'one', 'avg', 'lin'], got {smart_final}"

        self.smart_final = smart_final

        # Project both feature types to common dimension
        self.vi_projection = nn.Linear(input_dim, embedding_dim)

        # LSTP Components
        self.nst = num_entities  # Static tokens
        self.nsdt = 0  # No dynamic tokens
        self.d_model = embedding_dim

        # Learnable queries for static tokens
        self.Q_s = nn.Parameter(torch.empty([1, self.nst, embedding_dim], dtype=torch.float32))
        self.Q_s_b = nn.Parameter(torch.empty(embedding_dim, dtype=torch.float32))

        # Projections for key and value
        self.linear_K2d = nn.Linear(dino_dim, embedding_dim)
        self.linear_V2d = nn.Linear(dino_dim, embedding_dim)

        # Try to load LSTP parameters if available
        lstp_path = "./trained_lstp/lstp_params.pth"
        if os.path.exists(lstp_path):
            print(f"Loading LSTP parameters from {lstp_path}")
            lstp_state = torch.load(lstp_path, map_location='cpu')
            load_success = True

            # Check all parameters exist and shapes match
            param_pairs = [
                ('Q_s', self.Q_s),
                ('Q_s_b', self.Q_s_b),
                ('K_proj.weight', self.linear_K2d.weight),
                ('K_proj.bias', self.linear_K2d.bias),
                ('V_proj.weight', self.linear_V2d.weight),
                ('V_proj.bias', self.linear_V2d.bias)
            ]

            for saved_name, param in param_pairs:
                if saved_name not in lstp_state:
                    print(f"Missing parameter {saved_name} in LSTP file")
                    load_success = False
                    break
                if lstp_state[saved_name].shape != param.shape:
                    print(f"Shape mismatch for {saved_name}. Expected {param.shape}, "
                          f"got {lstp_state[saved_name].shape}")
                    load_success = False
                    break

            if load_success:
                # Load parameters
                for saved_name, param in param_pairs:
                    param.data.copy_(lstp_state[saved_name])
                    param.requires_grad = False  # Freeze parameter
                print("Successfully loaded and froze LSTP parameters")
            else:
                print("Failed to load LSTP parameters, using default initialization")
                self._init_lstp_params()
        else:
            print(f"LSTP parameter file not found at {lstp_path}")
            self._init_lstp_params()

        # You may want to verify parameters are frozen
        for name, param in self.named_parameters():
            if any(p[0].replace('.', '') in name for p in param_pairs):
                assert not param.requires_grad, f"Parameter {name} should be frozen"

        # Transformer components
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )

        # Linear reduction layer for 'lin' smart_final option
        if smart_final == 'lin':
            self.lin_final = nn.Linear(num_entities * embedding_dim, embedding_dim)

        # Final pose prediction
        self.pose_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(embedding_dim, 6)
        )

    def _init_lstp_params(self):
        """Default initialization for LSTP parameters"""
        nn.init.kaiming_uniform_(self.Q_s, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.Q_s)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.Q_s_b, -bound, bound)
        # Key and value projections use their default initializations

    def positional_embedding(self, seq_length, device):
        pos = torch.arange(0, seq_length, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device).float() *
                             -(math.log(10000.0) / self.d_model))
        pos_embedding = torch.zeros(seq_length, self.d_model, device=device)
        pos_embedding[:, 0::2] = torch.sin(pos * div_term)
        pos_embedding[:, 1::2] = torch.cos(pos * div_term)
        return pos_embedding.unsqueeze(0)  # [1, S, D]

    def forward(self, batch, gt=None):
        visual_inertial_features, dino_features, _, _ = batch
        batch_size, seq_length, num_tokens, dino_dim = dino_features.shape

        # LSTP Cross Attention
        K = self.linear_K2d(dino_features)  # [B, S, N+1, embedding_dim]
        V = self.linear_V2d(dino_features)  # [B, S, N+1, embedding_dim]
        Q = self.Q_s + self.Q_s_b  # [1, nst, embedding_dim]

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)  # [1, nst, N+1]
        attn = torch.softmax(scores, dim=-1)  # [1, nst, N+1]

        # Apply attention to values
        entity_features = torch.matmul(attn, V)  # [B*S, nst, embedding_dim]

        # Project visual_inertial_features to dino dimension
        vi_features = self.vi_projection(visual_inertial_features)  # [B, S, dino_dim]
        vi_features = vi_features.unsqueeze(2)  # [B, S, 1, dino_dim]

        # Combine features along token dimension
        combined_features = torch.cat([vi_features, entity_features], dim=2)  # [B, S, N+1, dino_dim]

        num_tokens = self.nst + 1

        # Reshape for transformer - combine batch and nst dimensions
        combined_features = combined_features.view(batch_size, seq_length, num_tokens, -1)  # [B, S, nst, embedding_dim]
        combined_features = combined_features.transpose(1, 2)  # [B, nst, S, embedding_dim]
        combined_features = combined_features.reshape(batch_size * num_tokens, seq_length, -1)  # [B*nst, S, embedding_dim]

        # Add positional embeddings
        pos_emb = self.positional_embedding(seq_length, combined_features.device)  # [1, S, D]
        combined_features = combined_features + pos_emb  # Broadcasting to [B*nst, S, D]

        # Generate causal mask for temporal sequence only
        mask = self.generate_causal_mask(seq_length, combined_features.device)  # [S, S]

        # Pass through transformer - each entity sequence attends to itself only
        time_fused_features = self.transformer_encoder(combined_features, mask=mask)  # [B*nst, S, D]

        # Reshape back
        time_fused_features = time_fused_features.view(batch_size, num_tokens, seq_length, -1)  # [B, nst, S, D]
        time_fused_features = time_fused_features.transpose(1, 2)  # [B, S, nst, D]

        # Apply different smart_final methods
        if self.smart_final == 'max':
            # Max pool across entity dimension
            pose_features, _ = torch.max(time_fused_features, dim=2)  # [B, S, D]

        elif self.smart_final == 'one':
            # Take first entity token (CLS-style)
            pose_features = time_fused_features[:, :, 0, :]  # [B, S, D]

        elif self.smart_final == 'avg':
            # Average pool across entity dimension
            pose_features = torch.mean(time_fused_features, dim=2)  # [B, S, D]

        elif self.smart_final == 'lin':
            # Linear reduction of concatenated entities
            time_fused_features = time_fused_features.view(batch_size, seq_length, -1)  # [B, S, nst*D]
            pose_features = self.lin_final(time_fused_features)  # [B, S, D]

        # Predict poses
        poses = self.pose_head(pose_features)  # [B, S, 6]

        return poses

    def generate_causal_mask(self, size, device):
        return torch.triu(
            torch.full((size, size), float('-inf'), device=device),
            diagonal=1
        )
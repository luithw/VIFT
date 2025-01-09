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
        self.vi_projection = nn.Linear(input_dim, dino_dim)

        # LSTP Components
        self.nst = num_entities  # Static tokens
        self.nsdt = 0  # No dynamic tokens
        self.d_model = embedding_dim

        # Learnable queries for static tokens
        self.Q_s = nn.Parameter(torch.empty([1, self.nst, embedding_dim], dtype=torch.float32))
        nn.init.kaiming_uniform_(self.Q_s, a=math.sqrt(5))
        self.Q_s_b = nn.Parameter(torch.empty(embedding_dim, dtype=torch.float32))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.Q_s)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.Q_s_b, -bound, bound)

        # Try to load LSTP parameters if available
        lstp_path = "./trained_lstp/lstp_params.pth"
        if os.path.exists(lstp_path):
            try:
                print(f"Loading LSTP parameters from {lstp_path}")
                lstp_state = torch.load(lstp_path, map_location='cpu')
                if 'Q_s' in lstp_state and 'Q_s_b' in lstp_state:
                    # Verify shapes match
                    if (lstp_state['Q_s'].shape == self.Q_s.shape and
                            lstp_state['Q_s_b'].shape == self.Q_s_b.shape):
                        self.Q_s.data.copy_(lstp_state['Q_s'])
                        self.Q_s_b.data.copy_(lstp_state['Q_s_b'])
                        print("Successfully loaded LSTP parameters")
                    else:
                        print(f"Shape mismatch in LSTP parameters. Expected {self.Q_s.shape}, {self.Q_s_b.shape} "
                              f"but got {lstp_state['Q_s'].shape}, {lstp_state['Q_s_b'].shape}")
                        self._init_lstp_params()  # Fall back to default initialization
                else:
                    print("LSTP parameter file doesn't contain expected keys")
                    self._init_lstp_params()
            except Exception as e:
                print(f"Error loading LSTP parameters: {str(e)}")
                self._init_lstp_params()
        else:
            print(f"LSTP parameter file not found at {lstp_path}")
            self._init_lstp_params()

        # Projections for key and value
        self.key_projection = nn.Linear(dino_dim, embedding_dim)
        self.value_projection = nn.Linear(dino_dim, embedding_dim)

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

        # Project visual_inertial_features to dino dimension
        vi_features = self.vi_projection(visual_inertial_features)  # [B, S, dino_dim]
        vi_features = vi_features.unsqueeze(2)  # [B, S, 1, dino_dim]

        # Combine features along token dimension
        combined_features = torch.cat([vi_features, dino_features], dim=2)  # [B, S, N+1, dino_dim]

        # LSTP Cross Attention
        K = self.key_projection(combined_features)  # [B, S, N+1, embedding_dim]
        V = self.value_projection(combined_features)  # [B, S, N+1, embedding_dim]
        Q = self.Q_s + self.Q_s_b  # [1, nst, embedding_dim]

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)  # [1, nst, N+1]
        attn = torch.softmax(scores, dim=-1)  # [1, nst, N+1]

        # Apply attention to values
        entity_features = torch.matmul(attn, V)  # [B*S, nst, embedding_dim]

        # Reshape for transformer - combine batch and nst dimensions
        entity_features = entity_features.view(batch_size, seq_length, self.nst, -1)  # [B, S, nst, embedding_dim]
        entity_features = entity_features.transpose(1, 2)  # [B, nst, S, embedding_dim]
        entity_features = entity_features.reshape(batch_size * self.nst, seq_length, -1)  # [B*nst, S, embedding_dim]

        # Add positional embeddings
        pos_emb = self.positional_embedding(seq_length, entity_features.device)  # [1, S, D]
        entity_features = entity_features + pos_emb  # Broadcasting to [B*nst, S, D]

        # Generate causal mask for temporal sequence only
        mask = self.generate_causal_mask(seq_length, entity_features.device)  # [S, S]

        # Pass through transformer - each entity sequence attends to itself only
        entity_features = self.transformer_encoder(entity_features, mask=mask)  # [B*nst, S, D]

        # Reshape back
        entity_features = entity_features.view(batch_size, self.nst, seq_length, -1)  # [B, nst, S, D]
        entity_features = entity_features.transpose(1, 2)  # [B, S, nst, D]

        # Apply different smart_final methods
        if self.smart_final == 'max':
            # Max pool across entity dimension
            entity_features, _ = torch.max(entity_features, dim=2)  # [B, S, D]

        elif self.smart_final == 'one':
            # Take first entity token (CLS-style)
            entity_features = entity_features[:, :, 0, :]  # [B, S, D]

        elif self.smart_final == 'avg':
            # Average pool across entity dimension
            entity_features = torch.mean(entity_features, dim=2)  # [B, S, D]

        elif self.smart_final == 'lin':
            # Linear reduction of concatenated entities
            entity_features = entity_features.view(batch_size, seq_length, -1)  # [B, S, nst*D]
            entity_features = self.lin_final(entity_features)  # [B, S, D]

        # Predict poses
        poses = self.pose_head(entity_features)  # [B, S, 6]

        return poses

    def generate_causal_mask(self, size, device):
        return torch.triu(
            torch.full((size, size), float('-inf'), device=device),
            diagonal=1
        )
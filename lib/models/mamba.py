import sys
import torch
import torch.nn as nn
import os

# Append PoseMamba path to reuse BiSTSSM locally
POSEMAMBA_ROOT_PATH = r'./PoseMamba'
if POSEMAMBA_ROOT_PATH not in sys.path:
    sys.path.append(POSEMAMBA_ROOT_PATH)

try:
    from lib.model.mambablocks import BiSTSSM as SS2D
except ImportError as e:
    print(f"Warning: Cannot import BiSTSSM from PoseMamba. Error: {e}")
    SS2D = None

class Mamba1DBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        # Norm layer for stability
        self.norm = nn.LayerNorm(d_model)
        
        # Bidirectional 1D Mamba (v052d uses scan_mode="bidi")
        if SS2D is not None:
            self.op = SS2D(
                d_model=d_model,
                forward_type="v052d", 
                channel_first=False # Expects (B, H, W, C)
            )
        else:
            self.op = nn.Identity()

    def forward(self, x):
        """
        x: [B, T, D] - Batch, Time (Frames), Feature Dim
        """
        if isinstance(self.op, nn.Identity):
            return x

        # 1. Normalize
        x_norm = self.norm(x)
        
        # 2. Reshape to (B, H, W, C) where H = T, W = 1, C = D
        x_reshaped = x_norm.unsqueeze(2) # [B, T, D] -> [B, T, 1, D]
        
        # 3. Pass through Mamba 1D
        y = self.op(x_reshaped) # Output: [B, T, 1, D]
        
        # 4. Restore shape [B, T, D]
        y = y.squeeze(2)
        
        # 5. Residual Connection
        return x + y

class Mamba2DSpatialBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        # Norm layer for stability
        self.norm = nn.LayerNorm(d_model)
        
        # Limb-based 2D Spatial Mamba
        if SS2D is not None:
            self.op = SS2D(
                d_model=d_model,
                forward_type="v2_plus_poselimbs", 
                channel_first=False # Expects (B, H, W, C) -> (B, T, J, D)
            )
        else:
            self.op = nn.Identity()

    def forward(self, x):
        """
        x: [B, T, J, D] - Batch, Time (Frames), Joints, Feature Dim
        """
        if isinstance(self.op, nn.Identity):
            return x

        # 1. Normalize
        x_norm = self.norm(x)
        
        # 2. Pass through Limb-based Mamba 2D
        # input shape: [B, H=T, W=J, C=D]
        y = self.op(x_norm) # Output: [B, T, J, D]
        
        # 3. Residual Connection
        return x + y

class Mamba1DLocalBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mamba1d = Mamba1DBlock(d_model)

    def forward(self, x):
        """
        x: [B, T, V, D] - Batch, Time (Frames), Vertex (Joints), Feature Dim
        """
        B, T, V, D = x.shape
        # Reshape to treat each joint in the batch as an independent sequence
        # [B, T, V, D] -> [B, V, T, D] -> [B*V, T, D]
        x_reshaped = x.transpose(1, 2).contiguous().view(B * V, T, D)
        
        # Pass through Mamba 1D (Processes the Time dimension for each joint separately)
        y = self.mamba1d(x_reshaped)
        
        # Restore original shape: [B*V, T, D] -> [B, V, T, D] -> [B, T, V, D]
        y = y.view(B, V, T, D).transpose(1, 2).contiguous()
        
        return y


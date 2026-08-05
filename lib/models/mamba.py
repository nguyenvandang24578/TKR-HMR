import sys
import torch
import torch.nn as nn
import os

# Append ps-mamba path to reuse SS2D without copying the heavy CUDA backend
PS_MAMBA_LIB_PATH = r'C:\Users\dvnguyen\HMR\ps-mamba\lib'
if PS_MAMBA_LIB_PATH not in sys.path:
    sys.path.append(PS_MAMBA_LIB_PATH)

try:
    from vmamba.models.vmamba import SS2D
except ImportError as e:
    print(f"Warning: Cannot import SS2D from ps-mamba. Error: {e}")
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

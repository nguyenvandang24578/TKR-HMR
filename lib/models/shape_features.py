"""
Shape Feature Extractor — Trích xuất đặc trưng hình thể từ kp2d.

Tính bone lengths (tỉ lệ từng chi) + global proportions (tổng thể)
→ embed thành vector 512-dim → inject vào shape_token.

19 joints = COCO 17 + Pelvis + Neck:
   0: Nose         5: L_Shoulder    10: R_Wrist    15: L_Ankle
   1: L_Eye        6: R_Shoulder    11: L_Hip      16: R_Ankle
   2: R_Eye        7: L_Elbow       12: R_Hip      17: Pelvis
   3: L_Ear        8: R_Elbow       13: L_Knee     18: Neck
   4: R_Ear        9: L_Wrist       14: R_Knee
"""

import torch
import torch.nn as nn


# ============================================================
# Bone pairs cho COCO 19-joint (17 + Pelvis + Neck)
# ============================================================
BONE_PAIRS_19 = [
    # === Chân trái ===
    (11, 13),  # L_Hip → L_Knee          (đùi trái)
    (13, 15),  # L_Knee → L_Ankle        (ống chân trái)
    # === Chân phải ===
    (12, 14),  # R_Hip → R_Knee          (đùi phải)
    (14, 16),  # R_Knee → R_Ankle        (ống chân phải)
    # === Tay trái ===
    (5,  7),   # L_Shoulder → L_Elbow    (cánh tay trên trái)
    (7,  9),   # L_Elbow → L_Wrist       (cánh tay dưới trái)
    # === Tay phải ===
    (6,  8),   # R_Shoulder → R_Elbow    (cánh tay trên phải)
    (8, 10),   # R_Elbow → R_Wrist       (cánh tay dưới phải)
    # === Thân (dọc) ===
    (17, 18),  # Pelvis → Neck           (chiều dài thân)
    (18,  0),  # Neck → Nose             (cổ+đầu)
    # === Độ rộng (ngang) ===
    (11, 12),  # L_Hip → R_Hip           (rộng hông)
    (5,   6),  # L_Shoulder → R_Shoulder (rộng vai)
]

NUM_BONES = len(BONE_PAIRS_19)  # 12


def compute_bone_lengths(kp2d, bone_pairs=None):
    """
    Tính Euclidean distance giữa các cặp joints từ kp2d.

    Args:
        kp2d: (B, T, 19, 2) — normalized 2D keypoints
        bone_pairs: list of (i, j) tuples. Default: BONE_PAIRS_19

    Returns:
        bone_lengths: (B, T, num_bones)
    """
    if bone_pairs is None:
        bone_pairs = BONE_PAIRS_19

    bones = []
    for i, j in bone_pairs:
        diff = kp2d[:, :, i] - kp2d[:, :, j]              # (B, T, 2)
        length = torch.norm(diff, dim=-1, keepdim=True)     # (B, T, 1)
        bones.append(length)

    return torch.cat(bones, dim=-1)                         # (B, T, num_bones)


def compute_global_proportions(kp2d):
    """
    Tính 4 đặc trưng hình thể global từ kp2d.

    Args:
        kp2d: (B, T, 19, 2)

    Returns:
        props: (B, T, 4) — [height, width, aspect_ratio, spread]
    """
    x_coords = kp2d[:, :, :, 0]                                         # (B, T, 19)
    y_coords = kp2d[:, :, :, 1]                                         # (B, T, 19)

    height = y_coords.max(dim=-1)[0] - y_coords.min(dim=-1)[0]          # (B, T)
    width  = x_coords.max(dim=-1)[0] - x_coords.min(dim=-1)[0]          # (B, T)
    aspect = height / (width + 1e-6)                                     # (B, T)

    cx = x_coords.mean(dim=-1, keepdim=True)                             # (B, T, 1)
    cy = y_coords.mean(dim=-1, keepdim=True)                             # (B, T, 1)
    spread = torch.sqrt(
        ((x_coords - cx) ** 2 + (y_coords - cy) ** 2).mean(dim=-1) + 1e-6
    )                                                                    # (B, T)

    return torch.stack([height, width, aspect, spread], dim=-1)          # (B, T, 4)


class ShapeFeatureExtractor(nn.Module):
    """
    Trích xuất + embed đặc trưng hình thể từ kp2d → vector cho shape_token.

    Input:  kp2d (B, T, 19, 2)
    Output: shape_feat (B, T, embed_dim)

    Gồm 2 loại features:
      - Bone lengths: 12 giá trị (tỉ lệ từng chi — đối xứng trái/phải)
      - Global proportions: 4 giá trị (height, width, aspect, spread)
      → Tổng: 16 features → embed → embed_dim
    """

    def __init__(self, embed_dim=512, bone_pairs=None):
        super().__init__()
        self.bone_pairs = bone_pairs or BONE_PAIRS_19
        num_bones = len(self.bone_pairs)
        feat_dim = num_bones + 4  # bones + global props = 12 + 4 = 16

        self.embed = nn.Sequential(
            nn.Linear(feat_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, kp2d):
        """
        Args:
            kp2d: (B, T, 19, 2)
        Returns:
            shape_feat: (B, T, embed_dim) — ready to add to shape_token
        """
        bones = compute_bone_lengths(kp2d, self.bone_pairs)   # (B, T, 12)
        props = compute_global_proportions(kp2d)               # (B, T, 4)

        feats = torch.cat([bones, props], dim=-1)              # (B, T, 16)
        out = self.embed(feats)                                # (B, T, embed_dim)
        out = self.norm(out)

        return out

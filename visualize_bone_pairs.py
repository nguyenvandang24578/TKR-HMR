"""
Visualize COCO 19 joints (17 + Pelvis + Neck) + 12 bone pairs.
Chạy: python visualize_bone_pairs.py
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ============================================================
# COCO 19 joints = COCO 17 + Pelvis(17) + Neck(18)
# ============================================================
JOINT_NAMES = {
    0:  "Nose",        1:  "L_Eye",       2:  "R_Eye",
    3:  "L_Ear",       4:  "R_Ear",       5:  "L_Shoulder",
    6:  "R_Shoulder",  7:  "L_Elbow",     8:  "R_Elbow",
    9:  "L_Wrist",    10:  "R_Wrist",    11:  "L_Hip",
   12:  "R_Hip",      13:  "L_Knee",     14:  "R_Knee",
   15:  "L_Ankle",    16:  "R_Ankle",    17:  "Pelvis",
   18:  "Neck",
}

# 2D positions (front view, y up)
JOINT_POS = {
    0:  ( 0.0, 10.0),   # Nose
    1:  ( 0.3,  10.3),  # L_Eye
    2:  (-0.3,  10.3),  # R_Eye
    3:  ( 0.6,  10.0),  # L_Ear
    4:  (-0.6,  10.0),  # R_Ear
    5:  ( 1.2,  8.5),   # L_Shoulder
    6:  (-1.2,  8.5),   # R_Shoulder
    7:  ( 2.0,  6.8),   # L_Elbow
    8:  (-2.0,  6.8),   # R_Elbow
    9:  ( 2.5,  5.2),   # L_Wrist
   10:  (-2.5,  5.2),   # R_Wrist
   11:  ( 0.7,  5.0),   # L_Hip
   12:  (-0.7,  5.0),   # R_Hip
   13:  ( 0.7,  3.0),   # L_Knee
   14:  (-0.7,  3.0),   # R_Knee
   15:  ( 0.7,  1.0),   # L_Ankle
   16:  (-0.7,  1.0),   # R_Ankle
   17:  ( 0.0,  5.2),   # Pelvis
   18:  ( 0.0,  9.0),   # Neck
}

# Kinematic edges (background)
KINEMATIC_EDGES = [
    (0,1),(0,2),(1,3),(2,4),           # face
    (18,0),(17,18),                     # spine
    (18,5),(18,6),                      # shoulders
    (5,7),(7,9),(6,8),(8,10),           # arms
    (17,11),(17,12),                    # hips
    (11,13),(13,15),(12,14),(14,16),    # legs
]

# 12 Bone pairs — matching shape_features.py
BONE_GROUPS = {
    "Chân trái (L_Leg)": {
        "pairs": [(11,13), (13,15)],
        "color": "#BB8FCE",
    },
    "Chân phải (R_Leg)": {
        "pairs": [(12,14), (14,16)],
        "color": "#F0B27A",
    },
    "Tay trái (L_Arm)": {
        "pairs": [(5,7), (7,9)],
        "color": "#45B7D1",
    },
    "Tay phải (R_Arm)": {
        "pairs": [(6,8), (8,10)],
        "color": "#F7DC6F",
    },
    "Thân dọc (Spine)": {
        "pairs": [(17,18), (18,0)],
        "color": "#4ECDC4",
    },
    "Độ rộng (Width)": {
        "pairs": [(11,12), (5,6)],
        "color": "#FF6B6B",
        "style": "--",
    },
}

def plot_bone_pairs():
    fig, ax = plt.subplots(1, 1, figsize=(10, 14))
    ax.set_aspect('equal')
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')

    # 1) Kinematic edges (background)
    for i, j in KINEMATIC_EDGES:
        x1, y1 = JOINT_POS[i]
        x2, y2 = JOINT_POS[j]
        ax.plot([x1,x2], [y1,y2], color='#444466', linewidth=1.5, alpha=0.4, zorder=1)

    # 2) Bone pairs
    legend_patches = []
    bone_idx = 0
    for group_name, info in BONE_GROUPS.items():
        color = info["color"]
        style = info.get("style", "-")
        for i, j in info["pairs"]:
            x1, y1 = JOINT_POS[i]
            x2, y2 = JOINT_POS[j]
            ax.plot([x1,x2], [y1,y2], color=color, linewidth=5, 
                    alpha=0.85, zorder=2, linestyle=style)
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.annotate(
                f"B{bone_idx}", (mx, my),
                fontsize=8, fontweight='bold', color=color, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#1a1a2e', 
                         edgecolor=color, alpha=0.9),
                zorder=5,
            )
            bone_idx += 1
        pairs_str = ", ".join(f"({i},{j})" for i,j in info["pairs"])
        legend_patches.append(mpatches.Patch(color=color, label=f"{group_name}: {pairs_str}"))

    # 3) Joints
    bone_joints = set()
    for grp in BONE_GROUPS.values():
        for a,b in grp["pairs"]:
            bone_joints.add(a)
            bone_joints.add(b)

    for j in range(19):
        x, y = JOINT_POS[j]
        in_bone = j in bone_joints
        color = '#FFFFFF' if in_bone else '#666688'
        size = 140 if in_bone else 80
        ax.scatter(x, y, c=color, s=size, zorder=4, edgecolors='white', linewidths=1.2)
        ha = 'left' if x >= 0 else 'right'
        ax.annotate(
            f"{j}: {JOINT_NAMES[j]}", (x, y),
            textcoords="offset points",
            xytext=(12 if x >= 0 else -12, 5),
            fontsize=8, color='white' if in_bone else '#888888',
            ha=ha, fontweight='bold' if in_bone else 'normal', zorder=6,
        )

    # 4) Legend
    ax.legend(handles=legend_patches, loc='lower center', ncol=2, fontsize=9,
              facecolor='#16213e', edgecolor='white', labelcolor='white',
              framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    # 5) Bone table
    bone_table = "Bone Index → Joint Pair:\n"
    bone_idx = 0
    for group_name, info in BONE_GROUPS.items():
        for i, j in info["pairs"]:
            bone_table += f"  B{bone_idx}: {JOINT_NAMES[i]}({i}) → {JOINT_NAMES[j]}({j})\n"
            bone_idx += 1

    ax.text(-4.2, 4.0, bone_table, fontsize=7.5, color='#CCCCCC',
            fontfamily='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#16213e', 
                     edgecolor='#444466', alpha=0.95), zorder=6)

    ax.set_title("ShapeFeatureExtractor — 12 Bone Pairs (COCO 19 joints)",
                 fontsize=14, fontweight='bold', color='white', pad=15)
    ax.set_xlim(-5.0, 4.5)
    ax.set_ylim(0.0, 11.5)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig("bone_pairs_visual.png", dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print("\n✅ Saved: bone_pairs_visual.png")
    plt.show()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("COCO 19 joints = COCO 17 + Pelvis(17) + Neck(18)")
    print("="*60)
    idx = 0
    for group_name, info in BONE_GROUPS.items():
        print(f"\n  [{group_name}]")
        for i, j in info["pairs"]:
            print(f"    B{idx:>2d}: ({i:>2d},{j:>2d})  {JOINT_NAMES[i]:>12s} → {JOINT_NAMES[j]}")
            idx += 1
    print(f"\n  Total: {idx} bones + 4 global props = {idx+4} features")
    print("="*60)

    plot_bone_pairs()

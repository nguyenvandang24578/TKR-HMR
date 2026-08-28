"""
Visualize 24 SMPL joints với 6 body parts.
Mỗi part một màu, hiển thị tên joint + index.
Chạy: python visualize_skeleton_parts.py
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ============================================================
# 24 SMPL Joint names
# ============================================================
JOINT_NAMES = {
    0:  "Pelvis",
    1:  "L_Hip",
    2:  "R_Hip",
    3:  "Spine1",
    4:  "L_Knee",
    5:  "R_Knee",
    6:  "Spine2",
    7:  "L_Ankle",
    8:  "R_Ankle",
    9:  "Spine3",
    10: "L_Foot",
    11: "R_Foot",
    12: "Neck",
    13: "L_Collar",
    14: "R_Collar",
    15: "Head",
    16: "L_Shoulder",
    17: "R_Shoulder",
    18: "L_Elbow",
    19: "R_Elbow",
    20: "L_Wrist",
    21: "R_Wrist",
    22: "L_Hand",
    23: "R_Hand",
}

# ============================================================
# Approximate 2D positions for visualization (front view)
# (x, y) — y tăng lên trên
# ============================================================
JOINT_POS = {
    0:  ( 0.0,  4.5),   # Pelvis
    1:  ( 0.8,  4.3),   # L_Hip
    2:  (-0.8,  4.3),   # R_Hip
    3:  ( 0.0,  5.5),   # Spine1
    4:  ( 0.8,  2.5),   # L_Knee
    5:  (-0.8,  2.5),   # R_Knee
    6:  ( 0.0,  6.5),   # Spine2
    7:  ( 0.8,  0.8),   # L_Ankle
    8:  (-0.8,  0.8),   # R_Ankle
    9:  ( 0.0,  7.5),   # Spine3
    10: ( 0.8,  0.2),   # L_Foot
    11: (-0.8,  0.2),   # R_Foot
    12: ( 0.0,  8.5),   # Neck
    13: ( 0.7,  8.3),   # L_Collar
    14: (-0.7,  8.3),   # R_Collar
    15: ( 0.0,  9.5),   # Head
    16: ( 1.5,  8.0),   # L_Shoulder
    17: (-1.5,  8.0),   # R_Shoulder
    18: ( 2.3,  6.5),   # L_Elbow
    19: (-2.3,  6.5),   # R_Elbow
    20: ( 2.8,  5.0),   # L_Wrist
    21: (-2.8,  5.0),   # R_Wrist
    22: ( 3.0,  4.5),   # L_Hand
    23: (-3.0,  4.5),   # R_Hand
}

# ============================================================
# Kinematic tree (parent → child edges)
# ============================================================
EDGES = [
    (0, 1), (0, 2), (0, 3),       # Pelvis → Hips, Spine1
    (1, 4), (4, 7), (7, 10),      # L_Hip → L_Knee → L_Ankle → L_Foot
    (2, 5), (5, 8), (8, 11),      # R_Hip → R_Knee → R_Ankle → R_Foot
    (3, 6), (6, 9),               # Spine1 → Spine2 → Spine3
    (9, 12), (9, 13), (9, 14),    # Spine3 → Neck, L_Collar, R_Collar
    (12, 15),                      # Neck → Head
    (13, 16), (16, 18), (18, 20), (20, 22),  # L_Collar → L_Shoulder → L_Elbow → L_Wrist → L_Hand
    (14, 17), (17, 19), (19, 21), (21, 23),  # R_Collar → R_Shoulder → R_Elbow → R_Wrist → R_Hand
]

# ============================================================
# 6 Body Parts — H_init incidence matrix
# ============================================================
PARTS = {
    "Head":      {"joints": [12, 15],                "color": "#FF6B6B", "marker": "^"},
    "Torso":     {"joints": [0, 3, 6, 9, 13, 14],   "color": "#4ECDC4", "marker": "s"},
    "Left Arm":  {"joints": [16, 18, 20, 22],        "color": "#45B7D1", "marker": "o"},
    "Right Arm": {"joints": [17, 19, 21, 23],        "color": "#F7DC6F", "marker": "o"},
    "Left Leg":  {"joints": [1, 4, 7, 10],           "color": "#BB8FCE", "marker": "D"},
    "Right Leg": {"joints": [2, 5, 8, 11],           "color": "#F0B27A", "marker": "D"},
}

# Build joint → part lookup
joint_to_part = {}
for part_name, info in PARTS.items():
    for j in info["joints"]:
        joint_to_part[j] = part_name

def get_edge_color(j1, j2):
    """Edge color = color of the part both joints belong to. 
    If cross-part edge, use gray."""
    p1 = joint_to_part.get(j1)
    p2 = joint_to_part.get(j2)
    if p1 == p2 and p1 is not None:
        return PARTS[p1]["color"]
    else:
        return "#AAAAAA"  # cross-part edge (gray)

# ============================================================
# H_init matrix print
# ============================================================
def print_H_init():
    print("\n" + "="*60)
    print("H_init Incidence Matrix (24 joints × 6 parts)")
    print("="*60)
    part_names = list(PARTS.keys())
    header = f"{'Joint':>20s} | " + " | ".join(f"{p:>9s}" for p in part_names)
    print(header)
    print("-" * len(header))
    for j in range(24):
        row = f"{j:>2d} {JOINT_NAMES[j]:>17s} | "
        for pname in part_names:
            val = 1 if j in PARTS[pname]["joints"] else 0
            mark = "    ■    " if val else "    ·    "
            row += mark + " | "
        print(row)
    print("="*60)

# ============================================================
# Plot
# ============================================================
def plot_skeleton():
    fig, ax = plt.subplots(1, 1, figsize=(10, 14))
    ax.set_aspect('equal')
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')

    # Draw edges
    for j1, j2 in EDGES:
        x1, y1 = JOINT_POS[j1]
        x2, y2 = JOINT_POS[j2]
        color = get_edge_color(j1, j2)
        lw = 3.5 if color != "#AAAAAA" else 2.0
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, alpha=0.85, zorder=1)

    # Draw joints
    for j in range(24):
        x, y = JOINT_POS[j]
        part_name = joint_to_part.get(j, None)
        if part_name:
            color = PARTS[part_name]["color"]
            marker = PARTS[part_name]["marker"]
        else:
            color = "white"
            marker = "o"
        
        ax.scatter(x, y, c=color, s=180, zorder=3, edgecolors='white', linewidths=1.5, marker=marker)
        
        # Label: index + name
        offset_x = 0.25 if x >= 0 else -0.25
        ha = 'left' if x >= 0 else 'right'
        ax.annotate(
            f"{j}: {JOINT_NAMES[j]}", 
            (x, y), 
            textcoords="offset points", 
            xytext=(12 if x >= 0 else -12, 5),
            fontsize=8, 
            color='white',
            ha=ha,
            fontweight='bold',
            zorder=4
        )

    # Legend
    legend_patches = []
    for part_name, info in PARTS.items():
        patch = mpatches.Patch(color=info["color"], label=f"{part_name}: {info['joints']}")
        legend_patches.append(patch)
    
    # Cross-part legend
    legend_patches.append(mpatches.Patch(color="#AAAAAA", label="Cross-part edge"))
    
    ax.legend(
        handles=legend_patches, 
        loc='lower center', 
        ncol=2,
        fontsize=9,
        facecolor='#16213e',
        edgecolor='white',
        labelcolor='white',
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02)
    )

    ax.set_title("SMPL 24 Joints — 6 Body Parts (H_init)", 
                 fontsize=16, fontweight='bold', color='white', pad=15)
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-0.5, 10.5)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig("skeleton_6parts.png", dpi=150, bbox_inches='tight', 
                facecolor=fig.get_facecolor())
    print("\n✅ Saved: skeleton_6parts.png")
    plt.show()

if __name__ == "__main__":
    print_H_init()
    plot_skeleton()

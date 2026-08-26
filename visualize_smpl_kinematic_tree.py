"""
Visualize SMPL 24 joints Kinematic Tree (Parent -> Child).
Chạy: python visualize_smpl_kinematic_tree.py
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

# ============================================================
# SMPL 24 joints Kinematic Tree (PARENT array)
# 0 = root (Pelvis), không có cha (-1)
# ============================================================
PARENT = [
    -1,  # 0  root
    0,   # 1  L_Hip
    0,   # 2  R_Hip
    0,   # 3  Spine1
    1,   # 4  L_Knee
    2,   # 5  R_Knee
    3,   # 6  Spine2
    4,   # 7  L_Ankle
    5,   # 8  R_Ankle
    6,   # 9  Spine3
    7,   # 10 L_Foot
    8,   # 11 R_Foot
    9,   # 12 Neck
    9,   # 13 L_Collar
    9,   # 14 R_Collar
    12,  # 15 Head
    13,  # 16 L_Shoulder
    14,  # 17 R_Shoulder
    16,  # 18 L_Elbow
    17,  # 19 R_Elbow
    18,  # 20 L_Wrist
    19,  # 21 R_Wrist
    20,  # 22 L_Hand
    21,  # 23 R_Hand
]

JOINT_NAMES = {
    0:  "Pelvis (Root)",1:  "L_Hip",      2:  "R_Hip",      3:  "Spine1",
    4:  "L_Knee",     5:  "R_Knee",     6:  "Spine2",     7:  "L_Ankle",
    8:  "R_Ankle",    9:  "Spine3",    10:  "L_Foot",    11:  "R_Foot",
   12:  "Neck",      13:  "L_Collar",  14:  "R_Collar",  15:  "Head",
   16:  "L_Shoulder", 17:  "R_Shoulder", 18:  "L_Elbow",   19:  "R_Elbow",
   20:  "L_Wrist",   21:  "R_Wrist",   22:  "L_Hand",    23:  "R_Hand",
}

# 2D positions (front view, y up)
JOINT_POS = {
    0:  ( 0.0,  4.5),   1:  ( 0.8,  4.3),   2:  (-0.8,  4.3),   3:  ( 0.0,  5.5),
    4:  ( 0.8,  2.5),   5:  (-0.8,  2.5),   6:  ( 0.0,  6.5),   7:  ( 0.8,  0.8),
    8:  (-0.8,  0.8),   9:  ( 0.0,  7.5),  10:  ( 0.8,  0.2),  11:  (-0.8,  0.2),
   12:  ( 0.0,  8.5),  13:  ( 0.7,  8.3),  14:  (-0.7,  8.3),  15:  ( 0.0,  9.5),
   16:  ( 1.5,  8.0),  17:  (-1.5,  8.0),  18:  ( 2.3,  6.5),  19:  (-2.3,  6.5),
   20:  ( 2.8,  5.0),  21:  (-2.8,  5.0),  22:  ( 3.0,  4.5),  23:  (-3.0,  4.5),
}

def plot_kinematic_tree():
    fig, ax = plt.subplots(1, 1, figsize=(12, 14))
    ax.set_aspect('equal')
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')

    # 1) Vẽ mũi tên từ Parent -> Child
    for child_idx, parent_idx in enumerate(PARENT):
        if parent_idx == -1:
            continue
        
        x1, y1 = JOINT_POS[parent_idx]
        x2, y2 = JOINT_POS[child_idx]
        
        # Chọn màu: Nhánh trái (x>0) màu xanh, nhánh phải (x<0) màu cam, giữa trục màu ngọc bích
        if x2 > 0.2:
            color = "#45B7D1" # Trái (Left)
        elif x2 < -0.2:
            color = "#F0B27A" # Phải (Right)
        else:
            color = "#4ECDC4" # Trục giữa (Spine/Head)
            
        arrow = FancyArrowPatch((x1, y1), (x2, y2), 
                                arrowstyle='-|>', mutation_scale=20, 
                                color=color, linewidth=2.5, alpha=0.8, zorder=2)
        ax.add_patch(arrow)

    # 2) Vẽ Joints
    for j in range(24):
        x, y = JOINT_POS[j]
        
        if j == 0:
            color = '#FF6B6B' # Root màu đỏ
            size = 250
            marker = '*'
        else:
            color = '#FFFFFF'
            size = 120
            marker = 'o'
            
        ax.scatter(x, y, c=color, s=size, zorder=4, edgecolors='white', linewidths=1.5, marker=marker)
        
        # Label: index + name
        offset_x = 0.25 if x >= 0 else -0.25
        ha = 'left' if x >= 0 else 'right'
        if j == 0:
            ha = 'center'
            offset_x = 0
            offset_y = -0.4
        else:
            offset_y = 5
            
        ax.annotate(
            f"{j}: {JOINT_NAMES[j]}", 
            (x, y), 
            textcoords="offset points", 
            xytext=(15 if offset_x > 0 else (-15 if offset_x < 0 else 0), offset_y),
            fontsize=8.5, 
            color='white' if j!=0 else '#FF6B6B',
            ha=ha,
            fontweight='bold',
            zorder=6
        )

    # 3) Legend & Tiêu đề
    legend_elements = [
        mpatches.Patch(color='#FF6B6B', label='Root (Pelvis) - Khởi nguồn (Parent=-1)'),
        mpatches.Patch(color='#4ECDC4', label='Trục giữa (Spine, Neck, Head)'),
        mpatches.Patch(color='#45B7D1', label='Nhánh Trái (Left Arm, Left Leg)'),
        mpatches.Patch(color='#F0B27A', label='Nhánh Phải (Right Arm, Right Leg)')
    ]
    ax.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=10,
              facecolor='#16213e', edgecolor='white', labelcolor='white',
              framealpha=0.9, bbox_to_anchor=(0.5, -0.05))

    ax.set_title("SMPL Kinematic Tree (Parent → Child Propagation)", 
                 fontsize=16, fontweight='bold', color='white', pad=20)
    
    # Text box giải thích
    info_text = (
        "Luồng thông tin trong RootChainProp:\n"
        "1. Khớp con nhận feature từ khớp cha trực tiếp qua mảng PARENT.\n"
        "2. h_child = W_self(h_child) + W_parent(h_parent)\n"
        "3. Root (0: Pelvis) được xử lý riêng và cộng dồn (additive)\n"
        "   vào TẤT CẢ các khớp con để cung cấp Global Orientation."
    )
    ax.text(-4.8, 8.5, info_text, fontsize=9, color='#CCCCCC',
            fontfamily='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#16213e', 
                     edgecolor='#444466', alpha=0.95), zorder=6)

    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-0.5, 10.5)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig("smpl_kinematic_tree.png", dpi=150, bbox_inches='tight', 
                facecolor=fig.get_facecolor())
    print("\n✅ Saved: smpl_kinematic_tree.png")
    plt.show()

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Mảng PARENT của 24 khớp SMPL:")
    print("="*50)
    for child, parent in enumerate(PARENT):
        if parent == -1:
            print(f"Khớp {child:>2d} ({JOINT_NAMES[child]:<15s}) | Không có cha (Root)")
        else:
            print(f"Khớp {child:>2d} ({JOINT_NAMES[child]:<15s}) | Cha: {parent:>2d} ({JOINT_NAMES[parent]})")
    print("="*50)
    plot_kinematic_tree()

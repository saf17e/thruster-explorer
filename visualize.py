"""
AUV Thruster Configuration Visualizer
=======================================
3D interactive visualization of thruster placement, thrust vectors,
allocation matrix heatmap, efficiency radar, and coupling matrix.

Usage:
    python visualize.py                        # default 8-thruster config
    python visualize.py --thrusters 10         # 10-thruster config
    python visualize.py --thrusters 8 --set-assignment 0 0 0 0 1 1 1 1
"""

import sys
import os
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, art3d, proj3d
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import (
    Vehicle, Thruster, AllocationMatrix,
    default_thruster_config, DOF_LABELS, DOF_SYMBOLS,
)

# ═══════════════════════════════════════════════════════════════
# Color palette
# ═══════════════════════════════════════════════════════════════

SET_COLORS = [
    '#4fc3f7',  # Set A — light blue
    '#ef5350',  # Set B — red
    '#66bb6a',  # Set C — green
    '#ffa726',  # Set D — orange
    '#ab47bc',  # Set E — purple
    '#26c6da',  # Set F — teal
]

VEHICLE_COLOR = '#37474f'
VEHICLE_ALPHA = 0.12
ARROW_SCALE = 12.0  # length of thrust direction arrows (cm)


# ═══════════════════════════════════════════════════════════════
# 3D vehicle wireframe
# ═══════════════════════════════════════════════════════════════

def _draw_vehicle_box(ax, vehicle):
    """Draw a wireframe rectangular box for the vehicle body."""
    Lh, Bh, Hh = vehicle.bounds()

    # 8 corners of the box
    corners = np.array([
        [ Lh,  Bh,  Hh],
        [ Lh,  Bh, -Hh],
        [ Lh, -Bh,  Hh],
        [ Lh, -Bh, -Hh],
        [-Lh,  Bh,  Hh],
        [-Lh,  Bh, -Hh],
        [-Lh, -Bh,  Hh],
        [-Lh, -Bh, -Hh],
    ])

    # 12 edges (pairs of corner indices)
    edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7),
    ]

    for i, j in edges:
        ax.plot3D(
            *zip(corners[i], corners[j]),
            color=VEHICLE_COLOR, linewidth=1.0, alpha=0.5
        )

    # Semi-transparent faces
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    faces = [
        [corners[0], corners[1], corners[3], corners[2]],  # +X face
        [corners[4], corners[5], corners[7], corners[6]],  # -X face
        [corners[0], corners[1], corners[5], corners[4]],  # +Y face
        [corners[2], corners[3], corners[7], corners[6]],  # -Y face
        [corners[0], corners[2], corners[6], corners[4]],  # +Z face
        [corners[1], corners[3], corners[7], corners[5]],  # -Z face
    ]
    poly = Poly3DCollection(faces, alpha=VEHICLE_ALPHA, facecolor=VEHICLE_COLOR,
                            edgecolor=VEHICLE_COLOR, linewidth=0.5)
    ax.add_collection3d(poly)

    # COM marker
    ax.scatter([0], [0], [0], color='white', s=50, marker='+', linewidths=2, zorder=5)

    # Forward arrow label
    ax.quiver(Lh + 2, 0, 0, 6, 0, 0, color='#e53935', arrow_length_ratio=0.3,
              linewidth=1.5)
    ax.text(Lh + 10, 0, 0, 'FWD (+X)', color='#e53935', fontsize=8, ha='left')


# ═══════════════════════════════════════════════════════════════
# Main 3D thruster plot
# ═══════════════════════════════════════════════════════════════

def plot_3d_thrusters(ax, vehicle, thrusters, assignment=None):
    """
    Plot thrusters as colored dots with thrust-direction arrows.
    """
    _draw_vehicle_box(ax, vehicle)

    for idx, t in enumerate(thrusters):
        color = '#ffffff'
        if assignment is not None and idx < len(assignment):
            color = SET_COLORS[assignment[idx] % len(SET_COLORS)]

        # Thruster position dot
        ax.scatter(
            [t.position[0]], [t.position[1]], [t.position[2]],
            color=color, s=80, edgecolors='black', linewidths=0.8, zorder=10,
            depthshade=False
        )

        # Thrust direction arrow
        ax.quiver(
            t.position[0], t.position[1], t.position[2],
            t.orientation[0] * ARROW_SCALE,
            t.orientation[1] * ARROW_SCALE,
            t.orientation[2] * ARROW_SCALE,
            color=color, arrow_length_ratio=0.2, linewidth=1.8, alpha=0.9
        )

        # Label
        ax.text(
            t.position[0] + 1.5,
            t.position[1] + 1.5,
            t.position[2] + 1.5,
            t.name, color=color, fontsize=8, fontweight='bold'
        )

    # Axis labels and styling
    Lh, Bh, Hh = vehicle.bounds()
    pad = max(Lh, Bh, Hh) * 0.5
    ax.set_xlim(-Lh - pad, Lh + pad)
    ax.set_ylim(-Bh - pad, Bh + pad)
    ax.set_zlim(-Hh - pad, Hh + pad)

    ax.set_xlabel('X (Surge)', fontsize=9, labelpad=8)
    ax.set_ylabel('Y (Sway)', fontsize=9, labelpad=8)
    ax.set_zlabel('Z (Heave)', fontsize=9, labelpad=8)
    ax.set_title('3D Thruster Layout', fontsize=12, fontweight='bold', pad=12)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')
    ax.grid(True, alpha=0.2)

    # Legend for sets
    if assignment is not None:
        unique_sets = sorted(set(assignment))
        for s in unique_sets:
            label = f"Set {chr(65 + s)}"
            t_names = [thrusters[i].name for i, a in enumerate(assignment) if a == s]
            ax.scatter([], [], [], color=SET_COLORS[s % len(SET_COLORS)],
                       s=60, label=f"{label}: {', '.join(t_names)}")
        ax.legend(loc='upper left', fontsize=8, framealpha=0.8)


# ═══════════════════════════════════════════════════════════════
# Top-down 2D view (XY plane)
# ═══════════════════════════════════════════════════════════════

def plot_top_view(ax, vehicle, thrusters, assignment=None):
    """Bird's-eye (top-down) 2D view — X forward, Y starboard."""
    Lh, Bh, _ = vehicle.bounds()

    # Vehicle outline
    rect = plt.Rectangle((-Lh, -Bh), 2 * Lh, 2 * Bh,
                          fill=True, facecolor=VEHICLE_COLOR, alpha=VEHICLE_ALPHA,
                          edgecolor=VEHICLE_COLOR, linewidth=1.5)
    ax.add_patch(rect)

    # COM
    ax.plot(0, 0, '+', color='white', markersize=10, markeredgewidth=2)

    # Forward indicator
    ax.annotate('FWD', xy=(Lh, 0), xytext=(Lh + 5, 0),
                color='#e53935', fontsize=8, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#e53935', lw=1.5))

    for idx, t in enumerate(thrusters):
        color = '#ffffff'
        if assignment is not None and idx < len(assignment):
            color = SET_COLORS[assignment[idx] % len(SET_COLORS)]

        # Only show XY components
        ax.scatter(t.position[0], t.position[1], color=color, s=60,
                   edgecolors='black', linewidths=0.6, zorder=10)

        # Thrust direction (XY projection)
        ox, oy = t.orientation[0], t.orientation[1]
        mag = math.sqrt(ox ** 2 + oy ** 2)
        if mag > 0.05:
            ax.annotate('', xy=(t.position[0] + ox * ARROW_SCALE,
                                t.position[1] + oy * ARROW_SCALE),
                        xytext=(t.position[0], t.position[1]),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
        else:
            # Vertical thruster — draw a circle
            circle = plt.Circle((t.position[0], t.position[1]), 1.5,
                                fill=False, edgecolor=color, linewidth=1.5)
            ax.add_patch(circle)

        ax.text(t.position[0] + 1.5, t.position[1] + 1.5, t.name,
                color=color, fontsize=7, fontweight='bold')

    pad = max(Lh, Bh) * 0.3
    ax.set_xlim(-Lh - pad, Lh + pad)
    ax.set_ylim(-Bh - pad, Bh + pad)
    ax.set_aspect('equal')
    ax.set_xlabel('X (Surge)', fontsize=9)
    ax.set_ylabel('Y (Sway)', fontsize=9)
    ax.set_title('Top View (XY)', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.2)


# ═══════════════════════════════════════════════════════════════
# Side view (XZ plane)
# ═══════════════════════════════════════════════════════════════

def plot_side_view(ax, vehicle, thrusters, assignment=None):
    """Side view — X forward, Z down."""
    Lh, _, Hh = vehicle.bounds()

    rect = plt.Rectangle((-Lh, -Hh), 2 * Lh, 2 * Hh,
                          fill=True, facecolor=VEHICLE_COLOR, alpha=VEHICLE_ALPHA,
                          edgecolor=VEHICLE_COLOR, linewidth=1.5)
    ax.add_patch(rect)

    ax.plot(0, 0, '+', color='white', markersize=10, markeredgewidth=2)

    for idx, t in enumerate(thrusters):
        color = '#ffffff'
        if assignment is not None and idx < len(assignment):
            color = SET_COLORS[assignment[idx] % len(SET_COLORS)]

        ax.scatter(t.position[0], t.position[2], color=color, s=60,
                   edgecolors='black', linewidths=0.6, zorder=10)

        ox, oz = t.orientation[0], t.orientation[2]
        mag = math.sqrt(ox ** 2 + oz ** 2)
        if mag > 0.05:
            ax.annotate('', xy=(t.position[0] + ox * ARROW_SCALE,
                                t.position[2] + oz * ARROW_SCALE),
                        xytext=(t.position[0], t.position[2]),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

        ax.text(t.position[0] + 1.5, t.position[2] + 1.5, t.name,
                color=color, fontsize=7, fontweight='bold')

    pad = max(Lh, Hh) * 0.3
    ax.set_xlim(-Lh - pad, Lh + pad)
    ax.set_ylim(-Hh - pad, Hh + pad)
    ax.set_aspect('equal')
    ax.set_xlabel('X (Surge)', fontsize=9)
    ax.set_ylabel('Z (Heave)', fontsize=9)
    ax.set_title('Side View (XZ)', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.2)


# ═══════════════════════════════════════════════════════════════
# Allocation matrix heatmap
# ═══════════════════════════════════════════════════════════════

def plot_allocation_heatmap(ax, vehicle, thrusters):
    """Heatmap of the 6×n allocation matrix B."""
    am = AllocationMatrix(vehicle, thrusters)
    B = am.B

    cmap = LinearSegmentedColormap.from_list(
        'bwr', ['#2196f3', '#1a1a2e', '#f44336'], N=256
    )
    vmax = np.max(np.abs(B)) or 1.0

    im = ax.imshow(B, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')

    ax.set_xticks(range(len(thrusters)))
    ax.set_xticklabels([t.name for t in thrusters], fontsize=8)
    ax.set_yticks(range(6))
    ax.set_yticklabels(DOF_LABELS, fontsize=9)

    # Annotate cell values
    for i in range(6):
        for j in range(len(thrusters)):
            val = B[i, j]
            txt_color = 'white' if abs(val) > vmax * 0.4 else 'gray'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color=txt_color, fontweight='bold')

    ax.set_title('Allocation Matrix B  (τ = B·u)', fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)


# ═══════════════════════════════════════════════════════════════
# Efficiency radar chart
# ═══════════════════════════════════════════════════════════════

def plot_efficiency_radar(ax, vehicle, thrusters):
    """Radar (spider) chart of per-DOF efficiency scores."""
    am = AllocationMatrix(vehicle, thrusters)
    scores = am.efficiency_scores()

    labels = list(scores.keys())
    values = list(scores.values())
    n = len(labels)

    # Close the polygon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    values_closed = values + [values[0]]
    angles_closed = angles + [angles[0]]

    ax.plot(angles_closed, values_closed, 'o-', color='#4fc3f7', linewidth=2,
            markersize=6)
    ax.fill(angles_closed, values_closed, color='#4fc3f7', alpha=0.2)

    ax.set_xticks(angles)
    ax.set_xticklabels([f'{l}\n{v:.0f}%' for l, v in zip(labels, values)],
                       fontsize=8, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=7, color='gray')
    ax.set_title('DOF Efficiency', fontsize=10, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)


# ═══════════════════════════════════════════════════════════════
# Coupling matrix heatmap
# ═══════════════════════════════════════════════════════════════

def plot_coupling_matrix(ax, vehicle, thrusters):
    """Heatmap of the 6×6 coupling matrix C = B · pinv(B)."""
    am = AllocationMatrix(vehicle, thrusters)
    C = am.coupling_matrix()

    cmap = LinearSegmentedColormap.from_list(
        'coupling', ['#1a237e', '#0d0d1a', '#b71c1c'], N=256
    )

    im = ax.imshow(C, cmap=cmap, vmin=-1, vmax=1, aspect='equal')

    ax.set_xticks(range(6))
    ax.set_xticklabels(DOF_SYMBOLS, fontsize=9, fontweight='bold')
    ax.set_yticks(range(6))
    ax.set_yticklabels(DOF_SYMBOLS, fontsize=9, fontweight='bold')

    for i in range(6):
        for j in range(6):
            val = C[i, j]
            txt_color = 'white' if abs(val) > 0.3 else 'gray'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=8, color=txt_color, fontweight='bold')

    ax.set_title('Coupling  C = B·pinv(B)', fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, pad=0.04)


# ═══════════════════════════════════════════════════════════════
# Full dashboard
# ═══════════════════════════════════════════════════════════════

def visualize_config(vehicle, thrusters, assignment=None, title=None):
    """
    Open an interactive matplotlib dashboard with:
      - 3D thruster layout (rotatable)
      - Top-down 2D view
      - Side 2D view
      - Allocation matrix heatmap
      - Efficiency radar
      - Coupling matrix heatmap
    """
    fig = plt.figure(figsize=(18, 10), facecolor='#0d0d1a')
    fig.patch.set_facecolor('#0d0d1a')

    if title is None:
        title = f'{len(thrusters)}-Thruster AUV Configuration'
    fig.suptitle(title, fontsize=14, fontweight='bold', color='white', y=0.98)

    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.35,
                           left=0.04, right=0.96, top=0.92, bottom=0.06)

    # ── 3D view (spans 2 rows, 2 cols) ─────────────────────
    ax3d = fig.add_subplot(gs[:, 0:2], projection='3d')
    ax3d.set_facecolor('#0d0d1a')
    plot_3d_thrusters(ax3d, vehicle, thrusters, assignment)

    # ── Top view ────────────────────────────────────────────
    ax_top = fig.add_subplot(gs[0, 2])
    ax_top.set_facecolor('#0d0d1a')
    ax_top.tick_params(colors='gray')
    for spine in ax_top.spines.values():
        spine.set_color('gray')
    ax_top.xaxis.label.set_color('white')
    ax_top.yaxis.label.set_color('white')
    ax_top.title.set_color('white')
    plot_top_view(ax_top, vehicle, thrusters, assignment)

    # ── Side view ───────────────────────────────────────────
    ax_side = fig.add_subplot(gs[1, 2])
    ax_side.set_facecolor('#0d0d1a')
    ax_side.tick_params(colors='gray')
    for spine in ax_side.spines.values():
        spine.set_color('gray')
    ax_side.xaxis.label.set_color('white')
    ax_side.yaxis.label.set_color('white')
    ax_side.title.set_color('white')
    plot_side_view(ax_side, vehicle, thrusters, assignment)

    # ── Allocation heatmap ──────────────────────────────────
    ax_alloc = fig.add_subplot(gs[0, 3])
    ax_alloc.set_facecolor('#0d0d1a')
    ax_alloc.tick_params(colors='gray')
    ax_alloc.title.set_color('white')
    plot_allocation_heatmap(ax_alloc, vehicle, thrusters)

    # ── Efficiency radar ────────────────────────────────────
    ax_radar = fig.add_subplot(gs[1, 3], projection='polar')
    ax_radar.set_facecolor('#0d0d1a')
    ax_radar.tick_params(colors='gray')
    ax_radar.title.set_color('white')
    plot_efficiency_radar(ax_radar, vehicle, thrusters)

    # Style 3D axis text
    for axis_obj in [ax3d.xaxis, ax3d.yaxis, ax3d.zaxis]:
        axis_obj.label.set_color('white')
        axis_obj.set_tick_params(colors='gray')
    ax3d.title.set_color('white')

    plt.show(block=False)


# ═══════════════════════════════════════════════════════════════
# Standalone: visualize with coupling matrix
# ═══════════════════════════════════════════════════════════════

def visualize_full(vehicle, thrusters, assignment=None, title=None):
    """
    Two-window visualization: main dashboard + coupling matrix.
    """
    visualize_config(vehicle, thrusters, assignment, title)


# ═══════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Visualize an AUV thruster configuration'
    )
    parser.add_argument(
        '-n', '--thrusters', type=int, default=8, choices=range(6, 13),
        help='Number of thrusters (6–12, default: 8)'
    )
    parser.add_argument(
        '-a', '--set-assignment', type=int, nargs='+', default=None,
        metavar='S',
        help='Set assignment per thruster (e.g. 0 0 0 0 1 1 1 1)'
    )
    parser.add_argument(
        '-L', '--length', type=float, default=60.0,
        help='Vehicle length in cm (default: 60)'
    )
    parser.add_argument(
        '-B', '--breadth', type=float, default=40.0,
        help='Vehicle breadth in cm (default: 40)'
    )
    parser.add_argument(
        '-H', '--height', type=float, default=20.0,
        help='Vehicle height in cm (default: 20)'
    )
    parser.add_argument(
        '--preset', type=str, default=None,
        choices=['horizontal-vertical', 'translation-rotation',
                 'heave-yaw', 'surge-yaw', 'all-separate'],
        help='Use a preset set assignment'
    )

    args = parser.parse_args()

    vehicle = Vehicle(args.length, args.breadth, args.height)
    thrusters = default_thruster_config(args.thrusters, vehicle)

    assignment = args.set_assignment
    title_suffix = ''

    # Handle presets
    if args.preset:
        n = args.thrusters
        if args.preset == 'horizontal-vertical':
            # First 4 horizontal→Set A, rest vertical→Set B
            assignment = [0] * min(4, n) + [1] * max(0, n - 4)
            title_suffix = ' — Horizontal/Vertical split'
        elif args.preset == 'translation-rotation':
            assignment = [0] * (n // 2) + [1] * (n - n // 2)
            title_suffix = ' — Translation/Rotation split'
        elif args.preset == 'heave-yaw':
            # Vertical thrusters → Set A (heave+yaw), horizontal → Set B
            assignment = [1] * min(4, n) + [0] * max(0, n - 4)
            title_suffix = ' — Heave+Yaw / Rest'
        elif args.preset == 'surge-yaw':
            assignment = [0] * min(4, n) + [1] * max(0, n - 4)
            title_suffix = ' — Surge+Yaw / Rest'
        elif args.preset == 'all-separate':
            assignment = list(range(n))
            title_suffix = ' — Each thruster separate'

    if assignment is not None and len(assignment) != len(thrusters):
        print(f"Error: assignment has {len(assignment)} entries but "
              f"there are {len(thrusters)} thrusters")
        sys.exit(1)

    title = f'{len(thrusters)}-Thruster Configuration{title_suffix}'

    print(f"Vehicle:    {vehicle.L} × {vehicle.B} × {vehicle.H} cm")
    print(f"Thrusters:  {len(thrusters)}")
    if assignment:
        n_sets = len(set(assignment))
        print(f"Sets:       {n_sets}")
        for s in sorted(set(assignment)):
            t_names = [thrusters[i].name for i, a in enumerate(assignment) if a == s]
            print(f"  Set {chr(65 + s)}: {', '.join(t_names)}")
    print()

    # Print allocation matrix
    am = AllocationMatrix(vehicle, thrusters)
    print("Allocation Matrix B:")
    header = "         " + "  ".join(f"{t.name:>7s}" for t in thrusters)
    print(header)
    for i in range(6):
        row = "  ".join(f"{am.B[i, j]:7.3f}" for j in range(len(thrusters)))
        print(f"{DOF_LABELS[i]:>8s} {row}")
    print(f"\nRank: {am.rank()}/6  "
          f"{'(fully actuated)' if am.is_fully_actuated() else '(under-actuated)'}")
    print()

    visualize_config(vehicle, thrusters, assignment, title)
    plt.show()  # block so the window stays open when run standalone


if __name__ == '__main__':
    main()

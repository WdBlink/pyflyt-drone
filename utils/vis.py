"""项目文件：轨迹记录与可视化工具

作者: wdblink

说明:
    提供简易的轨迹记录与静态绘图接口，便于观察 A→B 飞行效果。
"""

from __future__ import annotations

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
# 导入 3D 绘图支持
from mpl_toolkits.mplot3d import Axes3D

# 在非 GUI 环境下使用 Agg 后端
import os
if "WSL_DISTRO_NAME" in os.environ or os.environ.get("DISPLAY") is None:
    plt.switch_backend('Agg')


def plot_xy_trajectory(points: List[Tuple[float, float]], a: Tuple[float, float], b: Tuple[float, float]) -> None:
    """绘制二维平面轨迹。

    Args:
        points: 轨迹点列表，元素为 (x, y)。
        a: 起点 A 的 (x, y)。
        b: 终点 B 的 (x, y)。
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    plt.figure(figsize=(6, 6))
    plt.plot(xs, ys, label="trajectory")
    plt.scatter([a[0]], [a[1]], c="green", label="A")
    plt.scatter([b[0]], [b[1]], c="red", label="B")
    plt.axis("equal")
    plt.legend()
    plt.title("A→B XY 轨迹")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    
    # 尝试保存图片，如果无法显示窗口
    try:
        plt.savefig("trajectory_xy.png")
        print("2D 轨迹图已保存至: trajectory_xy.png")
    except Exception as e:
        print(f"保存轨迹图失败: {e}")
        
    # 只有在 GUI 环境下才调用 show
    if plt.get_backend() != 'Agg':
        plt.show()
    else:
        print("使用 Agg 后端，跳过窗口显示。")


def plot_3d_trajectory(points: List[Tuple[float, float, float]], a: Tuple[float, float, float], b: Tuple[float, float, float]) -> None:
    """绘制三维空间轨迹。

    Args:
        points: 轨迹点列表，元素为 (x, y, z)。
        a: 起点 A 的 (x, y, z)。
        b: 终点 B 的 (x, y, z)。
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(xs, ys, zs, label="Flight Path", linewidth=2)
    ax.scatter([a[0]], [a[1]], [a[2]], c="green", marker='^', s=100, label="Start (A)")
    ax.scatter([b[0]], [b[1]], [b[2]], c="red", marker='*', s=100, label="Goal (B)")
    
    # 标出起点和终点的连线（理想直线路径）
    ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], 'k--', alpha=0.3, label="Ideal Path")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Drone 3D Flight Trajectory")
    ax.legend()
    
    # 设置各轴比例一致，避免视觉变形
    all_x = np.array(xs + [a[0], b[0]])
    all_y = np.array(ys + [a[1], b[1]])
    all_z = np.array(zs + [a[2], b[2]])
    
    max_range = np.array([
        np.max(all_x) - np.min(all_x), 
        np.max(all_y) - np.min(all_y), 
        np.max(all_z) - np.min(all_z)
    ]).max() / 2.0

    mid_x = (np.max(all_x) + np.min(all_x)) * 0.5
    mid_y = (np.max(all_y) + np.min(all_y)) * 0.5
    mid_z = (np.max(all_z) + np.min(all_z)) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()

    try:
        plt.savefig("trajectory_3d.png")
        print("3D 轨迹图已保存至: trajectory_3d.png")
    except Exception as e:
        print(f"保存 3D 轨迹图失败: {e}")

    if plt.get_backend() != 'Agg':
        plt.show()
    else:
        plt.close(fig)

import time
import numpy as np
from typing import Optional

class PyBulletDebugOverlay:
    """在终端控制台和 PyBullet GUI 中显示实时飞行数据。"""

    def __init__(self):
        self.last_update = 0
        self.print_interval = 0.1  # 终端打印频率
        self.text_ids = {}  # 存储 PyBullet debug text ID
        self.p = None
        try:
            import pybullet as p
            self.p = p
        except ImportError:
            pass

    def update(self, altitude, airspeed, thrust, targets_remaining, target_dist):
        """更新并打印飞行遥测数据。
        
        Args:
            altitude: 高度 (m)
            airspeed: 空速 (m/s)
            thrust: 推力 (0-1)
            targets_remaining: 剩余航点数
            target_dist: 距离当前目标距离 (m)
        """
        now = time.time()
        
        # 1. 终端输出 (限制频率以避免闪烁)
        if now - self.last_update >= self.print_interval:
            self.last_update = now
            tgt_dist_str = f"{target_dist:.1f}" if target_dist is not None else "nan"
            print(
                f"\rAlt: {altitude:.1f} m | "
                f"Spd: {airspeed:.1f} m/s | "
                f"Thr: {thrust:.2f} | "
                f"Tgt: {targets_remaining} | "
                f"Dist: {tgt_dist_str} m    ",
                end="",
                flush=True
            )

        # 2. PyBullet GUI HUD 显示 (每帧更新以保持平滑)
        if self.p and self.p.isConnected():
            self._draw_hud(altitude, airspeed, thrust, targets_remaining, target_dist)

    def _draw_hud(self, altitude, airspeed, thrust, targets_remaining, target_dist):
        """在 PyBullet 窗口下方绘制跟随摄像机的 HUD 文本。"""
        try:
            # 获取摄像机参数
            # result: [width, height, viewMatrix, projectionMatrix, upAxis, forwardAxis, horizontalAxis, yaw, pitch, dist, target]
            cam_info = self.p.getDebugVisualizerCamera()
            if not cam_info:
                return
                
            # 解析 View Matrix (World -> Camera)
            view_mat = np.array(cam_info[2]).reshape(4, 4, order='F')
            # Camera -> World
            cam_world = np.linalg.inv(view_mat)
            
            cam_pos = cam_world[:3, 3]
            # OpenGL 坐标系: Forward 是 -Z
            cam_right = cam_world[:3, 0]
            cam_up = cam_world[:3, 1]
            cam_fwd = -cam_world[:3, 2]
            
            # 计算文本位置 (放置在摄像机前方固定距离)
            dist = 0.5  # 距离摄像机 0.5m
            # 调整偏移量以放置在窗口底部 (根据经验值微调)
            # 向下偏移 (y轴负方向) 和 向左/向右微调
            # 假设 FOV 约为 60-90 度
            base_pos = cam_pos + cam_fwd * dist
            
            # 定义显示的文本行
            tgt_dist_str = f"{target_dist:.1f}" if target_dist is not None else "nan"
            lines = [
                f"Alt: {altitude:.1f} m   Spd: {airspeed:.1f} m/s   Thr: {thrust:.2f}",
                f"Targets: {targets_remaining}   Dist: {tgt_dist_str} m"
            ]
            
            # 绘制每一行
            # 起始位置: 底部居中或偏左
            # 0.5m 距离处，屏幕高度约为 0.5 * tan(fov/2) * 2
            # 假设 vertical FOV 60deg -> tan(30) = 0.577
            # height ~ 0.577m. bottom ~ -0.28m
            start_y = -0.20 # 稍微靠下
            line_spacing = 0.04
            
            for i, text in enumerate(lines):
                # 计算当前行的 3D 坐标
                # 从下往上排，或者从上往下
                # 这里我们放到底部，所以第一行在最下面? 或者第一行在上面
                # 我们让 lines[0] 是上面一行，lines[1] 是下面一行
                current_y = start_y - i * line_spacing
                
                # 稍微向左偏移以居中 (简单估算: 每个字符宽度)
                # 0.5m 距离，文字大小 1.0 对应的物理大小
                text_len = len(text)
                start_x = -0.01 * text_len * 0.4 # 粗略居中
                
                pos = base_pos + cam_up * current_y + cam_right * start_x
                
                # 使用 replaceItemUniqueId 避免闪烁
                old_id = self.text_ids.get(i, -1)
                new_id = self.p.addUserDebugText(
                    text=text,
                    textPosition=pos,
                    textColorRGB=[0, 0, 0], # 黑色字体，对比度通常较好 (除非背景是黑的)
                    textSize=1.5,
                    lifeTime=0,
                    replaceItemUniqueId=old_id
                )
                self.text_ids[i] = new_id
                
        except Exception:
            pass

    def clear(self) -> None:
        """清理已创建的调试项。"""
        if self.p and self.p.isConnected():
            for item_id in self.text_ids.values():
                try:
                    self.p.removeUserDebugItem(item_id)
                except Exception:
                    pass
        self.text_ids = {}

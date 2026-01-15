#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# 文件作者: wdblink
# 文件说明:
#   离线测试 FastSAM 分割效果的脚本。遍历指定目录中的 RGB 图像，使用 FastSAM 执行实例分割，
#   并将生成的掩码保存到输出目录。支持 bboxes / points / texts 提示。
#
# 设计与风格:
#   - 使用 Google 风格注释与类型标注
#   - 结构清晰，可扩展性与可维护性优先
#   - 在 WSL2 Ubuntu 环境下工作
#
# 使用示例:
#   python scripts/yoloe_offline_seg.py \
#       --images_dir /home/wdblink/Project/pyflyt-drone/eval_frames/objlock/ep_000 \
#       --output_dir /home/wdblink/Project/pyflyt-drone/eval_frames/objlock/ep_000_seg \
#       --weights FastSAM-s.pt \
#       --device cpu \
#       --retina_masks \
#       --imgsz 1024 \
#       --conf 0.4 \
#       --iou 0.9 \
#       --texts "a photo of a yellow duck"
#
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence, Tuple, List

import numpy as np
from PIL import Image


def _safe_import_ultralytics():
    """安全导入 ultralytics 库。

    Returns:
        type: FastSAM 类
    """
    try:
        from ultralytics import FastSAM  # type: ignore
    except Exception as exc:
        print(f"[ERROR] 无法导入 ultralytics.FastSAM，请先安装: pip install ultralytics\n{exc}")
        raise
    return FastSAM


def _load_model(weights: str) -> object:
    """加载 FastSAM 分割模型。

    Args:
        weights (str): 权重文件路径或模型名（例如 'FastSAM-s.pt'）

    Returns:
        object: 已加载的模型实例
    """
    FastSAM = _safe_import_ultralytics()
    return FastSAM(weights)


def _list_images(images_dir: str) -> List[str]:
    """列举目录中的图像文件。

    Args:
        images_dir (str): 图像目录

    Returns:
        list[str]: 图像文件的绝对路径列表
    """
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    files = []
    try:
        for name in sorted(os.listdir(images_dir)):
            if name.lower().endswith(exts):
                files.append(os.path.join(images_dir, name))
    except Exception as exc:
        print(f"[ERROR] 读取目录失败: {images_dir}\n{exc}")
    return files


def _combine_masks_to_binary(results_obj) -> Optional[np.ndarray]:
    """将 FastSAM 的多实例掩码合并为单通道二值图。

    Args:
        results_obj: ultralytics 的结果对象 (results[0])

    Returns:
        Optional[np.ndarray]: 合并后的 (H, W) 二值掩码；失败返回 None
    """
    try:
        masks = getattr(results_obj, "masks", None)
        if masks is None or masks.data is None:
            return None
        data = masks.data  # (N, H, W)
        if hasattr(data, "cpu"):
            data = data.cpu().numpy()
        # 合并所有实例为单通道
        if data.ndim == 3:
            m = (data > 0.5).any(axis=0).astype(np.uint8)  # (H, W)
        elif data.ndim == 2:
            m = (data > 0.5).astype(np.uint8)
        else:
            return None
        return m
    except Exception as exc:
        print(f"[WARN] 合并掩码失败: {exc}")
        return None


def _save_mask(output_path: str, mask: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> bool:
    """保存二值掩码为 PNG 文件。

    Args:
        output_path (str): 输出文件路径
        mask (np.ndarray): 二值掩码 (H, W)
        target_size (Optional[Tuple[int, int]]): 目标尺寸 (W, H)，若提供会进行重采样

    Returns:
        bool: 保存是否成功
    """
    try:
        img = Image.fromarray((mask * 255).astype(np.uint8))
        if target_size is not None and img.size != target_size:
            img = img.resize(target_size, resample=Image.NEAREST)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path)
        return True
    except Exception as exc:
        print(f"[WARN] 保存掩码失败: {output_path}\n{exc}")
        return False


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """解析命令行参数。

    Args:
        argv (Optional[Sequence[str]]): 可选的参数列表

    Returns:
        argparse.Namespace: 参数命名空间
    """
    parser = argparse.ArgumentParser(description="离线 FastSAM 分割测试脚本")
    parser.add_argument("--images_dir", type=str, required=True, help="离线 RGB 图像目录")
    parser.add_argument("--output_dir", type=str, required=True, help="分割掩码输出目录")
    parser.add_argument("--weights", type=str, default="FastSAM-s.pt", help="FastSAM 权重文件路径或模型名（默认 FastSAM-s.pt）")
    parser.add_argument("--device", type=str, default="cpu", help="推理设备（默认 cpu）")
    parser.add_argument("--retina_masks", action="store_true", help="启用 retina_masks（默认关闭）")
    parser.add_argument("--imgsz", type=int, default=1024, help="推理分辨率 imgsz（默认 1024）")
    parser.add_argument("--conf", type=float, default=0.4, help="置信度阈值 conf（默认 0.4）")
    parser.add_argument("--iou", type=float, default=0.9, help="NMS iou 阈值（默认 0.9）")
    parser.add_argument("--bboxes", type=float, nargs=4, default=None, help="bbox 提示 [x1 y1 x2 y2]（可选）")
    parser.add_argument("--points", type=float, nargs=2, action="append", default=None, help="点提示 [x y]，可重复多次（可选）")
    parser.add_argument("--labels", type=int, nargs="*", default=None, help="点提示标签（与 points 数量一致，1=正/0=负）")
    parser.add_argument("--texts", type=str, default=None, help="文本提示（可选，例如 'a photo of a yellow duck'）")
    return parser.parse_args(argv)


def main() -> None:
    """脚本入口。遍历目录图像，执行分割并保存掩码。"""
    args = parse_args()
    if args.weights and os.path.sep in args.weights and not os.path.exists(args.weights):
        print(f"[ERROR] 权重文件不存在: {args.weights}")
        sys.exit(1)
    model = _load_model(args.weights)

    images = _list_images(args.images_dir)
    if not images:
        print(f"[ERROR] 图像目录为空或不可读: {args.images_dir}")
        sys.exit(1)

    points = args.points or None
    labels = args.labels or None
    if points is not None and labels is not None and len(labels) != len(points):
        print("[ERROR] labels 数量必须与 points 数量一致")
        sys.exit(1)

    total = 0
    saved = 0
    for img_path in images:
        total += 1
        try:
            results = model(
                img_path,
                device=args.device,
                retina_masks=bool(args.retina_masks),
                imgsz=int(args.imgsz),
                conf=float(args.conf),
                iou=float(args.iou),
                bboxes=list(args.bboxes) if args.bboxes is not None else None,
                points=points,
                labels=labels,
                texts=args.texts,
            )
            if not results:
                mask = None
            else:
                mask = _combine_masks_to_binary(results[0])
        except Exception as exc:
            print(f"[WARN] 推理失败: {img_path}\n{exc}")
            mask = None
        if mask is None:
            print(f"[INFO] 未生成掩码: {os.path.basename(img_path)}")
            continue

        # 保存到输出目录，文件名保持一致但扩展名改为 .png
        out_name = os.path.basename(os.path.splitext(img_path)[0]) + "_mask.png"
        out_path = os.path.join(args.output_dir, out_name)
        # 目标尺寸按原图保存
        with Image.open(img_path) as im:
            W, H = im.size
        if _save_mask(out_path, mask, target_size=(W, H)):
            saved += 1
            print(f"[OK] 保存掩码: {out_path}")

    print(f"[DONE] 处理完成: 总计 {total}, 成功保存 {saved}")


if __name__ == "__main__":
    main()

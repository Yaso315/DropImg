#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import math
import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
OVERLAY_HEX = "#25303F"
OVERLAY_OPACITY = 0.90
GLOW_HEX = "#B9B8AF"
CANVAS_SIZE = 32
CENTER_KEEP_SIZE = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="自生成 YOLO classify 数据集（train/val）。"
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(r"E:\HuiAi\YOLOv8\5.pick\拾取物列表_CN"),
        help="前景源图片目录（会递归扫描）。",
    )
    parser.add_argument(
        "--background-dir",
        type=Path,
        default=Path(r"E:\HuiTask\更好的原神\数据源\background"),
        help="背景图目录（会递归扫描）。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"E:\HuiAi\YOLOv8\5.pick\dataset_classify_auto"),
        help="输出数据集目录。",
    )
    parser.add_argument("--train-count", type=int, default=160, help="每类训练集数量。")
    parser.add_argument("--val-count", type=int, default=40, help="每类验证集数量。")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，便于复现。",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=20,
        help="按源文件并发生成的线程数。",
    )
    return parser.parse_args()


def list_images(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"颜色格式错误: {hex_color}")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def read_image_unicode(path: Path, flags: int) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    if img is None:
        raise RuntimeError(f"读取图片失败: {path}")
    return img


def write_png_unicode(path: Path, image_bgr: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError(f"编码 PNG 失败: {path}")
    encoded.tofile(str(path))


def add_left_white_glow(
    image_bgr: np.ndarray, rng: random.Random, strength: float = 0.2
) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    # 上/下边缘最亮，向图像内侧衰减；同时从左向右衰减
    strip_len = rng.randint(max(12, w // 3), max(18, w - 2))
    edge_depth = rng.uniform(2.4, 4.2)

    y = np.arange(h, dtype=np.float32)
    top_grad = np.exp(-y / edge_depth)
    bottom_grad = np.exp(-(h - 1 - y) / edge_depth)
    y_grad = np.maximum(top_grad, bottom_grad)[:, None]

    x = np.arange(w, dtype=np.float32)
    x_falloff = np.exp(-x / rng.uniform(7.0, 11.0))
    x_falloff[x >= strip_len] *= 0.18
    x_falloff = x_falloff[None, :]

    glow = y_grad * x_falloff
    glow = cv2.GaussianBlur(glow, (0, 0), sigmaX=1.6, sigmaY=0.8)
    glow = np.clip(glow * strength, 0.0, 1.0)[..., None]

    r, g, b = hex_to_rgb(GLOW_HEX)
    glow_color = np.array([b, g, r], dtype=np.float32).reshape(1, 1, 3)
    base = image_bgr.astype(np.float32)
    out = base * (1.0 - glow) + glow_color * glow
    return np.clip(out, 0, 255).astype(np.uint8)


def build_background(
    bg_path: Path, rng: random.Random, overlay_opacity: float
) -> np.ndarray:
    bg = read_image_unicode(bg_path, cv2.IMREAD_COLOR)
    h, w = bg.shape[:2]
    if w < CANVAS_SIZE or h < CANVAS_SIZE:
        scale = max(CANVAS_SIZE / w, CANVAS_SIZE / h)
        bg = cv2.resize(
            bg,
            (math.ceil(w * scale), math.ceil(h * scale)),
            interpolation=cv2.INTER_CUBIC,
        )
        h, w = bg.shape[:2]

    left = rng.randint(0, w - CANVAS_SIZE)
    top = rng.randint(0, h - CANVAS_SIZE)
    bg = bg[top : top + CANVAS_SIZE, left : left + CANVAS_SIZE].copy()

    # OpenCV 使用 BGR 顺序
    r, g, b = hex_to_rgb(OVERLAY_HEX)
    overlay = np.full((CANVAS_SIZE, CANVAS_SIZE, 3), (b, g, r), dtype=np.uint8)
    alpha = float(overlay_opacity)
    blended = cv2.addWeighted(bg, 1.0 - alpha, overlay, alpha, 0.0)
    return blended


def resize_foreground(src_path: Path, out_size: int) -> np.ndarray:
    fg = read_image_unicode(src_path, cv2.IMREAD_UNCHANGED)
    if fg.ndim == 2:
        fg = cv2.cvtColor(fg, cv2.COLOR_GRAY2BGRA)
    elif fg.shape[2] == 3:
        fg = cv2.cvtColor(fg, cv2.COLOR_BGR2BGRA)
    elif fg.shape[2] != 4:
        raise RuntimeError(f"不支持的前景通道数: {src_path}")
    return cv2.resize(fg, (out_size, out_size), interpolation=cv2.INTER_LANCZOS4)


def random_position_with_center_constraint(
    fg_size: int, rng: random.Random
) -> Tuple[int, int]:
    half_keep = CENTER_KEEP_SIZE / 2.0
    center = fg_size / 2.0

    x_min = math.ceil(-(center - half_keep))
    x_max = math.floor(CANVAS_SIZE - (center + half_keep))
    y_min = x_min
    y_max = x_max

    if x_min > x_max or y_min > y_max:
        raise ValueError(f"前景尺寸 {fg_size} 无法满足中心 {CENTER_KEEP_SIZE} 约束。")

    x = rng.randint(x_min, x_max)
    y = rng.randint(y_min, y_max)
    return x, y


def compose_sample(
    src_path: Path,
    bg_paths: Sequence[Path],
    scale_big: bool,
    rng: random.Random,
) -> np.ndarray:
    if scale_big:
        fg_size = rng.randint(52, 54)
        overlay_opacity = OVERLAY_OPACITY
        fg_opacity_scale = 1.0
        add_glow = rng.random() < 0.60
    else:
        fg_size = rng.randint(32, 51)
        overlay_opacity = rng.uniform(0.60, 0.95)
        fg_opacity_scale = overlay_opacity
        add_glow = True

    bg_path = rng.choice(bg_paths)
    bg = build_background(bg_path, rng, overlay_opacity=overlay_opacity)

    fg = resize_foreground(src_path, fg_size)
    if fg_opacity_scale < 1.0:
        fg_alpha = fg[:, :, 3].astype(np.float32) * float(fg_opacity_scale)
        fg[:, :, 3] = np.clip(fg_alpha, 0, 255).astype(np.uint8)
    pos_x, pos_y = random_position_with_center_constraint(fg_size, rng)

    canvas_h = CANVAS_SIZE
    canvas_w = CANVAS_SIZE
    fg_h, fg_w = fg.shape[:2]

    x1 = max(0, pos_x)
    y1 = max(0, pos_y)
    x2 = min(canvas_w, pos_x + fg_w)
    y2 = min(canvas_h, pos_y + fg_h)
    if x1 >= x2 or y1 >= y2:
        return bg

    fg_x1 = x1 - pos_x
    fg_y1 = y1 - pos_y
    fg_x2 = fg_x1 + (x2 - x1)
    fg_y2 = fg_y1 + (y2 - y1)

    fg_roi = fg[fg_y1:fg_y2, fg_x1:fg_x2]
    bg_roi = bg[y1:y2, x1:x2].astype(np.float32)

    alpha = (fg_roi[:, :, 3:4].astype(np.float32)) / 255.0
    fg_rgb = fg_roi[:, :, :3].astype(np.float32)
    out_roi = fg_rgb * alpha + bg_roi * (1.0 - alpha)
    bg[y1:y2, x1:x2] = np.clip(out_roi, 0, 255).astype(np.uint8)
    if add_glow:
        glow_strength = rng.uniform(0.14, 0.26)
        bg = add_left_white_glow(bg, rng, strength=glow_strength)
    return bg


def class_name_from_file(path: Path) -> str:
    return path.stem.strip()


def ensure_unique_class_names(files: Iterable[Path]) -> None:
    seen = {}
    duplicates = []
    for p in files:
        cls = class_name_from_file(p)
        if cls in seen:
            duplicates.append((cls, seen[cls], p))
        else:
            seen[cls] = p
    if duplicates:
        msg_lines = ["检测到重复类别名（文件名相同），请先处理冲突："]
        for cls, p1, p2 in duplicates[:20]:
            msg_lines.append(f"- {cls}: {p1} <-> {p2}")
        if len(duplicates) > 20:
            msg_lines.append(f"... 其余 {len(duplicates) - 20} 项未展示")
        raise ValueError("\n".join(msg_lines))


def generate_for_one_class(
    src_path: Path,
    bg_paths: Sequence[Path],
    out_root: Path,
    train_count: int,
    val_count: int,
    rng: random.Random,
) -> None:
    class_name = class_name_from_file(src_path)
    train_dir = out_root / "train" / class_name
    val_dir = out_root / "val" / class_name
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    train_big_count = train_count // 2
    val_big_count = val_count // 2

    train_flags = [True] * train_big_count + [False] * (train_count - train_big_count)
    val_flags = [True] * val_big_count + [False] * (val_count - val_big_count)
    rng.shuffle(train_flags)
    rng.shuffle(val_flags)

    for i, is_big in enumerate(train_flags, start=1):
        out_img = compose_sample(src_path, bg_paths, is_big, rng)
        write_png_unicode(train_dir / f"{i:04d}.png", out_img)

    for i, is_big in enumerate(val_flags, start=1):
        out_img = compose_sample(src_path, bg_paths, is_big, rng)
        write_png_unicode(val_dir / f"{i:04d}.png", out_img)


def main() -> None:
    args = parse_args()

    if not args.source_dir.exists():
        raise FileNotFoundError(f"source_dir 不存在: {args.source_dir}")
    if not args.background_dir.exists():
        raise FileNotFoundError(f"background_dir 不存在: {args.background_dir}")
    if args.train_count < 0 or args.val_count < 0:
        raise ValueError("train_count / val_count 必须 >= 0")

    src_files = list_images(args.source_dir)
    bg_files = list_images(args.background_dir)
    if not src_files:
        raise RuntimeError(f"未找到源图: {args.source_dir}")
    if not bg_files:
        raise RuntimeError(f"未找到背景图: {args.background_dir}")

    ensure_unique_class_names(src_files)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, args.workers)

    total = len(src_files)
    index_and_src = list(enumerate(src_files, start=1))

    def process_one(item: Tuple[int, Path]) -> Tuple[int, str]:
        idx, src = item
        # 每个源文件使用独立随机源，避免并发共享状态
        local_rng = random.Random(args.seed + idx * 1000003)
        generate_for_one_class(
            src_path=src,
            bg_paths=bg_files,
            out_root=args.output_dir,
            train_count=args.train_count,
            val_count=args.val_count,
            rng=local_rng,
        )
        return idx, src.stem

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_one, item): item for item in index_and_src}
        for future in concurrent.futures.as_completed(futures):
            idx, name = future.result()
            print(f"[{idx}/{total}] 完成: {name}")

    print(f"\n数据集生成完成: {args.output_dir}")


if __name__ == "__main__":
    main()

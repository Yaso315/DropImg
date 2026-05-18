import cv2
import numpy as np
import os
from pathlib import Path

def crop_center(image, crop_size=(64, 64)):
    """截取图片中心区域"""
    height, width = image.shape[:2]
    left = (width - crop_size[0]) // 2
    top = (height - crop_size[1]) // 2
    right = left + crop_size[0]
    bottom = top + crop_size[1]
    return image[top:bottom, left:right]

def seamless_blend(base_img, overlay_img, center_pos):
    """使用OpenCV的seamlessClone实现无缝融合"""
    if overlay_img.shape[2] < 4:
        # 如果overlay没有alpha通道，直接覆盖
        h, w = overlay_img.shape[:2]
        x_offset, y_offset = center_pos
        # 确保不会越界
        x_end = min(x_offset + w, base_img.shape[1])
        y_end = min(y_offset + h, base_img.shape[0])
        w_actual = x_end - x_offset
        h_actual = y_end - y_offset
        
        result = base_img.copy()
        result[y_offset:y_offset+h_actual, x_offset:x_offset+w_actual] = overlay_img[0:h_actual, 0:w_actual]
        return result
    
    # 提取overlay的alpha通道
    alpha = overlay_img[:, :, 3] / 255.0
    overlay_rgb = overlay_img[:, :, :3]
    
    # 获取要覆盖的位置
    h, w = overlay_rgb.shape[:2]
    x_offset, y_offset = center_pos
    
    # 确保不会越界
    x_end = min(x_offset + w, base_img.shape[1])
    y_end = min(y_offset + h, base_img.shape[0])
    w_actual = x_end - x_offset
    h_actual = y_end - y_offset
    
    # 更新实际使用的overlay部分
    overlay_rgb = overlay_rgb[0:h_actual, 0:w_actual]
    alpha = alpha[0:h_actual, 0:w_actual]
    
    # 创建mask
    mask = (alpha > 0.1).astype(np.uint8) * 255
    
    # 获取mask的轮廓作为融合中心点
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(largest_contour)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00']) + x_offset
            cy = int(moments['m01'] / moments['m00']) + y_offset
        else:
            cx = x_offset + w_actual // 2
            cy = y_offset + h_actual // 2
    else:
        cx = x_offset + w_actual // 2
        cy = y_offset + h_actual // 2
    
    # 确保中心点在图像范围内
    cx = max(1, min(cx, base_img.shape[1]-1))
    cy = max(1, min(cy, base_img.shape[0]-1))
    
    # 使用泊松融合，但先检查是否会导致越界
    try:
        if (w_actual > 0 and h_actual > 0 and 
            cx >= w_actual//2 and cx + w_actual//2 <= base_img.shape[1] and
            cy >= h_actual//2 and cy + h_actual//2 <= base_img.shape[0]):
            # 验证mask尺寸与overlay一致
            if mask.shape[:2] == overlay_rgb.shape[:2]:
                result = cv2.seamlessClone(overlay_rgb, base_img, mask, (cx, cy), cv2.NORMAL_CLONE)
            else:
                # 如果尺寸不一致，使用alpha混合
                result = base_img.copy()
                roi = result[y_offset:y_offset+h_actual, x_offset:x_offset+w_actual]
                for c in range(3):
                    roi[:, :, c] = (1 - alpha) * roi[:, :, c] + alpha * overlay_rgb[:, :, c]
                result[y_offset:y_offset+h_actual, x_offset:x_offset+w_actual] = roi
        else:
            # 如果可能导致越界，使用alpha混合
            result = base_img.copy()
            roi = result[y_offset:y_offset+h_actual, x_offset:x_offset+w_actual]
            for c in range(3):
                roi[:, :, c] = (1 - alpha) * roi[:, :, c] + alpha * overlay_rgb[:, :, c]
            result[y_offset:y_offset+h_actual, x_offset:x_offset+w_actual] = roi
    except:
        # 如果seamlessClone失败，使用alpha混合
        result = base_img.copy()
        roi = result[y_offset:y_offset+h_actual, x_offset:x_offset+w_actual]
        for c in range(3):
            roi[:, :, c] = (1 - alpha) * roi[:, :, c] + alpha * overlay_rgb[:, :, c]
        result[y_offset:y_offset+h_actual, x_offset:x_offset+w_actual] = roi
        
    return result

def alpha_blend(base_img, overlay_img, pos):
    """使用alpha混合叠加图像"""
    x_offset, y_offset = pos
    h, w = overlay_img.shape[:2]
    
    # 确保不会越界
    x_end = min(x_offset + w, base_img.shape[1])
    y_end = min(y_offset + h, base_img.shape[0])
    w_actual = x_end - x_offset
    h_actual = y_end - y_offset
    
    # 更新实际使用的overlay部分
    overlay_img = overlay_img[0:h_actual, 0:w_actual]
    
    # 如果overlay有alpha通道
    if overlay_img.shape[2] == 4:
        alpha = overlay_img[:, :, 3] / 255.0
        overlay_rgb = overlay_img[:, :, :3]
        
        roi = base_img[y_offset:y_offset+h_actual, x_offset:x_offset+w_actual]
        for c in range(3):
            roi[:, :, c] = (1 - alpha) * roi[:, :, c] + alpha * overlay_rgb[:, :, c]
        
        result = base_img.copy()
        result[y_offset:y_offset+h_actual, x_offset:x_offset+w_actual] = roi
        return result
    else:
        # 如果没有alpha通道，直接复制
        result = base_img.copy()
        result[y_offset:y_offset+h_actual, x_offset:x_offset+w_actual] = overlay_img
        return result

def main():
    # 定义路径
    background_folder = Path("background")
    backborder_path = Path("v2LightBg.png")
    relic_folder = Path("droplist/syw")
    output_folder = Path("output")
    
    # 创建输出文件夹
    output_folder.mkdir(exist_ok=True)
    
    # 获取background文件夹中的前5张图片
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}
    image_files = [f for f in background_folder.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    # 按文件名排序并取前5张
    image_files = sorted(image_files)[:5]
    
    print(f"找到 {len(image_files)} 张图片")
    
    # 加载backBorder.png
    if not backborder_path.exists():
        print(f"错误: {backborder_path} 不存在")
        return
    
    backborder = cv2.imread(str(backborder_path), cv2.IMREAD_UNCHANGED)
    if backborder is None:
        print(f"错误: 无法加载 {backborder_path}")
        return
    
    # 如果backborder没有alpha通道，转换为带alpha通道
    if backborder.shape[2] < 4:
        backborder = cv2.cvtColor(backborder, cv2.COLOR_BGR2BGRA)
    
    print(f"backBorder.png 尺寸: {backborder.shape[:2][::-1]}")
    
    # 获取圣遗物文件夹中的第一张图片
    relic_files = [f for f in relic_folder.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not relic_files:
        print(f"错误: {relic_folder} 中没有找到图片")
        return
    
    # 按文件名排序并取第一张
    relic_files = sorted(relic_files)
    first_relic_path = relic_files[0]
    print(f"使用圣遗物图片: {first_relic_path.name}")
    
    # 加载圣遗物图片
    relic_image = cv2.imread(str(first_relic_path), cv2.IMREAD_UNCHANGED)
    if relic_image is None:
        print(f"错误: 无法加载圣遗物图片 {first_relic_path}")
        return
    
    # 如果圣遗物图片没有alpha通道，转换为带alpha通道
    if len(relic_image.shape) == 3 and relic_image.shape[2] == 3:
        relic_image = cv2.cvtColor(relic_image, cv2.COLOR_BGR2BGRA)
    elif len(relic_image.shape) == 2:  # 灰度图
        relic_image = cv2.cvtColor(relic_image, cv2.COLOR_GRAY2BGRA)
    
    print(f"圣遗物图片尺寸: {relic_image.shape[:2][::-1]}")
    
    # 处理每张图片
    for idx, img_path in enumerate(image_files, 1):
        print(f"\n处理第 {idx} 张图片: {img_path.name}")
        
        # 打开图片
        base_image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if base_image is None:
            print(f"错误: 无法加载 {img_path.name}")
            continue
            
        # 如果是3通道图片，转换为4通道
        if len(base_image.shape) == 3 and base_image.shape[2] == 3:
            base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2BGRA)
        elif len(base_image.shape) == 2:  # 灰度图
            base_image = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGRA)
            
        print(f"原始图片尺寸: {base_image.shape[:2][::-1]}")
        
        # 截取中心64x64区域
        center_crop = crop_center(base_image, (64, 64))
        print(f"截取中心区域后尺寸: {center_crop.shape[:2][::-1]}")
        
        # 调整backBorder大小以匹配基础图（如果需要）
        if backborder.shape[:2] != center_crop.shape[:2]:
            backborder_resized = cv2.resize(backborder, (center_crop.shape[1], center_crop.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        else:
            backborder_resized = backborder
        
        # 将backBorder叠加到基础图上
        result = alpha_blend(center_crop, backborder_resized, (0, 0))
        
        # 调整圣遗物图片大小，使其比基础图小一些（例如70%的大小）
        relic_scale = 0.7  # 缩放比例，可以调整为0.5-0.8之间的值
        relic_new_size = (int(center_crop.shape[1] * relic_scale), int(center_crop.shape[0] * relic_scale))
        relic_resized = cv2.resize(relic_image, relic_new_size, interpolation=cv2.INTER_LANCZOS4)
        
        # 计算居中位置并添加偏移（向右5像素，向下5像素）
        paste_x = (center_crop.shape[1] - relic_new_size[0]) // 2 - 3
        paste_y = (center_crop.shape[0] - relic_new_size[1]) // 2 + 1
        
        # 确保粘贴位置在图像范围内
        paste_x = max(0, min(paste_x, result.shape[1] - relic_resized.shape[1]))
        paste_y = max(0, min(paste_y, result.shape[0] - relic_resized.shape[0]))
        
        # 将圣遗物图片叠加到结果图像上
        result = seamless_blend(result, relic_resized, (paste_x, paste_y))
        
        # 保存结果（OpenCV默认保存BGR，如果是4通道则需要特殊处理）
        output_path = output_folder / f"result_{idx}.png"
        
        # 保存结果图像
        success = cv2.imwrite(str(output_path), result)
        if success:
            print(f"已保存: {output_path}")
        else:
            print(f"保存失败: {output_path}")

    print(f"\n完成！共处理 {len(image_files)} 张图片，结果保存在 '{output_folder}' 文件夹中")

if __name__ == "__main__":
    main()
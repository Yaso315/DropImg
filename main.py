from PIL import Image as PILImage
import os
from pathlib import Path

def crop_center(image, crop_size=(64, 64)):
    """截取图片中心区域"""
    width, height = image.size
    left = (width - crop_size[0]) // 2
    top = (height - crop_size[1]) // 2
    right = left + crop_size[0]
    bottom = top + crop_size[1]
    return image.crop((left, top, right, bottom))

def main():
    # 定义路径
    background_folder = Path("background")
    backborder_path = Path("v2Bg64.png")
    relic_folder = Path("拾取物列表_EN/圣遗物")
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
    
    backborder = PILImage.open(backborder_path).convert("RGBA")
    print(f"backBorder.png 尺寸: {backborder.size}")
    
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
    relic_image = PILImage.open(first_relic_path).convert("RGBA")
    print(f"圣遗物图片尺寸: {relic_image.size}")
    
    # 处理每张图片
    for idx, img_path in enumerate(image_files, 1):
        print(f"\n处理第 {idx} 张图片: {img_path.name}")
        
        # 打开图片并转换为RGBA格式（保留透明通道）
        base_image = PILImage.open(img_path).convert("RGBA")
        print(f"原始图片尺寸: {base_image.size}")
        
        # 截取中心64x64区域
        center_crop = crop_center(base_image, (64, 64))
        print(f"截取中心区域后尺寸: {center_crop.size}")
        
        # 调整backBorder大小以匹配基础图（如果需要）
        if backborder.size != center_crop.size:
            backborder_resized = backborder.resize(center_crop.size, PILImage.LANCZOS)
        else:
            backborder_resized = backborder
        
        # 将backBorder叠加到基础图上
        # 使用alpha_composite进行透明通道混合
        result = PILImage.alpha_composite(center_crop, backborder_resized)
        
        # 调整圣遗物图片大小，使其比基础图小一些（例如70%的大小）
        relic_scale = 0.7  # 缩放比例，可以调整为0.5-0.8之间的值
        relic_new_size = (int(center_crop.size[0] * relic_scale), int(center_crop.size[1] * relic_scale))
        relic_resized = relic_image.resize(relic_new_size, PILImage.LANCZOS)
        
        # 计算居中位置并添加偏移（向右5像素，向下5像素）
        paste_x = (center_crop.size[0] - relic_new_size[0]) // 2 - 3
        paste_y = (center_crop.size[1] - relic_new_size[1]) // 2 + 1
        
        # 创建透明图层用于粘贴圣遗物图片
        result_with_relic = result.copy()
        result_with_relic.paste(relic_resized, (paste_x, paste_y), relic_resized)
        result = result_with_relic
        
        # 保存结果（强制转换为RGBA四通道格式，确保保留透明度信息）
        output_path = output_folder / f"result_{idx}.png"
        result.save(output_path, format="PNG")
        print(f"已保存: {output_path}")
    
    print(f"\n完成！共处理 {len(image_files)} 张图片，结果保存在 '{output_folder}' 文件夹中")

if __name__ == "__main__":
    main()

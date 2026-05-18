import cv2
import numpy as np
import os

def adjust_brightness(image, factor):
    """
    调整图像亮度
    :param image: 输入图像
    :param factor: 亮度调整因子，大于1变亮，小于1变暗
    :return: 调整后的图像
    """
    # 将图像转换为浮点数类型，防止溢出
    adjusted = image.astype(np.float32) * factor
    
    # 确保像素值在有效范围内
    adjusted = np.clip(adjusted, 0, 255)
    
    # 转换回uint8类型
    adjusted = adjusted.astype(np.uint8)
    
    return adjusted

def main():
    # 读取原始图片
    input_image_path = 'v2LightBg_with_glow.png'
    image = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
    
    if image is None:
        print(f"无法加载图片 {input_image_path}")
        return
    
    # 创建输出目录
    output_dir = 'v2_map'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建目录 {output_dir}")
    
    # 定义不同的亮度系数
    brightness_factors = [0.6, 0.8, 1.0, 1.2, 1.4]  # 变暗到变亮
    
    # 生成五种不同亮度的图片
    for i, factor in enumerate(brightness_factors):
        # 如果是RGBA图像，分别处理RGB和Alpha通道
        if len(image.shape) == 3 and image.shape[2] == 4:
            # 分离RGB和Alpha通道
            rgb_channels = image[:, :, :3]
            alpha_channel = image[:, :, 3]
            
            # 调整RGB通道的亮度
            adjusted_rgb = adjust_brightness(rgb_channels, factor)
            
            # 合并调整后的RGB通道和原始Alpha通道
            adjusted_image = np.dstack([adjusted_rgb, alpha_channel])
        else:
            # 对于普通RGB或灰度图像
            adjusted_image = adjust_brightness(image, factor)
        
        # 生成输出文件路径
        output_path = os.path.join(output_dir, f'v2LightBg_with_glow_brightness_{i+1}.png')
        
        # 保存调整后的图片
        cv2.imwrite(output_path, adjusted_image)
        print(f"已保存 {output_path} (亮度系数: {factor})")

if __name__ == "__main__":
    main()
import cv2
import numpy as np
import os

def process_image_channels(image_path, output_path):
    """
    处理四通道图片，根据条件修改第四通道并保存新图片
    - 如果四个通道值都是0，则不变
    - 如果第一通道<130，则第四通道设置为200
    - 如果第一通道>=130，则第四通道设置为255
    """
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"图片文件不存在: {image_path}")
        print("请确保图片文件存在于指定路径中")
        return False
    
    # 使用OpenCV读取图片（保持原格式，包括透明通道）
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if image is None:
        print(f"无法读取图片: {image_path}")
        print("可能是因为图片格式不支持或者文件损坏")
        return False
    
    print(f"原始图片形状: {image.shape}")
    print(f"原始图片数据类型: {image.dtype}")
    
    # 确认图片是四通道的
    if len(image.shape) != 3 or image.shape[2] != 4:
        print("图片不是四通道图片，无法按要求处理")
        return False
    
    # 创建输出图片副本
    output_image = image.copy()
    
    # 获取图片的高度、宽度和通道数
    height, width, channels = image.shape
    
    # 遍历每个像素点
    for y in range(height):
        for x in range(width):
            pixel = image[y, x]
            
            # 检查是否四个通道值都是0
            if all(value == 0 for value in pixel):
                # 四个通道都是0，保持不变
                continue
            # 检查第一通道是否小于130
            elif pixel[0] < 130:
                # 第一通道小于130，将第四通道设置为200
                output_image[y, x][3] = 200
            else:
                # 第一通道大于等于130，将第四通道设置为255
                output_image[y, x][3] = 255
    
    # 保存处理后的图片
    success = cv2.imwrite(output_path, output_image)
    
    if success:
        print(f"处理完成，新图片已保存至: {output_path}")
        print(f"新图片形状: {output_image.shape}")
        print(f"新图片数据类型: {output_image.dtype}")
        return True
    else:
        print(f"保存图片失败: {output_path}")
        return False

if __name__ == "__main__":
    input_image_path = "V2nb64.png"
    output_image_path = "V3nb64.png"
    
    print(f"正在处理图片: {input_image_path}")
    print(f"当前工作目录: {os.getcwd()}")
    
    process_image_channels(input_image_path, output_image_path)
import cv2
import numpy as np
import os


def convert_3channel_to_4channel(image_path, output_path=None):
    """
    将三通道图片转换为四通道图片（添加Alpha透明通道）
    
    Args:
        image_path: 输入图片路径
        output_path: 输出图片路径，默认为None时会自动命名
    
    Returns:
        bool: 转换是否成功
    """
    # 读取图片，保持原始通道数
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        print(f"无法读取图片: {image_path}")
        return False
        
    # 如果是三通道图片，则添加透明通道
    if len(img.shape) == 3 and img.shape[2] == 3:
        # 创建全透明的alpha通道
        alpha_channel = np.ones((img.shape[0], img.shape[1]), dtype=img.dtype) * 255
        # 合并RGB和Alpha通道
        img_4ch = cv2.merge((img[:, :, 0], img[:, :, 1], img[:, :, 2], alpha_channel))
        
        # 如果没有指定输出路径，则自动生成
        if output_path is None:
            name, ext = os.path.splitext(image_path)
            output_path = f"{name}_4ch{ext}"
            
        # 保存为PNG格式以保留Alpha通道
        success = cv2.imwrite(output_path, img_4ch, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        print(f"已将 {image_path} 从3通道转换为4通道，保存至 {output_path}")
        return success
    elif len(img.shape) == 3 and img.shape[2] == 4:
        print(f"图片 {image_path} 已经是4通道，无需转换")
        return True
    else:
        # 其他情况直接保存
        if output_path is None:
            name, ext = os.path.splitext(image_path)
            output_path = f"{name}_4ch{ext}"
            
        success = cv2.imwrite(output_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        print(f"图片 {image_path} 通道数为 {img.shape[2] if len(img.shape) > 2 else 1}，已保存至 {output_path}")
        return success


def batch_convert_directory(input_dir, output_dir=None):
    """
    批量转换目录中的所有图片
    """
    if output_dir is None:
        output_dir = input_dir
    
    # 支持的图片格式
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    for filename in os.listdir(input_dir):
        name, ext = os.path.splitext(filename.lower())
        if ext in supported_formats:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"{name}_4ch.png")  # 强制输出为PNG格式
            convert_3channel_to_4channel(input_path, output_path)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法:")
        print("  单个文件: python 3To4.py <input_image>")
        print("  整个目录: python 3To4.py <input_directory>")
        print("  示例: python 3To4.py image.jpg")
        print("  示例: python 3To4.py ./images/")
    else:
        input_path = sys.argv[1]
        
        if os.path.isdir(input_path):
            # 处理整个目录
            batch_convert_directory(input_path)
        else:
            # 处理单个文件
            convert_3channel_to_4channel(input_path)
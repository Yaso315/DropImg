import cv2
import numpy as np
import sys
import os


def get_image_channels(image_path):
    """
    获取图片的通道数
    
    Args:
        image_path (str): 图片路径
    
    Returns:
        tuple: (通道数, 图片形状)
    """
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误: 文件 {image_path} 不存在")
        return None, None
    
    # 使用OpenCV读取图片
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if image is None:
        print(f"错误: 无法读取图片 {image_path}")
        return None, None
    
    # 获取图片的形状
    if len(image.shape) == 2:  # 灰度图
        channels = 1
    elif len(image.shape) == 3:  # 彩色图
        channels = image.shape[2]
    else:
        channels = len(image.shape)
    
    return channels, image.shape, image


def display_channel_values(image, channels, shape):
    """
    显示各个通道的值
    
    Args:
        image: 图片数组
        channels (int): 通道数
        shape: 图片形状
    """
    print(f"\n各通道值信息:")
    
    if channels == 1:
        # 灰度图
        print("灰度图 - 单通道")
        print(f"值范围: {np.min(image)} - {np.max(image)}")
        print(f"平均值: {np.mean(image):.2f}")
    else:
        # 多通道图
        for i in range(channels):
            if len(shape) == 3:
                channel_data = image[:, :, i]
                print(f"通道 {i+1}:")
                print(f"  值范围: {np.min(channel_data)} - {np.max(channel_data)}")
                print(f"  平均值: {np.mean(channel_data):.2f}")
                
                # 根据常见的通道顺序解释含义
                if channels == 3:
                    if i == 0:
                        print(f"  通道含义: Blue (B)")
                    elif i == 1:
                        print(f"  通道含义: Green (G)")
                    elif i == 2:
                        print(f"  通道含义: Red (R)")
                elif channels == 4:
                    if i == 0:
                        print(f"  通道含义: Blue (B)")
                    elif i == 1:
                        print(f"  通道含义: Green (G)")
                    elif i == 2:
                        print(f"  通道含义: Red (R)")
                    elif i == 3:
                        print(f"  通道含义: Alpha (透明度)")
            else:
                print(f"通道 {i+1} 数据维度异常")


def analyze_image(image_path):
    """
    分析图片的完整信息
    
    Args:
        image_path (str): 图片路径
    """
    channels, shape, image = get_image_channels(image_path)
    
    if channels is None:
        return False
    
    print(f"图片路径: {image_path}")
    print(f"图片形状: {shape}")
    print(f"通道数: {channels}")
    
    # 根据通道数解释图片类型
    if channels == 1:
        print("图片类型: 灰度图")
    elif channels == 3:
        print("图片类型: RGB/BGR 彩色图")
    elif channels == 4:
        print("图片类型: RGBA 彩色图 (带透明通道)")
    else:
        print(f"图片类型: 未知 (通道数: {channels})")
    
    # 显示各通道的值
    display_channel_values(image, channels, shape)
    
    return True


def main():
    """
    主函数，从命令行参数获取图片路径并输出通道数
    """
    if len(sys.argv) < 2:
        print("使用方法: python tongdao.py <图片路径>")
        print("或者运行交互模式:")
        
        while True:
            image_path = input("\n请输入图片路径 (输入 'quit' 退出): ").strip()
            
            if image_path.lower() == 'quit':
                print("程序退出")
                break
            
            if image_path:
                if analyze_image(image_path):
                    print("-" * 50)
            else:
                print("请输入有效的图片路径")
    else:
        # 从命令行参数获取图片路径
        image_path = sys.argv[1]
        analyze_image(image_path)


if __name__ == "__main__":
    main()
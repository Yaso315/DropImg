from PIL import Image as PILImage
import os
import sys


def resize_image_to_64(input_path, output_path=None):
    """
    将输入图片调整为 64x64 分辨率
    
    Args:
        input_path: 输入图片路径
        output_path: 输出图片路径（可选），如果未指定则自动生成
    
    Returns:
        输出图片的路径
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    
    # 打开图片
    img = PILImage.open(input_path)
    
    # 调整图片大小为 64x64
    resized_img = img.resize((64, 64), PILImage.LANCZOS)
    
    # 如果未指定输出路径，则自动生成
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        ext = os.path.splitext(input_path)[1]
        output_path = f"{base_name}_64x64{ext}"
    
    # 保存调整后的图片
    resized_img.save(output_path)
    
    print(f"图片已成功调整为 64x64 分辨率")
    print(f"输入: {input_path}")
    print(f"输出: {output_path}")
    print(f"原始尺寸: {img.size[0]}x{img.size[1]}")
    print(f"新尺寸: 64x64")
    
    return output_path


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python imgChange64.py <输入图片路径> [输出图片路径]")
        print("\n示例:")
        print("  python imgChange64.py image.png")
        print("  python imgChange64.py image.png output.png")
        return
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        resize_image_to_64(input_path, output_path)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

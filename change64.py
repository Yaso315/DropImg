import cv2
import os

def resize_image(input_path, output_path, size=(64, 64)):
    """
    将输入图片调整为指定尺寸
    :param input_path: 输入图片路径
    :param output_path: 输出图片路径
    :param size: 目标尺寸，默认为(64, 64)
    """
    # 读取原始图片
    img = cv2.imread(input_path)
    
    if img is None:
        print(f"无法读取图片: {input_path}")
        return False
    
    # 调整图片尺寸
    resized_img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    
    # 保存调整后的图片
    success = cv2.imwrite(output_path, resized_img)
    
    if success:
        print(f"图片已成功调整为 {size[0]}x{size[1]} 并保存到: {output_path}")
    else:
        print(f"保存图片失败: {output_path}")
    
    return success

if __name__ == "__main__":
    input_image = "V2nb.png"
    output_image = "V2nb_64x64.png"
    
    if os.path.exists(input_image):
        resize_image(input_image, output_image, (64, 64))
    else:
        print(f"找不到文件: {input_image}")
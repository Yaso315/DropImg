import cv2
import numpy as np

def modify_black_transparency(image_path, output_path):
    """
    读取图片并将黑色部分的透明度改为200，但保持原本透明度为0的区域不变
    
    Args:
        image_path: 输入图片路径
        output_path: 输出图片路径
    """
    # 读取图片，保持原有的alpha通道
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        print(f"无法读取图片: {image_path}")
        return
    
    # 如果图片没有alpha通道，则添加一个全不透明的alpha通道
    if img.shape[-1] == 3:
        bgra_img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    else:
        bgra_img = img.copy()
    
    # 分离BGR和Alpha通道
    bgr_channels = bgra_img[:, :, :3]
    alpha_channel = bgra_img[:, :, 3]
    
    # 创建一个掩码，标记原本透明度为0的区域
    original_transparent_mask = (alpha_channel == 0)
    
    # 计算每个像素的亮度值（灰度）
    gray = cv2.cvtColor(bgr_channels, cv2.COLOR_BGR2GRAY)
    
    # 创建黑色区域的掩码（接近黑色的像素）
    # 设定阈值为30，即像素值小于30的被认为是黑色
    _, black_mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
    black_mask = black_mask.astype(np.uint8)
    
    # 结合黑色区域和非透明区域的掩码，只对非完全透明的黑色区域进行处理
    combined_mask = np.bitwise_and(black_mask, ~original_transparent_mask)
    
    # 将黑色区域的透明度设置为200（但排除原本透明度为0的区域）
    alpha_channel[combined_mask == 255] = 200
    
    # 合并通道
    modified_img = np.dstack([bgr_channels, alpha_channel])
    
    # 保存修改后的图片
    cv2.imwrite(output_path, modified_img)
    print(f"图片已保存到: {output_path}")


if __name__ == "__main__":
    input_image_path = "V2nb64.png"
    output_image_path = "V2nb64_modified.png"
    
    modify_black_transparency(input_image_path, output_image_path)
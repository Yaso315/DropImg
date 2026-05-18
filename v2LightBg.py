import cv2
import numpy as np

def add_glow_to_alpha_channel(image_path, output_path, blur_radius=15, intensity=0.5):
    """
    在图像的透明度通道轮廓处添加半透明光晕效果
    
    参数:
    image_path: 输入图像路径
    output_path: 输出图像路径
    blur_radius: 模糊半径，控制光晕扩散范围
    intensity: 光晕强度，0-1之间
    """
    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    # 确保图像是RGBA格式
    if img.shape[2] == 3:  # 如果是RGB，添加Alpha通道
        b_channel, g_channel, r_channel = cv2.split(img)
        alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
        img = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    
    # 提取Alpha通道
    alpha_channel = img[:, :, 3]
    
    # 获取前景轮廓 - 使用阈值获取非透明部分
    _, binary_mask = cv2.threshold(alpha_channel, 1, 255, cv2.THRESH_BINARY)
    binary_mask = binary_mask.astype(np.uint8)
    
    # 对二值化图像进行膨胀，扩展前景区域
    kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)
    
    # 计算边缘 - 这些是我们要添加光晕的地方
    contour_mask = cv2.subtract(dilated_mask, binary_mask)
    
    # 对边缘进行模糊处理，创建光晕效果
    blurred_contour = cv2.GaussianBlur(contour_mask, (blur_radius*2+1, blur_radius*2+1), blur_radius/3)
    
    # 根据强度调整光晕
    glow_effect = (blurred_contour * intensity).astype(np.uint8)
    
    # 将光晕效果应用到原始alpha通道
    # 这样会在轮廓附近增加半透明效果
    expanded_glow = cv2.GaussianBlur(binary_mask, (blur_radius*2+1, blur_radius*2+1), blur_radius/2)
    expanded_glow = (expanded_glow * intensity * 0.5).astype(np.uint8)  # 减少强度以避免过亮
    
    # 组合原始alpha和光晕效果
    final_alpha = cv2.add(alpha_channel, glow_effect)
    final_alpha = np.maximum(alpha_channel, final_alpha)  # 确保不减少原始透明度
    
    # 更新图像的alpha通道
    img[:, :, 3] = final_alpha
    
    # 保存结果
    cv2.imwrite(output_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    
    print(f"处理完成，输出文件: {output_path}")
    return img


def advanced_add_glow_to_alpha_channel(image_path, output_path, blur_radius=15, intensity=0.3):
    """
    更高级的光晕添加方法，使用多层模糊来创建更自然的光晕效果
    
    参数:
    image_path: 输入图像路径
    output_path: 输出图像路径
    blur_radius: 模糊半径，控制光晕扩散范围
    intensity: 光晕强度，0-1之间
    """
    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    # 确保图像是RGBA格式
    if img.shape[2] == 3:  # 如果是RGB，添加Alpha通道
        b_channel, g_channel, r_channel = cv2.split(img)
        alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
        img = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    
    # 提取Alpha通道
    alpha_channel = img[:, :, 3]
    
    # 创建一个二值掩码，突出显示非透明区域
    _, binary_mask = cv2.threshold(alpha_channel, 1, 255, cv2.THRESH_BINARY)
    binary_mask = binary_mask.astype(np.uint8)
    
    # 计算轮廓
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 创建轮廓掩码
    contour_mask = np.zeros_like(binary_mask)
    cv2.drawContours(contour_mask, contours, -1, (255), thickness=2)
    
    # 创建多个模糊层来模拟光晕效果
    glow_layers = []
    weights = [0.6, 0.3, 0.1]  # 不同层的权重
    radii = [blur_radius, blur_radius*2, blur_radius*3]  # 不同的模糊半径
    
    for i, radius in enumerate(radii):
        kernel_size = radius * 2 + 1
        if kernel_size % 2 == 0:  # 确保核大小为奇数
            kernel_size += 1
        blurred = cv2.GaussianBlur(contour_mask, (kernel_size, kernel_size), radius/3)
        # 应用强度并转换为uint8
        blurred = (blurred * intensity * weights[i]).astype(np.uint8)
        glow_layers.append(blurred)
    
    # 合并所有光晕层
    combined_glow = np.zeros_like(binary_mask, dtype=np.float32)
    for layer in glow_layers:
        combined_glow += layer.astype(np.float32)
    
    # 限制最大值为255
    combined_glow = np.clip(combined_glow, 0, 255).astype(np.uint8)
    
    # 将光晕效果合并到原始alpha通道
    final_alpha = cv2.add(alpha_channel, combined_glow)
    # 确保alpha值不超过255
    final_alpha = np.minimum(final_alpha, 255)
    
    # 更新图像的alpha通道
    img[:, :, 3] = final_alpha
    
    # 保存结果
    cv2.imwrite(output_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    
    print(f"高级处理完成，输出文件: {output_path}")
    return img


if __name__ == "__main__":
    input_path = "v2Bg.png"
    output_path = "v2Bg_with_glow.png"
    
    # 可以选择使用基础或高级方法
    # 使用高级方法，通常效果更好
    result_img = advanced_add_glow_to_alpha_channel(input_path, output_path, blur_radius=10, intensity=0.4)
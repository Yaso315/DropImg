import cv2
import numpy as np
import os

def extract_center_patch(image, patch_size=(64, 64)):
    """
    提取图像中心的指定大小区域
    """
    h, w = image.shape[:2]
    
    # 计算中心点坐标
    center_x, center_y = w // 2, h // 2
    
    # 计算要截取的区域
    half_w, half_h = patch_size[0] // 2, patch_size[1] // 2
    
    start_x = max(0, center_x - half_w)
    end_x = min(w, center_x + half_w)
    start_y = max(0, center_y - half_h)
    end_y = min(h, center_y + half_h)
    
    # 提取中心区域
    center_patch = image[start_y:end_y, start_x:end_x]
    
    # 如果提取的区域小于目标尺寸，则填充
    if center_patch.shape[0] != patch_size[1] or center_patch.shape[1] != patch_size[0]:
        # 创建一个目标尺寸的画布
        if len(image.shape) == 3:  # 彩色图像
            patch = np.zeros((patch_size[1], patch_size[0], image.shape[2]), dtype=image.dtype)
        else:  # 灰度图像
            patch = np.zeros((patch_size[1], patch_size[0]), dtype=image.dtype)
        
        # 计算偏移量
        offset_y = (patch_size[1] - center_patch.shape[0]) // 2
        offset_x = (patch_size[0] - center_patch.shape[1]) // 2
        
        # 将中心区域复制到画布上
        patch[offset_y:offset_y+center_patch.shape[0], 
              offset_x:offset_x+center_patch.shape[1]] = center_patch
        
        return patch
    
    return center_patch

def ensure_four_channel(image):
    """
    确保图像是四通道的，透明度设为完全不透明
    """
    if len(image.shape) == 2:  # 灰度图
        # 转换为BGR
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    if image.shape[2] == 3:  # 三通道图像，添加alpha通道
        # 创建alpha通道，值全为255（完全不透明）
        alpha_channel = np.full((image.shape[0], image.shape[1]), 255, dtype=image.dtype)
        # 合并三个颜色通道和alpha通道
        four_channel_image = cv2.merge((image[:,:,0], image[:,:,1], image[:,:,2], alpha_channel))
        return four_channel_image
    elif image.shape[2] == 4:  # 四通道图像
        # 确保alpha通道为完全不透明
        image[:, :, 3] = 255
        return image
    
    return image

def process_images():
    """
    处理background文件夹中的前两张图片
    """
    bg_folder = 'background'
    
    # 检查并创建输出文件夹
    output_folder = '64Bg'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 获取background文件夹中的所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_files = []
    
    for file in os.listdir(bg_folder):
        ext = os.path.splitext(file)[1].lower()
        if ext in image_extensions:
            image_files.append(file)
    
    # 只处理前两张图片
    image_files = sorted(image_files)[:2]
    
    if len(image_files) < 2:
        print(f"警告: 在background文件夹中只找到 {len(image_files)} 张图片")
    
    for i, filename in enumerate(image_files):
        print(f"正在处理第 {i+1} 张图片: {filename}")
        
        # 读取图片
        img_path = os.path.join(bg_folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # 保持原图的所有通道
        
        if img is None:
            print(f"无法读取图片: {img_path}")
            continue
        
        # 提取中心64x64区域
        center_patch = extract_center_patch(img, (64, 64))
        
        # 确保是四通道且完全不透明
        four_channel_img = ensure_four_channel(center_patch)
        
        # 保存结果到64Bg文件夹
        output_filename = f"extracted_bg_{i+1}_{os.path.splitext(filename)[0]}.png"
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, four_channel_img)
        print(f"已保存: {output_path}")

if __name__ == "__main__":
    process_images()
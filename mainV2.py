import cv2
import numpy as np
import os
import random

def overlay_images(bg_folder, overlay_img_path, top_img_folder=None, fourth_img_path=None, top_size=(64,64), fourth_size=(64,64), pos_x=0, pos_y=0, bg_size=(1024, 1024), overlay_size=(1024, 1024)):
    """
    从指定文件夹随机读取一张背景图片，然后叠加另外两张图片，最后可选择性地叠加第四张图片
    
    Args:
        bg_folder: 背景图片文件夹路径
        overlay_img_path: 要叠加的图片路径
        top_img_folder: 最顶层要叠加的图片文件夹路径（可选）
        fourth_img_path: 第四层图片路径（可选）
        top_size: 顶层图片调整后的尺寸 (width, height)
        fourth_size: 第四层图片调整后的尺寸 (width, height)
        pos_x: 第四层图片的X坐标位置
        pos_y: 第四层图片的Y坐标位置
        bg_size: 背景图片调整后的尺寸 (width, height)
        overlay_size: 中间层图片调整后的尺寸 (width, height)
    
    Returns:
        合成后的图片
    """
    # 获取背景文件夹中的所有图片文件
    bg_images = [f for f in os.listdir(bg_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    if not bg_images:
        raise ValueError(f"在 {bg_folder} 文件夹中没有找到图片文件")
    
    # 随机选择一张背景图片
    selected_bg = random.choice(bg_images)
    bg_img_path = os.path.join(bg_folder, selected_bg)
    
    # 读取背景图片
    bg_img = cv2.imread(bg_img_path, cv2.IMREAD_UNCHANGED)
    if bg_img is None:
        raise ValueError(f"无法读取背景图片: {bg_img_path}")
    
    # 读取要叠加的图片
    overlay_img = cv2.imread(overlay_img_path, cv2.IMREAD_UNCHANGED)
    if overlay_img is None:
        raise ValueError(f"无法读取叠加图片: {overlay_img_path}")
    
    # 调整背景图片大小
    bg_img_resized = cv2.resize(bg_img, bg_size)
    
    # 调整中间层叠加图片大小
    overlay_img_resized = cv2.resize(overlay_img, overlay_size)
    
    # 如果中间层叠加图片是RGBA格式，我们需要特殊处理透明通道
    if overlay_img_resized.shape[-1] == 4:
        # 分离RGB和Alpha通道
        rgb_channels = overlay_img_resized[:, :, :3]
        alpha_channel = overlay_img_resized[:, :, 3] / 255.0
        
        # 计算叠加位置（居中）
        y_offset = (bg_size[1] - overlay_size[1]) // 2
        x_offset = (bg_size[0] - overlay_size[0]) // 2
        
        # 确保偏移量不会超出边界
        y_end = min(y_offset + overlay_size[1], bg_size[1])
        x_end = min(x_offset + overlay_size[0], bg_size[0])
        
        # 创建感兴趣的区域
        roi = bg_img_resized[y_offset:y_end, x_offset:x_end]
        
        # 使用alpha混合进行叠加
        for c in range(0, 3):
            roi[:, :, c] = ((1 - alpha_channel[0:(y_end-y_offset), 0:(x_end-x_offset)]) * 
                            roi[:, :, c] + 
                            alpha_channel[0:(y_end-y_offset), 0:(x_end-x_offset)] * 
                            rgb_channels[0:(y_end-y_offset), 0:(x_end-x_offset), c])
    else:
        # 如果没有透明通道，直接覆盖
        y_offset = (bg_size[1] - overlay_size[1]) // 2
        x_offset = (bg_size[0] - overlay_size[0]) // 2
        
        y_end = min(y_offset + overlay_size[1], bg_size[1])
        x_end = min(x_offset + overlay_size[0], bg_size[0])
        
        bg_img_resized[y_offset:y_end, x_offset:x_end] = overlay_img_resized[0:(y_end-y_offset), 0:(x_end-x_offset)]
    
    # 如果指定了顶层图片文件夹，则获取并叠加第一张图片
    if top_img_folder and os.path.exists(top_img_folder):
        # 获取顶层图片文件夹中的所有图片文件
        top_images = [f for f in os.listdir(top_img_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        if top_images:
            # 选取第一张图片
            top_img_name = top_images[0]
            top_img_path = os.path.join(top_img_folder, top_img_name)
            
            # 读取顶层图片
            top_img = cv2.imread(top_img_path, cv2.IMREAD_UNCHANGED)
            if top_img is not None:
                # 调整顶层图片大小
                top_img_resized = cv2.resize(top_img, top_size)
                
                # 如果顶层图片是RGBA格式，我们需要特殊处理透明通道
                if top_img_resized.shape[-1] == 4:
                    # 分离RGB和Alpha通道
                    rgb_channels_top = top_img_resized[:, :, :3]
                    alpha_channel_top = top_img_resized[:, :, 3] / 255.0
                    
                    # 计算叠加位置（居中）
                    y_offset_top = (bg_size[1] - top_size[1]) // 2
                    x_offset_top = (bg_size[0] - top_size[0]) // 2
                    
                    # 确保偏移量不会超出边界
                    y_end_top = min(y_offset_top + top_size[1], bg_size[1])
                    x_end_top = min(x_offset_top + top_size[0], bg_size[0])
                    
                    # 创建感兴趣的区域
                    roi_top = bg_img_resized[y_offset_top:y_end_top, x_offset_top:x_end_top]
                    
                    # 使用alpha混合进行叠加
                    for c in range(0, 3):
                        roi_top[:, :, c] = ((1 - alpha_channel_top[0:(y_end_top-y_offset_top), 0:(x_end_top-x_offset_top)]) * 
                                        roi_top[:, :, c] + 
                                        alpha_channel_top[0:(y_end_top-y_offset_top), 0:(x_end_top-x_offset_top)] * 
                                        rgb_channels_top[0:(y_end_top-y_offset_top), 0:(x_end_top-x_offset_top), c])
                else:
                    # 如果没有透明通道，直接覆盖
                    y_offset_top = (bg_size[1] - top_size[1]) // 2
                    x_offset_top = (bg_size[0] - top_size[0]) // 2
                    
                    y_end_top = min(y_offset_top + top_size[1], bg_size[1])
                    x_end_top = min(x_offset_top + top_size[0], bg_size[0])
                    
                    bg_img_resized[y_offset_top:y_end_top, x_offset_top:x_end_top] = top_img_resized[0:(y_end_top-y_offset_top), 0:(x_end_top-x_offset_top)]

    # 如果指定了第四张图片，则叠加这张图片
    if fourth_img_path and os.path.exists(fourth_img_path):
        # 读取第四张图片
        fourth_img = cv2.imread(fourth_img_path, cv2.IMREAD_UNCHANGED)
        if fourth_img is not None:
            # 调整第四张图片大小
            fourth_img_resized = cv2.resize(fourth_img, fourth_size)
            
            # 如果第四张图片是RGBA格式，我们需要特殊处理透明通道
            if fourth_img_resized.shape[-1] == 4:
                # 分离RGB和Alpha通道
                rgb_channels_fourth = fourth_img_resized[:, :, :3]
                alpha_channel_fourth = fourth_img_resized[:, :, 3] / 255.0
                
                # 根据指定的位置叠加
                y_offset_fourth = pos_y
                x_offset_fourth = pos_x
                
                # 确保偏移量不会超出边界
                y_end_fourth = min(y_offset_fourth + fourth_size[1], bg_size[1])
                x_end_fourth = min(x_offset_fourth + fourth_size[0], bg_size[0])
                
                # 计算实际绘制区域
                h_diff = y_end_fourth - y_offset_fourth
                w_diff = x_end_fourth - x_offset_fourth
                
                # 创建感兴趣的区域
                roi_fourth = bg_img_resized[y_offset_fourth:y_end_fourth, x_offset_fourth:x_end_fourth]
                
                # 使用alpha混合进行叠加
                for c in range(0, 3):
                    roi_fourth[:, :, c] = ((1 - alpha_channel_fourth[0:h_diff, 0:w_diff]) * 
                                    roi_fourth[:, :, c] + 
                                    alpha_channel_fourth[0:h_diff, 0:w_diff] * 
                                    rgb_channels_fourth[0:h_diff, 0:w_diff, c])
            else:
                # 如果没有透明通道，直接覆盖
                y_offset_fourth = pos_y
                x_offset_fourth = pos_x
                
                y_end_fourth = min(y_offset_fourth + fourth_size[1], bg_size[1])
                x_end_fourth = min(x_offset_fourth + fourth_size[0], bg_size[0])
                
                h_diff = y_end_fourth - y_offset_fourth
                w_diff = x_end_fourth - x_offset_fourth
                
                bg_img_resized[y_offset_fourth:y_end_fourth, x_offset_fourth:x_end_fourth] = fourth_img_resized[0:h_diff, 0:w_diff]

    return bg_img_resized


def main():
    bg_folder = "64Bg"  # 背景图片所在文件夹
    overlay_img_path = "v2Bgnew.png"  # 要叠加的图片
    top_img_folder = "droplist/syw"  # 最顶层图片所在文件夹
    fourth_img_path = "V3nb64.png"  # 第四张图片路径
    top_size = (48, 48)  # 顶层图片尺寸，可以根据需要调整
    fourth_size = (16, 16)  # 第四张图片尺寸
    pos_x = 32  # 第四张图片的X坐标位置
    pos_y = 32  # 第四张图片的Y坐标位置
    
    # 检查文件是否存在
    if not os.path.exists(bg_folder):
        print(f"错误: 背景图片文件夹 '{bg_folder}' 不存在")
        return
    
    if not os.path.exists(overlay_img_path):
        print(f"错误: 叠加图片 '{overlay_img_path}' 不存在")
        return

    if not os.path.exists(fourth_img_path):
        print(f"警告: 第四张图片 '{fourth_img_path}' 不存在，跳过该图层")
        fourth_img_path = None

    # 检查顶层图片文件夹是否存在
    if not os.path.exists(top_img_folder):
        print(f"警告: 顶层图片文件夹 '{top_img_folder}' 不存在，仅进行两层叠加")
        top_img_folder = None
    
    # 设置图片尺寸 (宽度, 高度)
    bg_size = (64,64)      # 背景图片尺寸
    overlay_size = (64,64)   # 中间层图片尺寸
    
    try:
        # 叠加图片
        result_img = overlay_images(bg_folder, overlay_img_path, top_img_folder, fourth_img_path, top_size, fourth_size, pos_x, pos_y, bg_size, overlay_size)
        
        # 显示结果
        cv2.imshow("Combined Image", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 保存结果
        output_filename = "output\combined_result.png"
        cv2.imwrite(output_filename, result_img)
        print(f"结果已保存为: {output_filename}")
        
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")


if __name__ == "__main__":
    main()
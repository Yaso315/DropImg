import cv2
import numpy as np

# 读取图片
image = cv2.imread('v2Bg.png', cv2.IMREAD_UNCHANGED)

if image is not None:
    print("图片读取成功")
    print(f"图片形状: {image.shape}")
    
    # 检查图片是否有第四个通道(透明度通道)
    if len(image.shape) == 3 and image.shape[2] >= 4:
        # 获取第四个通道(透明度通道)
        alpha_channel = image[:, :, 3]
        print(f"第四个通道的形状: {alpha_channel.shape}")
        print(f"第四个通道的数据类型: {alpha_channel.dtype}")
        
        # 将第四通道中大于230的值改为200
        alpha_channel[alpha_channel > 230] = 200
        
        # 更新原图像的第四个通道
        image[:, :, 3] = alpha_channel
        
        # 保存新图片
        success = cv2.imwrite('v2Bgnew.png', image)
        if success:
            print("新图片 v2Bgnew.png 已保存")
        else:
            print("保存图片失败")
    else:
        print("该图片没有第四个通道(透明度通道)")
        print(f"当前图片的通道数: {image.shape[2] if len(image.shape) > 2 else 1}")
else:
    print("图片读取失败，请检查文件路径是否正确")
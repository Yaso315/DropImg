import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os


class ImageCropper:
    def __init__(self, root):
        self.root = root
        self.root.title("图片裁剪工具 - 裁剪为64x64")
        
        # 初始化变量
        self.image = None
        self.photo_image = None
        self.canvas = None
        self.start_x = None
        self.start_y = None
        self.crop_rect_id = None
        self.crop_rect = None
        self.is_dragging = False
        
        # 创建界面
        self.create_widgets()
        
    def create_widgets(self):
        # 按钮框架
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # 加载图片按钮
        load_btn = tk.Button(btn_frame, text="加载图片", command=self.load_image)
        load_btn.pack(side=tk.LEFT, padx=5)
        
        # 裁剪按钮
        crop_btn = tk.Button(btn_frame, text="裁剪并保存", command=self.crop_and_save)
        crop_btn.pack(side=tk.LEFT, padx=5)
        
        # 提示信息
        info_label = tk.Label(btn_frame, text="提示：拖拽红色框选择裁剪区域（64x64）")
        info_label.pack(side=tk.LEFT, padx=10)
        
        # 画布
        self.canvas = tk.Canvas(self.root, bg='gray')
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 绑定鼠标事件
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        
    def load_image(self):
        # 打开文件对话框
        file_path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图片文件", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        
        if not file_path:
            return
            
        # 加载图片
        self.image = Image.open(file_path)
        
        # 调整图片大小以适应窗口
        self.display_image = self.resize_image_to_fit(self.image)
        self.photo_image = ImageTk.PhotoImage(self.display_image)
        
        # 在画布上显示图片
        self.canvas.delete("all")
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # 如果画布还没有大小，使用默认大小
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 800
            canvas_height = 600
            
        x = (canvas_width - self.photo_image.width()) // 2
        y = (canvas_height - self.photo_image.height()) // 2
        
        self.canvas.create_image(x, y, anchor=tk.NW, image=self.photo_image)
        self.canvas.image_x = x
        self.canvas.image_y = y
        
        # 创建初始的64x64裁剪框（居中显示）
        self.create_crop_box(x, y)
        
    def create_crop_box(self, img_x, img_y):
        """创建64x64的裁剪框"""
        # 计算缩放后的64x64大小
        scale_x = self.display_image.width / self.image.width
        scale_y = self.display_image.height / self.image.height
        scaled_size = int(64 * scale_x)  # 假设宽高比例相同
        
        # 计算裁剪框位置（居中）
        center_x = img_x + self.display_image.width // 2
        center_y = img_y + self.display_image.height // 2
        
        x1 = center_x - scaled_size // 2
        y1 = center_y - scaled_size // 2
        x2 = x1 + scaled_size
        y2 = y1 + scaled_size
        
        # 确保裁剪框在图片范围内
        x1 = max(x1, img_x)
        y1 = max(y1, img_y)
        x2 = min(x2, img_x + self.display_image.width)
        y2 = min(y2, img_y + self.display_image.height)
        
        # 绘制裁剪框
        self.crop_rect_id = self.canvas.create_rectangle(
            x1, y1, x2, y2,
            outline='red', width=3, tags='crop_box'
        )
        
        # 添加半透明填充
        self.canvas.create_rectangle(
            x1, y1, x2, y2,
            fill='red', stipple='gray50', tags='crop_box'
        )
        
        # 保存裁剪框坐标
        self.crop_rect = (x1, y1, x2, y2)
        
    def resize_image_to_fit(self, img):
        # 获取画布大小
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # 如果画布还没有大小，使用默认大小
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 800
            canvas_height = 600
            
        # 计算缩放比例
        width_ratio = canvas_width / img.width
        height_ratio = canvas_height / img.height
        ratio = min(width_ratio, height_ratio, 1.0)  # 不放大图片
        
        # 计算新尺寸
        new_width = int(img.width * ratio)
        new_height = int(img.height * ratio)
        
        # 调整图片大小
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        return resized_img
        
    def on_mouse_down(self, event):
        if self.image is None:
            return
            
        # 检查是否点击在裁剪框内
        if self.crop_rect:
            x1, y1, x2, y2 = self.crop_rect
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                self.is_dragging = True
                self.start_x = event.x
                self.start_y = event.y
                
    def on_mouse_drag(self, event):
        if not self.is_dragging or self.crop_rect is None:
            return
            
        # 计算移动距离
        dx = event.x - self.start_x
        dy = event.y - self.start_y
        
        # 更新裁剪框位置
        x1, y1, x2, y2 = self.crop_rect
        new_x1 = x1 + dx
        new_y1 = y1 + dy
        new_x2 = x2 + dx
        new_y2 = y2 + dy
        
        # 获取图片边界
        if hasattr(self.canvas, 'image_x'):
            img_x = self.canvas.image_x
            img_y = self.canvas.image_y
            img_width = self.display_image.width
            img_height = self.display_image.height
            
            # 限制裁剪框在图片范围内
            if new_x1 < img_x:
                offset = img_x - new_x1
                new_x1 += offset
                new_x2 += offset
            if new_y1 < img_y:
                offset = img_y - new_y1
                new_y1 += offset
                new_y2 += offset
            if new_x2 > img_x + img_width:
                offset = new_x2 - (img_x + img_width)
                new_x1 -= offset
                new_x2 -= offset
            if new_y2 > img_y + img_height:
                offset = new_y2 - (img_y + img_height)
                new_y1 -= offset
                new_y2 -= offset
        
        # 删除旧的裁剪框
        self.canvas.delete('crop_box')
        
        # 绘制新的裁剪框
        self.crop_rect_id = self.canvas.create_rectangle(
            new_x1, new_y1, new_x2, new_y2,
            outline='red', width=3, tags='crop_box'
        )
        
        self.canvas.create_rectangle(
            new_x1, new_y1, new_x2, new_y2,
            fill='red', stipple='gray50', tags='crop_box'
        )
        
        # 更新裁剪框坐标
        self.crop_rect = (new_x1, new_y1, new_x2, new_y2)
        
        # 更新起始点
        self.start_x = event.x
        self.start_y = event.y
        
    def on_mouse_up(self, event):
        self.is_dragging = False
        
    def crop_and_save(self):
        if self.image is None:
            messagebox.showwarning("警告", "请先加载图片！")
            return
            
        if self.crop_rect is None:
            messagebox.showwarning("警告", "请先选择裁剪区域！")
            return
            
        # 计算实际裁剪区域（考虑缩放）
        x1, y1, x2, y2 = self.crop_rect
        
        # 如果有偏移，需要减去偏移
        if hasattr(self.canvas, 'image_x'):
            img_x = self.canvas.image_x
            img_y = self.canvas.image_y
            x1 -= img_x
            y1 -= img_y
            x2 -= img_x
            y2 -= img_y
            
        # 计算缩放比例
        scale_x = self.image.width / self.display_image.width
        scale_y = self.image.height / self.display_image.height
        
        # 转换到原始图片坐标
        orig_x1 = int(x1 * scale_x)
        orig_y1 = int(y1 * scale_y)
        orig_x2 = int(x2 * scale_x)
        orig_y2 = int(y2 * scale_y)
        
        # 裁剪图片
        cropped = self.image.crop((orig_x1, orig_y1, orig_x2, orig_y2))
        
        # 调整为64x64
        resized = cropped.resize((64, 64), Image.LANCZOS)
        
        # 保存文件
        save_path = filedialog.asksaveasfilename(
            title="保存裁剪后的图片",
            defaultextension=".png",
            filetypes=[("PNG图片", "*.png"), ("JPEG图片", "*.jpg"), ("所有文件", "*.*")]
        )
        
        if save_path:
            resized.save(save_path)
            messagebox.showinfo("成功", f"图片已保存到:\n{save_path}")


def main():
    root = tk.Tk()
    root.geometry("900x700")
    
    app = ImageCropper(root)
    root.mainloop()


if __name__ == "__main__":
    main()

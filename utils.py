"""
工具函数
"""

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision import transforms


def load_image(img_path, max_size=512, shape=None, device='cpu'):
    """
    加载和预处理图像，保持原始宽高比
    
    Args:
        img_path: 图像路径
        max_size: 最大尺寸
        shape: 指定形状
        device: 设备
    """
    image = Image.open(img_path).convert('RGB')
    
    if shape is not None:
        # 如果指定了形状，直接使用
        transform = transforms.Compose([
            transforms.Resize(shape),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    else:
        # 保持宽高比的缩放
        w, h = image.size
        if max(w, h) > max_size:
            if w > h:
                new_w = max_size
                new_h = int(h * max_size / w)
            else:
                new_h = max_size
                new_w = int(w * max_size / h)
        else:
            new_w, new_h = w, h
        
        transform = transforms.Compose([
            transforms.Resize((new_h, new_w)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
    image = transform(image)[:3, :, :].unsqueeze(0)
    return image.to(device)

def im_convert(tensor):
    """
    将张量转换为可显示的图像
    """
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image

def save_image(tensor, path, title=None):
    """
    保存图像
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image = im_convert(tensor)
    image_pil = Image.fromarray((image * 255).astype(np.uint8))
    image_pil.save(path)
    
    if title:
        print(f'{title}: {path}')

def create_gif(image_list, output_path, duration=500):
    """
    创建GIF动画
    """
    if len(image_list) < 2:
        return
        
    pil_images = []
    for tensor in image_list:
        image = im_convert(tensor)
        pil_image = Image.fromarray((image * 255).astype(np.uint8))
        pil_images.append(pil_image)
    
    pil_images[0].save(
        output_path,
        save_all=True,
        append_images=pil_images[1:],
        duration=duration,
        loop=0
    )

def plot_progress(losses, save_path=None):
    """
    绘制损失曲线
    """
    plt.figure(figsize=(12, 4))
    
    # 总损失
    plt.subplot(1, 3, 1)
    plt.plot(losses['total'])
    plt.title('Total Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    
    # 内容损失
    plt.subplot(1, 3, 2)
    plt.plot(losses['content'])
    plt.title('Content Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    
    # 风格损失
    plt.subplot(1, 3, 3)
    plt.plot(losses['style'])
    plt.title('Style Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def visualize_features(features, layer_name, save_path=None):
    """
    可视化特征图
    """
    feature_maps = features[layer_name].squeeze(0).detach().cpu()
    
    # 显示前16个特征图
    num_maps = min(16, feature_maps.shape[0])
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    
    for i in range(num_maps):
        ax = axes[i // 4, i % 4]
        ax.imshow(feature_maps[i], cmap='viridis')
        ax.set_title(f'Feature {i+1}')
        ax.axis('off')
    
    plt.suptitle(f'Feature Maps - {layer_name}')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def blend_images(img1, img2, alpha=0.5):
    """
    混合两个图像
    """
    return alpha * img1 + (1 - alpha) * img2

def apply_mask(target, content, mask_path):
    """
    应用遮罩进行局部风格迁移
    """
    mask = Image.open(mask_path).convert('L')
    mask = transforms.ToTensor()(mask).unsqueeze(0).to(target.device)
    
    # 确保遮罩尺寸匹配
    mask = torch.nn.functional.interpolate(mask, size=target.shape[-2:], mode='bilinear')
    
    return target * mask + content * (1 - mask)

def get_timestamp():
    """
    获取时间戳字符串
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def print_progress(iteration, total_iterations, losses, start_time=None):
    """
    打印训练进度
    """
    if start_time:
        elapsed = datetime.now() - start_time
        remaining = elapsed * (total_iterations - iteration) / iteration if iteration > 0 else None
        time_info = f" | 已用时间: {elapsed} | 预计剩余: {remaining}" if remaining else f" | 已用时间: {elapsed}"
    else:
        time_info = ""
    
    print(f"迭代 {iteration}/{total_iterations} - "
          f"总损失: {losses['total'][-1]:.4f} | "
          f"内容损失: {losses['content'][-1]:.4f} | "
          f"风格损失: {losses['style'][-1]:.4f}"
          f"{time_info}")

def validate_paths(content_path, style_path):
    """
    验证输入路径
    """
    if not os.path.exists(content_path):
        raise FileNotFoundError(f"内容图像文件未找到: {content_path}")
    if not os.path.exists(style_path):
        raise FileNotFoundError(f"风格图像文件未找到: {style_path}")
    
    # 验证是否为有效图像文件
    try:
        Image.open(content_path)
        Image.open(style_path)
    except Exception as e:
        raise ValueError(f"无效的图像文件: {e}")

def enhance_image(image_tensor, enhancement_type='sharpen'):
    """
    图像增强
    """
    image_pil = Image.fromarray((im_convert(image_tensor) * 255).astype(np.uint8))
    
    if enhancement_type == 'sharpen':
        image_pil = image_pil.filter(ImageFilter.SHARPEN)
    elif enhancement_type == 'smooth':
        image_pil = image_pil.filter(ImageFilter.SMOOTH)
    elif enhancement_type == 'detail':
        image_pil = image_pil.filter(ImageFilter.DETAIL)
    
    # 转换回张量
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    return transform(image_pil).unsqueeze(0)

def resize_keep_aspect_ratio(image, max_size):
    """
    调整图像尺寸同时保持宽高比
    
    Args:
        image: PIL Image对象
        max_size: 最大尺寸
    
    Returns:
        调整后的PIL Image对象和原始尺寸
    """
    original_size = image.size
    w, h = original_size
    
    # 计算缩放比例
    if max(w, h) > max_size:
        if w > h:
            new_w = max_size
            new_h = int(h * max_size / w)
        else:
            new_h = max_size
            new_w = int(w * max_size / h)
    else:
        new_w, new_h = w, h
    
    # 调整图像尺寸
    resized_image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    return resized_image, original_size, (new_w, new_h)

"""
批处理风格迁移脚本
"""

import glob
import os

import torch

from config import Config
from neural_style import NeuralStyleTransfer


def batch_style_transfer(content_dir, style_dir, output_base_dir, config_template=None):
    """
    批量处理风格迁移
    
    Args:
        content_dir: 内容图像目录
        style_dir: 风格图像目录  
        output_base_dir: 输出基础目录
        config_template: 配置模板
    """
    
    if config_template is None:
        config_template = Config()
    
    # 获取所有图像文件
    content_files = glob.glob(os.path.join(content_dir, "*.jpg")) + \
                   glob.glob(os.path.join(content_dir, "*.png"))
    style_files = glob.glob(os.path.join(style_dir, "*.jpg")) + \
                 glob.glob(os.path.join(style_dir, "*.png"))
    
    if not content_files:
        print(f"❌ 在 {content_dir} 中未找到内容图像")
        return
        
    if not style_files:
        print(f"❌ 在 {style_dir} 中未找到风格图像")
        return
    
    print(f"📁 找到 {len(content_files)} 个内容图像")
    print(f"🎨 找到 {len(style_files)} 个风格图像")
    print(f"📊 将生成 {len(content_files) * len(style_files)} 个结果")
    
    total_combinations = len(content_files) * len(style_files)
    current_combination = 0
    
    for content_file in content_files:
        content_name = os.path.splitext(os.path.basename(content_file))[0]
        
        for style_file in style_files:
            style_name = os.path.splitext(os.path.basename(style_file))[0]
            
            current_combination += 1
            print(f"\n🔄 处理组合 {current_combination}/{total_combinations}")
            print(f"📷 内容: {content_name}")
            print(f"🎭 风格: {style_name}")
            
            # 创建输出目录
            output_dir = os.path.join(output_base_dir, f"{content_name}_{style_name}")
            os.makedirs(output_dir, exist_ok=True)
            
            # 更新配置
            config = Config()
            config.content_path = content_file
            config.style_path = style_file
            config.output_dir = output_dir
            config.iterations = config_template.iterations
            config.max_size = config_template.max_size
            config.style_weight = config_template.style_weight
            config.content_weight = config_template.content_weight
            
            # 执行风格迁移
            try:
                nst = NeuralStyleTransfer(config)
                nst.run()
                print(f"✅ 完成: {content_name} + {style_name}")
            except Exception as e:
                print(f"❌ 失败: {content_name} + {style_name} - {e}")
                continue
    
    print(f"\n🎉 批处理完成！结果保存在: {output_base_dir}")

def create_style_matrix(content_files, style_files, output_dir, config_template=None):
    """
    创建风格矩阵 - 显示所有内容和风格的组合结果
    """
    import matplotlib.pyplot as plt
    from utils import im_convert, load_image
    
    if config_template is None:
        config_template = Config()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载所有图像
    content_images = []
    style_images = []
    
    for content_file in content_files[:3]:  # 限制为3个内容图像
        img = load_image(content_file, max_size=config_template.max_size, device=device)
        content_images.append(img)
    
    for style_file in style_files[:3]:  # 限制为3个风格图像
        img = load_image(style_file, max_size=config_template.max_size, device=device)
        style_images.append(img)
    
    # 创建矩阵图
    fig, axes = plt.subplots(len(content_images) + 1, len(style_images) + 1, 
                            figsize=(4 * (len(style_images) + 1), 4 * (len(content_images) + 1)))
    
    # 显示风格图像（第一行）
    axes[0, 0].axis('off')  # 左上角空白
    for j, style_img in enumerate(style_images):
        axes[0, j + 1].imshow(im_convert(style_img))
        axes[0, j + 1].set_title(f'Style {j+1}')
        axes[0, j + 1].axis('off')
    
    # 显示内容图像和结果
    for i, content_img in enumerate(content_images):
        # 显示内容图像（第一列）
        axes[i + 1, 0].imshow(im_convert(content_img))
        axes[i + 1, 0].set_title(f'Content {i+1}')
        axes[i + 1, 0].axis('off')
        
        # 生成和显示风格迁移结果
        for j, style_img in enumerate(style_images):
            print(f"生成结果: Content {i+1} + Style {j+1}")
            
            # 快速风格迁移（较少迭代）
            config = Config()
            config.content_path = content_files[i]
            config.style_path = style_files[j]
            config.iterations = 500  # 快速预览
            config.max_size = 256
            config.show_every = 1000  # 不显示中间结果
            
            nst = NeuralStyleTransfer(config)
            nst.load_images()
            nst.setup_losses()
            nst.optimize()
            
            result = nst.target_img.clone()
            axes[i + 1, j + 1].imshow(im_convert(result))
            axes[i + 1, j + 1].set_title(f'C{i+1}+S{j+1}')
            axes[i + 1, j + 1].axis('off')
    
    plt.tight_layout()
    matrix_path = os.path.join(output_dir, 'style_matrix.png')
    plt.savefig(matrix_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"风格矩阵已保存: {matrix_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='批处理风格迁移')
    parser.add_argument('--content-dir', required=True, help='内容图像目录')
    parser.add_argument('--style-dir', required=True, help='风格图像目录')
    parser.add_argument('--output-dir', required=True, help='输出目录')
    parser.add_argument('--matrix', action='store_true', help='创建风格矩阵')
    parser.add_argument('--iterations', type=int, default=1000, help='迭代次数')
    parser.add_argument('--max-size', type=int, default=512, help='最大图像尺寸')
    
    args = parser.parse_args()
    
    # 创建配置模板
    config_template = Config()
    config_template.iterations = args.iterations
    config_template.max_size = args.max_size
    
    if args.matrix:
        content_files = glob.glob(os.path.join(args.content_dir, "*.jpg")) + \
                       glob.glob(os.path.join(args.content_dir, "*.png"))
        style_files = glob.glob(os.path.join(args.style_dir, "*.jpg")) + \
                     glob.glob(os.path.join(args.style_dir, "*.png"))
        
        create_style_matrix(content_files, style_files, args.output_dir, config_template)
    else:
        batch_style_transfer(args.content_dir, args.style_dir, args.output_dir, config_template)

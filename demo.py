"""
演示脚本 - 展示不同的风格迁移效果
"""

import os

import matplotlib.pyplot as plt
import torch

from config import Config
from neural_style import NeuralStyleTransfer
from utils import im_convert


def demo_basic_style_transfer():
    """基础风格迁移演示"""
    print("🎨 基础风格迁移演示")
    
    config = Config()
    config.iterations = 1000
    config.max_size = 512
    config.show_every = 200
    
    nst = NeuralStyleTransfer(config)
    nst.run()

def demo_parameter_comparison():
    """参数对比演示"""
    print("📊 参数对比演示")
    
    # 不同的风格权重
    style_weights = [1e4, 1e6, 1e8]
    results = []
    
    for weight in style_weights:
        print(f"测试风格权重: {weight}")
        
        config = Config()
        config.style_weight = weight
        config.iterations = 500  # 快速测试
        config.max_size = 256
        config.show_every = 1000  # 不显示中间过程
        
        nst = NeuralStyleTransfer(config)
        nst.load_images()
        nst.setup_losses()
        nst.optimize()
        
        results.append(nst.target_img.clone())
    
    # 显示对比结果
    fig, axes = plt.subplots(1, len(style_weights) + 2, figsize=(20, 4))
    
    # 内容图像
    axes[0].imshow(im_convert(nst.content_img))
    axes[0].set_title('Content')
    axes[0].axis('off')
    
    # 风格图像
    axes[1].imshow(im_convert(nst.style_img))
    axes[1].set_title('Style')
    axes[1].axis('off')
    
    # 不同权重的结果
    for i, (result, weight) in enumerate(zip(results, style_weights)):
        axes[i + 2].imshow(im_convert(result))
        axes[i + 2].set_title(f'Style Weight: {weight:.0e}')
        axes[i + 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('output/parameter_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def demo_color_preservation():
    """颜色保持演示"""
    print("🌈 颜色保持演示")
    
    from models import preserve_color
    
    config = Config()
    config.iterations = 1000
    config.max_size = 512
    config.show_every = 1000
    
    # 不保持颜色的结果
    nst1 = NeuralStyleTransfer(config)
    nst1.load_images()
    nst1.setup_losses()
    nst1.optimize()
    result_normal = nst1.target_img.clone()
    
    # 保持颜色的结果
    result_preserved = preserve_color(nst1.content_img, result_normal)
    
    # 显示对比
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(im_convert(nst1.content_img))
    axes[0].set_title('Content')
    axes[0].axis('off')
    
    axes[1].imshow(im_convert(nst1.style_img))
    axes[1].set_title('Style')
    axes[1].axis('off')
    
    axes[2].imshow(im_convert(result_normal))
    axes[2].set_title('Normal Transfer')
    axes[2].axis('off')
    
    axes[3].imshow(im_convert(result_preserved))
    axes[3].set_title('Color Preserved')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig('output/color_preservation_demo.png', dpi=150, bbox_inches='tight')
    plt.show()

def demo_multi_style():
    """多风格混合演示"""
    print("🎭 多风格混合演示")
    
    # 注意：这需要多个风格图像
    style_paths = [
        'data/style/style.jpg',
        # 可以添加更多风格图像
    ]
    
    if len(style_paths) < 2:
        print("⚠️  需要至少2个风格图像进行多风格演示")
        return
    
    # 实现多风格混合逻辑
    # 这里可以扩展实现多风格的混合效果
    print("多风格混合功能正在开发中...")

def demo_progressive_transfer():
    """渐进式风格迁移演示"""
    print("⏳ 渐进式风格迁移演示")
    
    config = Config()
    config.iterations = 2000
    config.max_size = 512
    config.show_every = 200
    config.save_intermediate = True
    config.save_every = 200
    
    nst = NeuralStyleTransfer(config)
    nst.load_images()
    nst.setup_losses()
    
    # 记录关键阶段
    key_iterations = [200, 500, 1000, 1500, 2000]
    results = []
    
    optimizer = torch.optim.Adam([nst.target_img], lr=config.learning_rate)
    
    for iteration in range(1, config.iterations + 1):
        # 标准优化步骤
        optimizer.zero_grad()
        
        target_features = nst.vgg_features(nst.target_img)
        
        content_loss = nst.content_loss(target_features[config.content_layer])
        content_loss *= config.content_weight
        
        style_loss = 0
        for layer, loss_info in nst.style_losses.items():
            if layer in target_features:
                layer_loss = loss_info['loss'](target_features[layer])
                style_loss += layer_loss * loss_info['weight']
        style_loss *= config.style_weight
        
        total_loss = content_loss + style_loss
        total_loss.backward()
        optimizer.step()
        
        # 保存关键阶段
        if iteration in key_iterations:
            results.append(nst.target_img.clone())
            print(f"保存第 {iteration} 次迭代结果")
    
    # 显示渐进过程
    fig, axes = plt.subplots(1, len(key_iterations) + 2, figsize=(4 * (len(key_iterations) + 2), 4))
    
    axes[0].imshow(im_convert(nst.content_img))
    axes[0].set_title('Content')
    axes[0].axis('off')
    
    axes[1].imshow(im_convert(nst.style_img))
    axes[1].set_title('Style')
    axes[1].axis('off')
    
    for i, (result, iteration) in enumerate(zip(results, key_iterations)):
        axes[i + 2].imshow(im_convert(result))
        axes[i + 2].set_title(f'Iter {iteration}')
        axes[i + 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('output/progressive_transfer.png', dpi=150, bbox_inches='tight')
    plt.show()

def run_all_demos():
    """运行所有演示"""
    print("🚀 开始运行所有演示...")
    
    # 确保输出目录存在
    os.makedirs('output', exist_ok=True)
    
    try:
        demo_basic_style_transfer()
        demo_parameter_comparison()
        demo_color_preservation()
        demo_progressive_transfer()
        
        print("✅ 所有演示完成！")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='神经风格迁移演示')
    parser.add_argument('--demo', choices=['basic', 'params', 'color', 'progressive', 'all'], 
                        default='all', help='选择演示类型')
    
    args = parser.parse_args()
    
    if args.demo == 'basic':
        demo_basic_style_transfer()
    elif args.demo == 'params':
        demo_parameter_comparison()
    elif args.demo == 'color':
        demo_color_preservation()
    elif args.demo == 'progressive':
        demo_progressive_transfer()
    elif args.demo == 'all':
        run_all_demos()

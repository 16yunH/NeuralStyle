"""
高级神经风格迁移实现
支持多种优化选项和可视化功能
"""

import os
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.optim as optim

warnings.filterwarnings('ignore')

from config import get_config
from models import VGGFeatures, ContentLoss, StyleLoss, TotalVariationLoss, preserve_color
from utils import (load_image, im_convert, save_image, create_gif, plot_progress,
                   get_timestamp, print_progress, validate_paths)

class NeuralStyleTransfer:
    """神经风格迁移类"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'使用设备: {self.device}')
        
        # 初始化模型
        self.vgg_features = VGGFeatures(self.device)
        
        # 损失记录
        self.losses = {'total': [], 'content': [], 'style': [], 'tv': []}
        
        # 中间结果
        self.intermediate_images = []
        
    def load_images(self):
        """加载内容和风格图像"""
        print("验证图像路径...")
        validate_paths(self.config.content_path, self.config.style_path)
        
        print(f"加载内容图像: {self.config.content_path}")
        self.content_img = load_image(
            self.config.content_path, 
            self.config.max_size, 
            device=self.device
        )
        
        print(f"加载风格图像: {self.config.style_path}")
        self.style_img = load_image(
            self.config.style_path, 
            shape=self.content_img.shape[-2:], 
            device=self.device
        )
        
        # 初始化目标图像
        self.target_img = self.content_img.clone().requires_grad_(True).to(self.device)
        
        print(f"图像尺寸: {self.content_img.shape[-2:]}")
        
    def setup_losses(self):
        """设置损失函数"""
        print("设置损失函数...")
        
        # 提取内容和风格特征
        content_features = self.vgg_features(self.content_img)
        style_features = self.vgg_features(self.style_img)
        
        # 内容损失
        self.content_loss = ContentLoss(content_features[self.config.content_layer])
        
        # 风格损失
        self.style_losses = {}
        for layer, weight in self.config.style_layers.items():
            if layer in style_features:
                self.style_losses[layer] = {
                    'loss': StyleLoss(style_features[layer]),
                    'weight': weight
                }
        
        # 总变分损失
        if self.config.total_variation_weight > 0:
            self.tv_loss = TotalVariationLoss(self.config.total_variation_weight)
        else:
            self.tv_loss = None
            
        print(f"内容层: {self.config.content_layer}")
        print(f"风格层: {list(self.style_losses.keys())}")
        
    def optimize(self):
        """执行优化过程"""
        print("开始风格迁移...")
        
        # 设置优化器
        optimizer = optim.Adam([self.target_img], lr=self.config.learning_rate)
        
        start_time = datetime.now()
        
        for iteration in range(1, self.config.iterations + 1):
            # 清零梯度
            optimizer.zero_grad()
            
            # 提取目标图像特征
            target_features = self.vgg_features(self.target_img)
            
            # 计算内容损失
            content_loss = self.content_loss(target_features[self.config.content_layer])
            content_loss *= self.config.content_weight
            
            # 计算风格损失
            style_loss = 0
            for layer, loss_info in self.style_losses.items():
                if layer in target_features:
                    layer_loss = loss_info['loss'](target_features[layer])
                    style_loss += layer_loss * loss_info['weight']
            style_loss *= self.config.style_weight
            
            # 计算总变分损失
            tv_loss = 0
            if self.tv_loss:
                tv_loss = self.tv_loss(self.target_img)
            
            # 总损失
            total_loss = content_loss + style_loss + tv_loss
            
            # 反向传播
            total_loss.backward()
            optimizer.step()
            
            # 记录损失
            self.losses['total'].append(total_loss.item())
            self.losses['content'].append(content_loss.item())
            self.losses['style'].append(style_loss.item())
            self.losses['tv'].append(tv_loss.item() if isinstance(tv_loss, torch.Tensor) else 0)
            
            # 显示进度
            if iteration % self.config.show_every == 0:
                print_progress(iteration, self.config.iterations, self.losses, start_time)
                
                # 可视化当前结果
                self.visualize_current_result()
            
            # 保存中间结果
            if self.config.save_intermediate and iteration % self.config.save_every == 0:
                self.save_intermediate_result(iteration)
                self.intermediate_images.append(self.target_img.clone())
        
        print(f"\n优化完成！总用时: {datetime.now() - start_time}")
        
    def visualize_current_result(self):
        """可视化当前结果"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 内容图像
        axes[0].imshow(im_convert(self.content_img))
        axes[0].set_title('Content Image')
        axes[0].axis('off')
        
        # 风格图像
        axes[1].imshow(im_convert(self.style_img))
        axes[1].set_title('Style Image')
        axes[1].axis('off')
        
        # 当前结果
        current_result = self.target_img.clone()
        if self.config.preserve_colors:
            current_result = preserve_color(self.content_img, current_result)
            
        axes[2].imshow(im_convert(current_result))
        axes[2].set_title('Current Result')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    def save_intermediate_result(self, iteration):
        """保存中间结果"""
        output_dir = os.path.join(self.config.output_dir, 'intermediate')
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f'iter_{iteration:04d}.jpg'
        filepath = os.path.join(output_dir, filename)
        
        current_result = self.target_img.clone()
        if self.config.preserve_colors:
            current_result = preserve_color(self.content_img, current_result)
            
        save_image(current_result, filepath)
        
    def save_final_result(self):
        """保存最终结果"""
        print("保存最终结果...")
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # 最终结果
        final_result = self.target_img.clone()
        if self.config.preserve_colors:
            final_result = preserve_color(self.content_img, final_result)
        
        # 保存主要结果
        timestamp = get_timestamp()
        main_output = os.path.join(self.config.output_dir, f'styled_{timestamp}.jpg')
        save_image(final_result, main_output, "最终结果已保存")
        
        # 保存对比图
        self.save_comparison(final_result, timestamp)
        
        # 保存损失曲线
        loss_plot_path = os.path.join(self.config.output_dir, f'losses_{timestamp}.png')
        plot_progress(self.losses, loss_plot_path)
        
        # 创建GIF（如果有中间结果）
        if self.intermediate_images:
            gif_path = os.path.join(self.config.output_dir, f'process_{timestamp}.gif')
            create_gif(self.intermediate_images, gif_path)
            print(f"处理过程GIF已保存: {gif_path}")
            
    def save_comparison(self, final_result, timestamp):
        """保存对比图"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(im_convert(self.content_img))
        axes[0].set_title('Content Image')
        axes[0].axis('off')
        
        axes[1].imshow(im_convert(self.style_img))
        axes[1].set_title('Style Image')
        axes[1].axis('off')
        
        axes[2].imshow(im_convert(final_result))
        axes[2].set_title('Result')
        axes[2].axis('off')
        
        plt.tight_layout()
        comparison_path = os.path.join(self.config.output_dir, f'comparison_{timestamp}.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"对比图已保存: {comparison_path}")
        
    def run(self):
        """执行完整的风格迁移流程"""
        try:
            self.load_images()
            self.setup_losses()
            self.optimize()
            self.save_final_result()
            
            print("\n🎨 神经风格迁移完成！")
            
        except Exception as e:
            print(f"❌ 错误: {e}")
            import traceback
            traceback.print_exc()

def main():
    """主函数"""
    # 获取配置
    config = get_config()
    
    # 创建并运行风格迁移
    nst = NeuralStyleTransfer(config)
    nst.run()

if __name__ == "__main__":
    main()

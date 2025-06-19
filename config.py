"""
配置文件 - 神经风格迁移项目
"""

import argparse


class Config:
    def __init__(self):
        # 默认配置
        self.content_path = 'data/content/content.jpg'
        self.style_path = 'data/style/style.jpg'
        self.output_dir = 'output'
        self.output_name = 'styled_image.jpg'
        
        # 模型参数
        self.max_size = 512
        self.style_weight = 1e6
        self.content_weight = 1
        self.iterations = 3000
        self.learning_rate = 0.003
        self.show_every = 500
        
        # 风格层权重
        self.style_layers = {
            'conv1_1': 1.0,
            'conv2_1': 0.8,
            'conv3_1': 0.5,
            'conv4_1': 0.3,
            'conv5_1': 0.1
        }
        
        # 内容层
        self.content_layer = 'conv4_2'
        
        # 高级选项
        self.preserve_colors = False
        self.total_variation_weight = 0
        self.save_intermediate = False
        self.save_every = 1000

def get_config():
    """解析命令行参数并返回配置对象"""
    parser = argparse.ArgumentParser(description='Neural Style Transfer')
    
    # 文件路径参数
    parser.add_argument('--content', type=str, help='内容图像路径')
    parser.add_argument('--style', type=str, help='风格图像路径')
    parser.add_argument('--output', type=str, help='输出文件路径')
    
    # 模型参数
    parser.add_argument('--max-size', type=int, default=512, help='图像最大尺寸')
    parser.add_argument('--style-weight', type=float, default=1e6, help='风格权重')
    parser.add_argument('--content-weight', type=float, default=1, help='内容权重')
    parser.add_argument('--iterations', type=int, default=3000, help='迭代次数')
    parser.add_argument('--lr', type=float, default=0.003, help='学习率')
    parser.add_argument('--show-every', type=int, default=500, help='显示间隔')
    
    # 高级选项
    parser.add_argument('--preserve-colors', action='store_true', help='保持原始颜色')
    parser.add_argument('--tv-weight', type=float, default=0, help='总变分权重')
    parser.add_argument('--save-intermediate', action='store_true', help='保存中间结果')
    parser.add_argument('--save-every', type=int, default=1000, help='保存间隔')
    
    args = parser.parse_args()
    config = Config()
    
    # 更新配置
    if args.content:
        config.content_path = args.content
    if args.style:
        config.style_path = args.style
    if args.output:
        config.output_name = args.output
    
    config.max_size = args.max_size
    config.style_weight = args.style_weight
    config.content_weight = args.content_weight
    config.iterations = args.iterations
    config.learning_rate = args.lr
    config.show_every = args.show_every
    config.preserve_colors = args.preserve_colors
    config.total_variation_weight = args.tv_weight
    config.save_intermediate = args.save_intermediate
    config.save_every = args.save_every
    
    return config

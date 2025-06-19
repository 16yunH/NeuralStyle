"""
神经风格迁移模型类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class VGGFeatures(nn.Module):
    """VGG特征提取器"""
    
    def __init__(self, device):
        super(VGGFeatures, self).__init__()
        self.device = device
        
        # 加载预训练VGG19
        vgg = models.vgg19(pretrained=True).features.to(device)
        
        # 冻结参数
        for param in vgg.parameters():
            param.requires_grad_(False)
        
        # 定义特征层
        self.layers = {
            '0': 'conv1_1',
            '5': 'conv2_1', 
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',
            '28': 'conv5_1'
        }
        
        self.model = vgg
        
    def forward(self, x, layers=None):
        """提取指定层的特征"""
        if layers is None:
            layers = self.layers
            
        features = {}
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
                
        return features

class GramMatrix(nn.Module):
    """Gram矩阵计算"""
    
    def forward(self, input_tensor):
        batch_size, depth, height, width = input_tensor.size()
        features = input_tensor.view(batch_size * depth, height * width)
        gram = torch.mm(features, features.t())
        # 归一化
        return gram.div(batch_size * depth * height * width)

class ContentLoss(nn.Module):
    """内容损失"""
    
    def __init__(self, target_features):
        super(ContentLoss, self).__init__()
        self.target = target_features.detach()
        
    def forward(self, input_features):
        return F.mse_loss(input_features, self.target)

class StyleLoss(nn.Module):
    """风格损失"""
    
    def __init__(self, target_features):
        super(StyleLoss, self).__init__()
        self.gram = GramMatrix()
        self.target = self.gram(target_features).detach()
        
    def forward(self, input_features):
        gram_input = self.gram(input_features)
        return F.mse_loss(gram_input, self.target)

class TotalVariationLoss(nn.Module):
    """总变分损失 - 用于平滑结果"""
    
    def __init__(self, weight=1):
        super(TotalVariationLoss, self).__init__()
        self.weight = weight
        
    def forward(self, input_tensor):
        batch_size, channels, height, width = input_tensor.size()
        
        # 水平变分
        tv_h = torch.pow(input_tensor[:, :, 1:, :] - input_tensor[:, :, :-1, :], 2).sum()
        # 垂直变分
        tv_w = torch.pow(input_tensor[:, :, :, 1:] - input_tensor[:, :, :, :-1], 2).sum()
        
        return self.weight * (tv_h + tv_w) / (batch_size * channels * height * width)

def preserve_color(content_img, stylized_img):
    """保持内容图像的原始颜色"""
    # 转换到YUV颜色空间
    content_yuv = rgb_to_yuv(content_img)
    stylized_yuv = rgb_to_yuv(stylized_img)
    
    # 使用内容的UV通道和风格化的Y通道
    result_yuv = torch.cat([
        stylized_yuv[:, 0:1, :, :],  # Y通道来自风格化图像
        content_yuv[:, 1:3, :, :]    # UV通道来自内容图像
    ], dim=1)
    
    return yuv_to_rgb(result_yuv)

def rgb_to_yuv(rgb_img):
    """RGB转YUV"""
    transformation_matrix = torch.tensor([
        [0.299, 0.587, 0.114],
        [-0.14714119, -0.28886916, 0.43601035],
        [0.61497538, -0.51496512, -0.10001026]
    ], device=rgb_img.device, dtype=rgb_img.dtype)
    
    yuv_img = torch.tensordot(rgb_img.permute(0, 2, 3, 1), transformation_matrix, dims=([3], [1]))
    return yuv_img.permute(0, 3, 1, 2)

def yuv_to_rgb(yuv_img):
    """YUV转RGB"""
    transformation_matrix = torch.tensor([
        [1.0, 0.0, 1.13988303],
        [1.0, -0.394642334, -0.58062185],
        [1.0, 2.03206185, 0.0]
    ], device=yuv_img.device, dtype=yuv_img.dtype)
    
    rgb_img = torch.tensordot(yuv_img.permute(0, 2, 3, 1), transformation_matrix, dims=([3], [1]))
    return torch.clamp(rgb_img.permute(0, 3, 1, 2), 0, 1)

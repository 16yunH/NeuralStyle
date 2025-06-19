# 🎨 神经风格迁移项目

一个功能强大的神经风格迁移（Neural Style Transfer）项目，基于PyTorch实现，支持多种运行模式和高级配置选项。

![项目横幅](https://img.shields.io/badge/Python-3.8+-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg) ![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ 特性

- 🖼️ **高质量风格迁移**：基于VGG网络的先进神经风格迁移算法
- 🌐 **Web界面**：简洁易用的Streamlit交互式界面
- 📦 **批量处理**：支持大批量图像的自动化处理
- 🎛️ **丰富参数**：可调节风格权重、内容权重、学习率等多种参数
- 🎨 **颜色保持**：可选择保持原始图像的颜色信息
- 📊 **进度可视化**：实时显示训练进度和损失曲线
- 🎬 **GIF生成**：自动生成风格迁移过程的动画
- 📈 **演示模式**：内置多种演示效果展示不同参数的影响

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.9+
- CUDA（可选，用于GPU加速）

### 安装依赖

```bash
# 克隆项目
git clone <repository-url>
cd NeuralStyle

# 安装依赖
pip install -r requirements.txt
```

### 基础使用

#### 1. 单个图像风格迁移

```bash
# 基础风格迁移
python main.py style --content data/content/content.jpg --style data/style/style.jpg

# 自定义参数
python main.py style \
  --content data/content/photo.jpg \
  --style data/style/art.jpg \
  --output output/my_result.jpg \
  --iterations 5000 \
  --style-weight 1e6 \
  --max-size 1024
```

#### 2. 启动Web界面

```bash
python main.py web
```

或者(最方便且直接的方法):

```bash
streamlit run web_interface.py
```

然后在浏览器中访问 `http://localhost:8501`

#### 3. 批量处理

```bash
# 批量处理多个图像
python main.py batch \
  --content-dir data/content/ \
  --style-dir data/style/ \
  --output-dir output/batch/

# 创建风格矩阵（所有内容和风格的组合）
python main.py batch \
  --content-dir data/content/ \
  --style-dir data/style/ \
  --output-dir output/matrix/ \
  --matrix
```

#### 4. 运行演示

```bash
# 运行所有演示
python main.py demo --demo all

# 运行特定演示
python main.py demo --demo basic      # 基础演示
python main.py demo --demo params     # 参数对比
python main.py demo --demo color      # 颜色保持演示
python main.py demo --demo progressive # 渐进式风格迁移
```

## 📁 项目结构

```text
NeuralStyle/
├── main.py              # 主入口文件
├── neural_style.py      # 核心风格迁移实现
├── models.py            # 神经网络模型定义
├── config.py            # 配置文件
├── utils.py             # 工具函数
├── web_interface.py     # Web界面
├── batch_process.py     # 批处理脚本
├── demo.py              # 演示脚本
├── requirements.txt     # 依赖包列表
├── data/
│   ├── content/         # 内容图像目录
│   │   └── content.jpg
│   └── style/           # 风格图像目录
│       └── style.jpg
└── output/              # 输出目录
    └── styled_image.jpg
```

## ⚙️ 配置参数

### 基础参数

| 参数           | 默认值                    | 说明         |
| -------------- | ------------------------- | ------------ |
| `--content`    | -                         | 内容图像路径 |
| `--style`      | -                         | 风格图像路径 |
| `--output`     | `output/styled_image.jpg` | 输出文件路径 |
| `--max-size`   | 512                       | 图像最大尺寸 |
| `--iterations` | 3000                      | 优化迭代次数 |

### 高级参数

| 参数                  | 默认值 | 说明             |
| --------------------- | ------ | ---------------- |
| `--style-weight`      | 1e6    | 风格损失权重     |
| `--content-weight`    | 1      | 内容损失权重     |
| `--lr`                | 0.003  | 学习率           |
| `--preserve-colors`   | False  | 是否保持原始颜色 |
| `--tv-weight`         | 0      | 总变分正则化权重 |
| `--save-intermediate` | False  | 是否保存中间结果 |

## 🎯 使用示例

### 示例1：艺术风格迁移

```bash
python main.py style \
  --content data/content/landscape.jpg \
  --style data/style/vangogh.jpg \
  --output output/landscape_vangogh.jpg \
  --iterations 4000 \
  --style-weight 5e5
```

### 示例2：保持颜色的风格迁移

```bash
python main.py style \
  --content data/content/portrait.jpg \
  --style data/style/abstract.jpg \
  --preserve-colors \
  --style-weight 1e6 \
  --tv-weight 1e-4
```

### 示例3：高分辨率处理

```bash
python main.py style \
  --content data/content/highres.jpg \
  --style data/style/painting.jpg \
  --max-size 1024 \
  --iterations 5000 \
  --lr 0.001
```

## 🌐 Web界面功能

Web界面提供了以下功能：

- 📤 **图像上传**：支持拖拽上传内容图像和风格图像
- 🎛️ **参数调节**：实时调整风格权重、迭代次数等参数
- 👀 **实时预览**：查看风格迁移的实时进度
- 💾 **结果下载**：直接下载生成的结果图像
- 📊 **进度监控**：显示训练进度和损失曲线

## 🔧 高级功能

### 1. 颜色保持

启用颜色保持功能可以在进行风格迁移时保持原始图像的颜色信息：

```bash
python main.py style --preserve-colors
```

### 2. 总变分正则化

添加总变分正则化可以减少结果图像的噪声：

```bash
python main.py style --tv-weight 1e-4
```

### 3. 中间结果保存

保存训练过程中的中间结果：

```bash
python main.py style --save-intermediate
```

### 4. 自定义风格层权重

可以在 `config.py` 中修改风格层的权重分布：

```python
style_layers = {
    'conv1_1': 1.0,   # 纹理细节
    'conv2_1': 0.8,   # 局部模式
    'conv3_1': 0.5,   # 中等复杂度特征
    'conv4_1': 0.3,   # 高级特征
    'conv5_1': 0.1    # 最高级特征
}
```

## 📊 性能优化

### GPU加速

项目自动检测并使用可用的GPU：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### 内存优化

- 自动调整图像尺寸以适应可用内存
- 支持渐进式处理大尺寸图像
- 可配置批处理大小

### 速度优化

- 使用预训练VGG网络提取特征
- 优化的损失函数计算
- 支持多种优化器选择

## 🛠️ 故障排除

### 常见问题

1. **CUDA内存不足**

   ```bash
   # 减小图像尺寸
   python main.py style --max-size 256
   ```

2. **依赖包冲突**

   ```bash
   # 创建虚拟环境
   python -m venv neural_style_env
   source neural_style_env/bin/activate  # Linux/Mac
   # 或
   neural_style_env\Scripts\activate     # Windows
   pip install -r requirements.txt
   ```

3. **Web界面启动失败**

   ```bash
   # 尝试指定端口
   python main.py web --port 8502
   ```

### 错误代码

- `错误代码 1`：图像路径不存在
- `错误代码 2`：GPU内存不足
- `错误代码 3`：依赖包版本不兼容

## 📝 更新日志

### v1.0.0 (2024-06-18)

- ✨ 初始版本发布
- 🎨 实现基础神经风格迁移功能
- 🌐 添加Web界面支持
- 📦 支持批量处理
- 🎬 添加演示模式

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

1. Fork 这个项目
2. 创建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启一个Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- 感谢 [PyTorch](https://pytorch.org/) 提供强大的深度学习框架
- 感谢 [Streamlit](https://streamlit.io/) 提供简洁的Web界面解决方案
- 感谢原始的神经风格迁移论文作者 Gatys et al.

## 📧 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 [Issue](https://github.com/your-username/NeuralStyle/issues)
- 发送邮件至：<your-email@example.com>

---

🎨 让艺术与科技完美融合 🎨

Made with ❤️ by [Your Name]

"""
Web界面 - 使用Streamlit创建交互式风格迁移应用
运行: streamlit run web_interface.py
"""

import streamlit as st

# 页面配置必须是第一个Streamlit命令
st.set_page_config(
    page_title="神经风格迁移",
    page_icon="🎨",
    layout="wide"
)

# 修复PyTorch和Streamlit的兼容性问题
import os

# 设置环境变量来避免PyTorch和Streamlit的冲突
os.environ['TORCH_FORCE_CPU_ONNX_EXPORT'] = '1'
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

# 禁用PyTorch的某些功能以避免与Streamlit冲突
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch')

# 延迟导入torch以避免初始化问题
@st.cache_resource
def load_torch():
    import torch
    torch.set_num_threads(1)  # 限制线程数以避免冲突
    return torch

torch = load_torch()

from PIL import Image
import io
import tempfile
from neural_style import NeuralStyleTransfer
from config import Config
from utils import im_convert
import numpy as np

def display_image_with_aspect_ratio(image, caption, max_width=400):
    """
    显示图像并保持宽高比
    
    Args:
        image: PIL Image对象
        caption: 图像标题
        max_width: 最大显示宽度
    """
    # 计算显示尺寸
    w, h = image.size
    if w > max_width:
        display_width = max_width
        display_height = int(h * max_width / w)
    else:
        display_width = w
        display_height = h
    
    # 创建一个调整后的图像用于显示
    display_image = image.resize((display_width, display_height), Image.Resampling.LANCZOS)
    
    st.image(display_image, caption=caption)
    return display_image

def validate_and_convert_image(image, image_type="图像"):
    """
    验证并转换图像格式，确保兼容性
    
    Args:
        image: PIL Image对象
        image_type: 图像类型描述（用于错误信息）
    
    Returns:
        PIL Image对象 (RGB格式)
    """
    try:
        # 检查图像模式
        if image.mode not in ['RGB', 'RGBA', 'L', 'P']:
            st.warning(f"⚠️ {image_type}格式可能不受支持，尝试转换为RGB格式")
        
        # 转换为RGB格式
        if image.mode != 'RGB':
            if image.mode == 'RGBA':
                # 对于RGBA图像，需要处理透明度
                # 创建白色背景
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])  # 使用alpha通道作为mask
                image = background
                st.info(f"✅ 已将{image_type}的透明背景转换为白色背景")
            else:
                image = image.convert('RGB')
                st.info(f"✅ 已将{image_type}转换为RGB格式")
        
        # 验证图像尺寸
        w, h = image.size
        if w < 32 or h < 32:
            st.error(f"❌ {image_type}尺寸过小（{w}x{h}），建议至少32x32像素")
            return None
        
        if w > 4096 or h > 4096:
            st.warning(f"⚠️ {image_type}尺寸很大（{w}x{h}），可能需要较长处理时间")
        
        return image
        
    except Exception as e:
        st.error(f"❌ 处理{image_type}时出错: {str(e)}")
        return None

# 标题
st.title("🎨 神经风格迁移")
st.markdown("---")

# 侧边栏配置
st.sidebar.header("⚙️ 参数设置")

# 基本参数
max_size = st.sidebar.slider("图像最大尺寸", 256, 1024, 512, 64)
iterations = st.sidebar.slider("迭代次数", 100, 5000, 1000, 100)
style_weight = st.sidebar.number_input("风格权重", 1e3, 1e8, 1e6, format="%.0e")
content_weight = st.sidebar.number_input("内容权重", 0.1, 10.0, 1.0, 0.1)
learning_rate = st.sidebar.number_input("学习率", 0.001, 0.01, 0.003, step=0.001, format="%.3f")

# 高级选项
st.sidebar.subheader("🔧 高级选项")
preserve_colors = st.sidebar.checkbox("保持原始颜色")
tv_weight = st.sidebar.number_input("总变分权重", 0.0, 100.0, 0.0, 0.1)
save_intermediate = st.sidebar.checkbox("保存中间结果")

# 主界面
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📷 内容图像")
    content_file = st.file_uploader("上传内容图像", type=['jpg', 'jpeg', 'png'], key="content")
    
    if content_file is not None:
        try:
            content_image = Image.open(content_file)
            content_image = validate_and_convert_image(content_image, "内容图像")
            if content_image is not None:
                display_image_with_aspect_ratio(content_image, "内容图像", 300)
        except Exception as e:
            st.error(f"❌ 加载内容图像失败: {str(e)}")
            content_image = None

with col2:
    st.subheader("🎭 风格图像")
    style_file = st.file_uploader("上传风格图像", type=['jpg', 'jpeg', 'png'], key="style")
    
    if style_file is not None:
        try:
            style_image = Image.open(style_file)
            style_image = validate_and_convert_image(style_image, "风格图像")
            if style_image is not None:
                display_image_with_aspect_ratio(style_image, "风格图像", 300)
        except Exception as e:
            st.error(f"❌ 加载风格图像失败: {str(e)}")
            style_image = None

with col3:
    st.subheader("🖼️ 结果图像")
    result_placeholder = st.empty()
    
    # 显示初始提示
    if content_file is None or style_file is None:
        result_placeholder.info("👆 请先上传内容图像和风格图像")
    else:
        result_placeholder.success("✅ 图像已准备就绪，点击下方按钮开始风格迁移")

# 处理按钮
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])

with col_btn2:
    if st.button("🚀 开始风格迁移", type="primary", use_container_width=True):
        # 检查图像是否已正确加载
        if content_file is not None and style_file is not None:
            # 重新验证图像（确保在按钮点击时图像仍然有效）
            try:
                content_image = Image.open(content_file)
                content_image = validate_and_convert_image(content_image, "内容图像")
                
                style_image = Image.open(style_file)
                style_image = validate_and_convert_image(style_image, "风格图像")
                
                if content_image is None or style_image is None:
                    st.error("❌ 图像验证失败，请重新上传有效的图像文件")
                    st.stop()
                
                # 创建进度条
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # 显示开始处理的提示
                result_placeholder.info("🚀 正在初始化神经风格迁移...")
                
                # 保存临时文件
                with tempfile.TemporaryDirectory() as temp_dir:
                    content_path = os.path.join(temp_dir, "content.jpg")
                    style_path = os.path.join(temp_dir, "style.jpg")
                    
                    # 获取原始内容图像的尺寸
                    original_content_size = content_image.size  # (width, height)
                    
                    # 图像已经在validate_and_convert_image函数中转换为RGB格式
                    content_image.save(content_path, 'JPEG')
                    style_image.save(style_path, 'JPEG')
                    
                    # 创建配置
                    config = Config()
                    config.content_path = content_path
                    config.style_path = style_path
                    config.output_dir = temp_dir
                    config.max_size = max_size
                    config.iterations = iterations
                    config.style_weight = style_weight
                    config.content_weight = content_weight
                    config.learning_rate = learning_rate
                    config.preserve_colors = preserve_colors
                    config.total_variation_weight = tv_weight
                    config.save_intermediate = save_intermediate
                    config.show_every = max(1, iterations // 20)  # 显示20次更新
                      # 执行风格迁移
                    status_text.text("🔄 正在加载神经网络模型...")
                    nst = NeuralStyleTransfer(config)
                    nst.load_images()
                    nst.setup_losses()
                    
                    status_text.text("🎨 开始风格迁移优化过程...")
                      # 自定义优化循环以显示进度
                    optimizer = torch.optim.Adam([nst.target_img], lr=config.learning_rate)
                      # 获取内容图像的原始尺寸
                    content_width, content_height = original_content_size  # PIL的size是(width, height)
                    
                    for iteration in range(1, config.iterations + 1):
                        optimizer.zero_grad()
                        
                        target_features = nst.vgg_features(nst.target_img)
                        
                        # 内容损失
                        content_loss = nst.content_loss(target_features[config.content_layer])
                        content_loss *= config.content_weight
                        
                        # 风格损失
                        style_loss = 0
                        for layer, loss_info in nst.style_losses.items():
                            if layer in target_features:
                                layer_loss = loss_info['loss'](target_features[layer])
                                style_loss += layer_loss * loss_info['weight']
                        style_loss *= config.style_weight
                        
                        # 总变分损失
                        tv_loss = 0
                        if nst.tv_loss:
                            tv_loss = nst.tv_loss(nst.target_img)
                        
                        total_loss = content_loss + style_loss + tv_loss
                        total_loss.backward()
                        optimizer.step()
                        
                        # 更新进度
                        progress = iteration / config.iterations
                        progress_bar.progress(progress)
                        
                        # 实时更新显示
                        if iteration % config.show_every == 0:
                            status_text.text(f"迭代 {iteration}/{config.iterations} - 损失: {total_loss.item():.4f}")
                            
                            # 显示中间结果
                            current_result = nst.target_img.clone()
                            if config.preserve_colors:
                                from models import preserve_color
                                current_result = preserve_color(nst.content_img, current_result)
                            
                            result_image = im_convert(current_result)
                            result_pil = Image.fromarray((result_image * 255).astype(np.uint8))
                            
                            # 调整结果图像尺寸与内容图像一致
                            result_pil = result_pil.resize((content_width, content_height), Image.Resampling.LANCZOS)
                              # 显示中间结果，保持原始尺寸比例
                            result_placeholder.empty()
                            with result_placeholder.container():
                                display_image_with_aspect_ratio(result_pil, f"实时结果 (迭代 {iteration})", 400)# 最终结果
                    final_result = nst.target_img.clone()
                    if config.preserve_colors:
                        from models import preserve_color
                        final_result = preserve_color(nst.content_img, final_result)
                    
                    result_image = im_convert(final_result)
                    result_pil = Image.fromarray((result_image * 255).astype(np.uint8))
                    
                    # 调整最终结果图像尺寸与内容图像一致
                    result_pil = result_pil.resize((content_width, content_height), Image.Resampling.LANCZOS)
                    
                    # 显示最终结果，保持原始尺寸比例
                    result_placeholder.empty()
                    with result_placeholder.container():
                        display_image_with_aspect_ratio(result_pil, "✨ 最终结果", 400)
                    
                    status_text.text("✅ 风格迁移完成！")
                    progress_bar.progress(1.0)
                    
                    # 提供下载按钮
                    img_buffer = io.BytesIO()
                    result_pil.save(img_buffer, format='JPEG')
                    img_buffer.seek(0)
                    
                    st.download_button(
                        label="💾 下载结果",
                        data=img_buffer.getvalue(),
                        file_name="neural_style_result.jpg",
                        mime="image/jpeg"
                    )                    
            except Exception as e:
                st.error(f"❌ 处理过程中发生错误: {str(e)}")
                st.error("💡 建议检查:")
                st.error("• 图像格式是否正确（支持JPG, JPEG, PNG）")
                st.error("• 图像文件是否完整（未损坏）")
                st.error("• 图像尺寸是否合理（不要过小或过大）")
                st.error("• 确保有足够的系统内存")
                
                # 重置进度条和状态
                if 'progress_bar' in locals():
                    progress_bar.progress(0.0)
                if 'status_text' in locals():
                    status_text.text("❌ 处理失败")
                
                result_placeholder.error("❌ 风格迁移失败，请尝试重新上传图像或调整参数")
                
        else:
            st.warning("⚠️ 请先上传内容图像和风格图像！")

# 示例图像
st.markdown("---")
st.subheader("🎨 示例图像")

example_col1, example_col2 = st.columns(2)

with example_col1:
    st.markdown("**内容图像示例**")
    if st.button("使用示例内容图像"):
        if os.path.exists("data/content/content.jpg"):
            st.session_state.example_content = Image.open("data/content/content.jpg")
            st.image(st.session_state.example_content, caption="示例内容图像")

with example_col2:
    st.markdown("**风格图像示例**")
    if st.button("使用示例风格图像"):
        if os.path.exists("data/style/style.jpg"):
            st.session_state.example_style = Image.open("data/style/style.jpg")
            st.image(st.session_state.example_style, caption="示例风格图像")

# 说明
st.markdown("---")
with st.expander("📖 使用说明"):
    st.markdown("""
    ### 如何使用神经风格迁移：
    
    1. **上传图像**：分别上传内容图像和风格图像
    2. **调整参数**：在左侧调整迭代次数、权重等参数
    3. **开始处理**：点击"开始风格迁移"按钮
    4. **查看结果**：等待处理完成并下载结果
    
    ### 参数说明：
    
    - **图像最大尺寸**：较大的尺寸会产生更高质量的结果，但需要更多时间
    - **迭代次数**：更多迭代通常产生更好的结果
    - **风格权重**：控制风格迁移的强度
    - **内容权重**：控制内容保持的程度
    - **保持原始颜色**：保持内容图像的原始颜色信息
    - **总变分权重**：减少结果中的噪声
    
    ### 提示：
    
    - 建议从较少的迭代次数开始测试
    - 高分辨率图像需要更多GPU内存
    - 可以尝试不同的参数组合获得最佳效果
    """)

# 页脚
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and PyTorch")

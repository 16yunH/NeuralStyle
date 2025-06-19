#!/usr/bin/env python3
"""
神经风格迁移项目主入口
"""

import argparse
import os
import sys


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='🎨 神经风格迁移项目',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py style --content data/content/photo.jpg --style data/style/art.jpg
  python main.py web
  python main.py batch --content-dir data/content/ --style-dir data/style/
  python main.py demo --demo all
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 风格迁移命令
    style_parser = subparsers.add_parser('style', help='单个风格迁移')
    style_parser.add_argument('--content', type=str, help='内容图像路径')
    style_parser.add_argument('--style', type=str, help='风格图像路径')
    style_parser.add_argument('--output', type=str, help='输出文件路径')
    style_parser.add_argument('--max-size', type=int, default=512, help='图像最大尺寸')
    style_parser.add_argument('--iterations', type=int, default=3000, help='迭代次数')
    style_parser.add_argument('--style-weight', type=float, default=1e6, help='风格权重')
    style_parser.add_argument('--content-weight', type=float, default=1, help='内容权重')
    style_parser.add_argument('--lr', type=float, default=0.003, help='学习率')
    style_parser.add_argument('--preserve-colors', action='store_true', help='保持原始颜色')
    style_parser.add_argument('--tv-weight', type=float, default=0, help='总变分权重')
    style_parser.add_argument('--save-intermediate', action='store_true', help='保存中间结果')
    
    # Web界面命令
    web_parser = subparsers.add_parser('web', help='启动Web界面')
    web_parser.add_argument('--port', type=int, default=8501, help='端口号')
    
    # 批处理命令
    batch_parser = subparsers.add_parser('batch', help='批量处理')
    batch_parser.add_argument('--content-dir', required=True, help='内容图像目录')
    batch_parser.add_argument('--style-dir', required=True, help='风格图像目录')
    batch_parser.add_argument('--output-dir', required=True, help='输出目录')
    batch_parser.add_argument('--matrix', action='store_true', help='创建风格矩阵')
    batch_parser.add_argument('--iterations', type=int, default=1000, help='迭代次数')
    batch_parser.add_argument('--max-size', type=int, default=512, help='最大图像尺寸')
    
    # 演示命令
    demo_parser = subparsers.add_parser('demo', help='运行演示')
    demo_parser.add_argument('--demo', choices=['basic', 'params', 'color', 'progressive', 'all'], 
                            default='all', help='演示类型')
    
    # 帮助命令
    help_parser = subparsers.add_parser('help', help='显示帮助信息')
    
    args = parser.parse_args()
    
    if args.command is None or args.command == 'help':
        parser.print_help()
        return
    
    try:
        if args.command == 'style':
            from neural_style import main as style_main
            # 设置sys.argv以便neural_style.py解析参数
            sys.argv = ['neural_style.py']
            if args.content:
                sys.argv.extend(['--content', args.content])
            if args.style:
                sys.argv.extend(['--style', args.style])
            if args.output:
                sys.argv.extend(['--output', args.output])
            sys.argv.extend(['--max-size', str(args.max_size)])
            sys.argv.extend(['--iterations', str(args.iterations)])
            sys.argv.extend(['--style-weight', str(args.style_weight)])
            sys.argv.extend(['--content-weight', str(args.content_weight)])
            sys.argv.extend(['--lr', str(args.lr)])
            if args.preserve_colors:
                sys.argv.append('--preserve-colors')
            if args.tv_weight > 0:
                sys.argv.extend(['--tv-weight', str(args.tv_weight)])
            if args.save_intermediate:
                sys.argv.append('--save-intermediate')
            
            style_main()
            
        elif args.command == 'web':
            print("🌐 启动Web界面...")
            try:
                # 尝试使用修复版本的启动脚本
                import subprocess
                result = subprocess.run([sys.executable, "run_web.py"], capture_output=True, text=True)
                if result.returncode != 0:
                    print("⚠️ 标准Web界面启动失败，尝试简化版本...")
                    os.system(f"streamlit run web_simple.py --server.port {args.port}")
            except Exception as e:
                print(f"❌ Web界面启动失败: {e}")
                print("💡 请尝试手动运行: python run_web.py 或 streamlit run web_simple.py")
            
        elif args.command == 'batch':
            from batch_process import batch_style_transfer, create_style_matrix
            from config import Config
            
            config_template = Config()
            config_template.iterations = args.iterations
            config_template.max_size = args.max_size
            
            if args.matrix:
                import glob
                content_files = glob.glob(os.path.join(args.content_dir, "*.jpg")) + \
                               glob.glob(os.path.join(args.content_dir, "*.png"))
                style_files = glob.glob(os.path.join(args.style_dir, "*.jpg")) + \
                             glob.glob(os.path.join(args.style_dir, "*.png"))
                
                create_style_matrix(content_files, style_files, args.output_dir, config_template)
            else:
                batch_style_transfer(args.content_dir, args.style_dir, args.output_dir, config_template)
                
        elif args.command == 'demo':
            from demo import (demo_basic_style_transfer, demo_parameter_comparison,
                            demo_color_preservation, demo_progressive_transfer, run_all_demos)
            
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
                
    except KeyboardInterrupt:
        print("\n⏹️  操作被用户中断")
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🎨 欢迎使用神经风格迁移项目！")
    print("=" * 50)
    main()

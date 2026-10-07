# Stable Diffusion 文生图模型实操作业

本仓库为《Stable Diffusion 文生图模型实操作业》的完整交付物，涵盖环境搭建、模型配置、命令行生图与问题总结反思等内容。

## 仓库内容

| 文件 | 说明 |
|------|------|
| `StableDiffusion文生图模型实操作业报告.docx` | 完整作业报告：任务 1 环境搭建与模型配置、任务 2 命令行生成图像、任务 3 问题总结与反思 |
| `batch_txt2img.sh` | 批量生成"同主题、不同参数（CFG × 扩散步数）"图像的脚本，共 8 组对比实验 |
| `generated-images/` | 生成效果示例图（水墨画风格风景画） |

## 作业要点

1. **环境搭建**：`git clone` 官方 stable-diffusion 代码库 → `conda env create -f environment.yaml` → `conda activate ldm` → `pip install diffusers==0.12.1`
2. **模型配置**：sd-v1-4.ckpt（主模型）、checkpoint_liberty_with_aug.pth（特征提取）、clip-vit-large-patch14（CLIP 文本编码器）、safety_checker（安全检查）
3. **命令行生图**：
   ```bash
   python scripts/txt2img.py --prompt "A landscape painting in the style of ink wash painting" --plms --ddim_steps 50 --scale 9.0
   ```
4. **批量实验**：
   ```bash
   bash batch_txt2img.sh
   ```

## 使用说明

- 报告为 Word 格式，可直接下载查看；
- `batch_txt2img.sh` 需在已配置好 Stable Diffusion 环境的代码库根目录下运行；
- 示例图对应报告"任务 2"生成效果（PLMS 采样，50 步，CFG=9.0）。

## 作者信息

- 姓名：________
- 学号：________
- 日期：2026-10-07

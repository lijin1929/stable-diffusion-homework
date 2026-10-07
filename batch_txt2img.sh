#!/bin/bash
# ==========================================================
# batch_txt2img.sh —— 批量生成"同主题、不同参数"的水墨风景图
# 使用场景：需要批量生成 10 张同主题、不同参数的图像进行对比实验时，
#           通过脚本循环执行命令，无需在 Web 界面重复手动输入参数。
# 运行方式：bash batch_txt2img.sh
# ==========================================================

PROMPT="A landscape painting in the style of ink wash painting"

# 实验变量 1：提示词服从度（CFG Scale）
for scale in 5.0 7.0 9.0 11.0; do
  # 实验变量 2：扩散步数（DDIM Steps）
  for steps in 25 50; do
    outdir="outputs/inkwash/scale_${scale}/steps_${steps}"
    echo ">>> Generating: scale=${scale}, steps=${steps} -> ${outdir}"
    python scripts/txt2img.py \
      --prompt "${PROMPT}" \
      --plms \
      --ddim_steps ${steps} \
      --scale ${scale} \
      --n_samples 1 \
      --outdir "${outdir}"
  done
done

echo "全部 8 组图像生成完毕，结果保存在 outputs/inkwash/ 目录下。"

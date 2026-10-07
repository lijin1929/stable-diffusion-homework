# -*- coding: utf-8 -*-
"""
Stable Diffusion Web 界面（文生图 / 图生图）
用法：python app.py  →  浏览器打开控制台打印的 gradio.live 链接
依赖：diffusers==0.12.1  transformers==4.26.0  huggingface_hub==0.16.4  gradio==3.50.2
模型：webui_model/ 目录（diffusers 格式，下载方式见《部署指南-课程算力平台.md》）
"""
import os
import torch
import gradio as gr
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

# 模型目录：优先环境变量 SD_MODEL_DIR；否则取脚本所在目录下的 webui_model。
# 用绝对路径可保证在任意工作目录下运行都能找到本地模型（否则会被当成 HuggingFace 仓库名联网下载，报 401）
MODEL_DIR = os.environ.get("SD_MODEL_DIR") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "webui_model"
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# CPU 用 fp32（diffusers 0.12.1 不支持 fp16 CPU 推理）；需已安装 accelerate 以低内存方式加载
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
PORT = int(os.environ.get("SD_PORT", "6006"))
SHARE = os.environ.get("SD_SHARE", "1") == "1"

if not os.path.isdir(MODEL_DIR):
    raise SystemExit(
        "[app] 找不到模型目录：%s\n"
        "      请在包含 webui_model 的目录下运行，或指定绝对路径：\n"
        "      SD_MODEL_DIR=/workspace/homework/webui_model python app.py" % MODEL_DIR
    )

print(f"[app] 加载模型 {MODEL_DIR}（设备: {DEVICE}）...")


def _load_pipeline():
    """优先完整加载；若安全检查器权重命名不兼容（如 fp16 分支），自动跳过安全检查器重试"""
    try:
        return StableDiffusionPipeline.from_pretrained(MODEL_DIR, torch_dtype=DTYPE).to(DEVICE)
    except OSError as e:
        print("[app] 完整加载失败，跳过 safety_checker 重试：", e)
        return StableDiffusionPipeline.from_pretrained(
            MODEL_DIR, torch_dtype=DTYPE, safety_checker=None, feature_extractor=None
        ).to(DEVICE)


pipe = _load_pipeline()

if DEVICE == "cpu":
    # CPU 模式：注意力切片可显著降低内存峰值（8GB 内存环境必备）
    pipe.enable_attention_slicing()
    print("[app] CPU 模式：已启用 attention slicing")

# 使用 PLMS 采样器，与作业 txt2img.py 的 --plms 参数保持一致
try:
    from diffusers import PLMSScheduler
    pipe.scheduler = PLMScheduler.from_config(pipe.scheduler.config)
    print("[app] 已切换采样器: PLMS")
except Exception as e:
    print("[app] 使用默认采样器（PNDM）:", e)

# 复用组件构建图生图管线（同一套模型权重，不额外占用显存）
img2img_pipe = StableDiffusionImg2ImgPipeline(
    vae=pipe.vae,
    text_encoder=pipe.text_encoder,
    tokenizer=pipe.tokenizer,
    unet=pipe.unet,
    scheduler=pipe.scheduler,
    safety_checker=getattr(pipe, "safety_checker", None),
    feature_extractor=getattr(pipe, "feature_extractor", None),
)


def _generator(seed):
    if seed is None or int(seed) < 0:
        return None
    return torch.Generator(device=DEVICE).manual_seed(int(seed))


DEFAULT_RES = 512 if DEVICE == "cuda" else 320
DEFAULT_STEPS = 50 if DEVICE == "cuda" else 15


def txt2img_fn(prompt, negative_prompt, resolution, steps, scale, seed):
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        width=int(resolution),
        height=int(resolution),
        num_inference_steps=int(steps),
        guidance_scale=float(scale),
        generator=_generator(seed),
    ).images[0]
    return image


def img2img_fn(prompt, negative_prompt, init_image, resolution, steps, scale, strength, seed):
    if init_image is None:
        raise gr.Error("请先上传一张参考图（init image）")
    # CPU/低内存环境：把参考图长边缩放到目标分辨率，避免内存爆掉
    res = int(resolution)
    if max(init_image.size) > res:
        ratio = res / max(init_image.size)
        init_image = init_image.resize((max(1, round(init_image.width * ratio)),
                                        max(1, round(init_image.height * ratio))))
    image = img2img_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        image=init_image,
        num_inference_steps=int(steps),
        guidance_scale=float(scale),
        strength=float(strength),
        generator=_generator(seed),
    ).images[0]
    return image


with gr.Blocks(title="Stable Diffusion Web 界面") as demo:
    gr.Markdown(
        "# Stable Diffusion 文生图 / 图生图\n"
        "文生图：直接通过文本提示词生成全新图像；"
        "图生图：上传参考图 + 提示词调整，`strength` 越大改动越大。"
    )
    with gr.Tab("文生图 txt2img"):
        with gr.Row():
            with gr.Column():
                t_prompt = gr.Textbox(label="提示词 Prompt", value="A landscape painting in the style of ink wash painting", lines=2)
                t_neg = gr.Textbox(label="负向提示词（可空）", value="")
                t_res = gr.Slider(256, 512, value=DEFAULT_RES, step=64, label="分辨率（CPU 建议 384）")
                t_steps = gr.Slider(10, 100, value=DEFAULT_STEPS, step=1, label="扩散步数 Steps")
                t_scale = gr.Slider(1.0, 20.0, value=9.0, step=0.5, label="提示词服从度 CFG Scale")
                t_seed = gr.Number(value=-1, label="随机种子（-1 为随机）")
                t_btn = gr.Button("生成", variant="primary")
            t_out = gr.Image(label="生成结果")
        t_btn.click(txt2img_fn, [t_prompt, t_neg, t_res, t_steps, t_scale, t_seed], t_out)

    with gr.Tab("图生图 img2img"):
        with gr.Row():
            with gr.Column():
                i_prompt = gr.Textbox(label="提示词 Prompt", value="an oil painting style of this landscape", lines=2)
                i_neg = gr.Textbox(label="负向提示词（可空）", value="")
                i_init = gr.Image(label="参考图（必传）", type="pil")
                i_res = gr.Slider(256, 512, value=DEFAULT_RES, step=64, label="分辨率（CPU 建议 384）")
                i_strength = gr.Slider(0.1, 1.0, value=0.75, step=0.05, label="改动强度 Strength")
                i_steps = gr.Slider(10, 100, value=DEFAULT_STEPS, step=1, label="扩散步数 Steps")
                i_scale = gr.Slider(1.0, 20.0, value=9.0, step=0.5, label="提示词服从度 CFG Scale")
                i_seed = gr.Number(value=-1, label="随机种子（-1 为随机）")
                i_btn = gr.Button("生成", variant="primary")
            i_out = gr.Image(label="生成结果")
        i_btn.click(img2img_fn, [i_prompt, i_neg, i_init, i_res, i_steps, i_scale, i_strength, i_seed], i_out)

demo.queue()
demo.launch(server_name="0.0.0.0", server_port=PORT, share=SHARE)

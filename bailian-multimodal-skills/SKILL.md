---
name: bailian-multimodal-skills
description: Generate images, video, speech, and transcribe audio using Aliyun Bailian models.
homepage: https://dashscope.aliyun.com
metadata:
  {
    "openclaw":
      {
        "emoji": "🎭",
        "requires": { "bins": ["uv"], "env": ["DASHSCOPE_API_KEY"] },
        "primaryEnv": "DASHSCOPE_API_KEY",
        "install":
          [
            {
              "id": "uv-brew",
              "kind": "brew",
              "formula": "uv",
              "bins": ["uv"],
              "label": "Install uv (brew)",
            },
            {
              "id": "uv-curl",
              "kind": "download",
              "url": "https://astral.sh/uv/install.sh",
              "bins": ["uv"],
              "label": "Install uv (curl)",
            },
          ],
      },
  }
---

# Bailian Multimodal Skills

Generate images, audio, video, and transcribe speech using Aliyun Bailian (Qwen/Wan/PixVerse/Kling/CosyVoice) models.

## Features

- **Image Generation**: `z-image-turbo`, `wan2.6-t2i`, `wan2.7-image-pro`
- **Image Editing**: `wan2.7-image-pro`
- **Video Editing**: `wan2.7-videoedit`
- **ASR (Speech-to-Text)**: `qwen3-asr-flash`
- **TTS (Text-to-Speech)**: `qwen3-tts-flash`
- **Text-to-Video**: `wan2.7-t2v`, `wan2.6-t2v`, `pixverse/pixverse-v5.6-t2v`, `kling/kling-v3-video-generation`
- **Image-to-Video**: `wan2.6-i2v-flash`, `wan2.6-i2v`, `pixverse/pixverse-v5.6-it2v`, `kling/kling-v3-video-generation`
- **Reference-to-Video**: `wan2.6-r2v-flash`, `wan2.6-r2v`, `pixverse/pixverse-v5.6-r2v`

## Usage

### 1. Image Generation

Generate images from text.

```bash
uv run {baseDir}/scripts/run_multimodal.py --mode image --model z-image-turbo --prompt "A futuristic city" --output "city.png"
```

Models: `z-image-turbo`, `wan2.6-t2i`, `wan2.7-image-pro`

`wan2.7-image-pro` supports multi-image sequential generation and `2K` size:

```bash
uv run {baseDir}/scripts/run_multimodal.py --mode image --model wan2.7-image-pro --prompt "电影感组图，记录同一只流浪橘猫的四季" --n 4 --enable-sequential --size 2K --output "cat_seasons.png"
```

Options (wan2.7-image-pro): `--n` (number of images), `--enable-sequential` (keep character consistency across images), `--size` (e.g., 2K, 1024*1024), `--watermark`

### 2. Image Editing

Edit images with text instructions. Supports multiple input images.

```bash
uv run {baseDir}/scripts/run_multimodal.py --mode image-edit --model wan2.7-image-pro --input-images "car.png" "graffiti.png" --prompt "把图2的涂鸦喷绘在图1的汽车上" --output "edited.png"
```

Options: `--input-images` (required, one or more image URLs/local paths), `--prompt` (required, editing instruction), `--size` (default: 2K), `--n`, `--watermark`, `--thinking-mode` (enable deeper reasoning for complex edits)

### 3. ASR (Speech Recognition)

Transcribe audio files or URLs to text.

```bash
uv run {baseDir}/scripts/run_multimodal.py --mode asr --model qwen3-asr-flash --input-audio "https://example.com/audio.mp3"
```

### 4. TTS (Speech Synthesis)

Convert text to speech.

```bash
uv run {baseDir}/scripts/run_multimodal.py --mode tts --model qwen3-tts-flash --text "Hello world" --output "hello.wav"
```

### 5. Text-to-Video (T2V)

Generate video from text prompt. Async task with auto-polling.

**wan2.7-t2v** (new protocol — uses `resolution` + `ratio` instead of `size`, no `shot_type`):

```bash
uv run {baseDir}/scripts/run_multimodal.py --mode t2v --model wan2.7-t2v --prompt "一只小猫在月光下奔跑" --resolution 1080P --ratio 16:9 --duration 10 --output "cat.mp4"
```

Options (wan2.7-t2v): `--resolution` (720P/1080P, default 1080P), `--ratio` (16:9/9:16/1:1, default 16:9), `--duration`, `--prompt-extend`/`--no-prompt-extend`, `--negative-prompt`, `--audio-url`, `--audio`/`--no-audio`, `--watermark`, `--seed`

> Note: `--shot-type` is **not** supported for wan2.7-t2v. Use natural language in `--prompt` to describe shot structure (e.g., "生成多镜头视频" or timestamp-based descriptions).

**wan2.6-t2v and earlier** (legacy protocol — uses `size` and `shot-type`):

```bash
uv run {baseDir}/scripts/run_multimodal.py --mode t2v --model wan2.6-t2v --prompt "一只小猫在月光下奔跑" --duration 10 --size "1280*720" --output "cat.mp4"
```

Models: `wan2.6-t2v`, `pixverse/pixverse-v5.6-t2v`, `kling/kling-v3-video-generation`

Options: `--size` (e.g., 1280*720, 1920*1080), `--duration`, `--prompt-extend`/`--no-prompt-extend`, `--shot-type single|multi`, `--negative-prompt`, `--audio-url`, `--audio`/`--no-audio`, `--watermark`, `--seed`, `--quality-mode std|pro`

### 5.5 Video Editing (videoedit)

Edit an existing video with text instructions (style transfer, content modification, etc.).

```bash
uv run {baseDir}/scripts/run_multimodal.py --mode videoedit --model wan2.7-videoedit \
  --video-url "https://example.com/input.mp4" \
  --prompt "将整个画面转换为黏土风格" \
  --resolution 1080P \
  --output "edited.mp4"
```

With reference images (for appearance/style guidance):

```bash
uv run {baseDir}/scripts/run_multimodal.py --mode videoedit --model wan2.7-videoedit \
  --video-url "clip.mp4" \
  --prompt "为人物换上参考图里的服装" \
  --ref-images "outfit.png" \
  --resolution 1080P \
  --output "result.mp4"
```

Models: `wan2.7-videoedit`

Options:
- `--video-url` (required): input video URL or local file path
- `--prompt` (optional): editing instruction (up to 5000 chars)
- `--negative-prompt` (optional): content to avoid
- `--ref-images` (optional): one or more reference image URLs/local paths
- `--resolution` (720P/1080P, default 1080P)
- `--ratio` (16:9/9:16/1:1; omit to preserve input video's ratio)
- `--prompt-extend`/`--no-prompt-extend` (default: enabled)
- `--watermark`, `--seed`

> Note: Billing = input video duration + output video duration (seconds).

### 6. Image-to-Video (I2V)
Generate video from a reference image (first frame).

```bash
uv run {baseDir}/scripts/run_multimodal.py --mode i2v --model wan2.6-i2v-flash --img-url "https://example.com/cat.png" --prompt "A cat running" --resolution 720P --duration 5 --output "cat_run.mp4"
```

Models: `wan2.6-i2v-flash`, `wan2.6-i2v`, `pixverse/pixverse-v5.6-it2v`, `kling/kling-v3-video-generation`

Options: `--img-url` (required, image URL, base64, or local file path), `--prompt`, `--resolution` (480P/720P/1080P), `--duration`, `--prompt-extend`/`--no-prompt-extend`, `--shot-type single|multi`, `--negative-prompt`, `--audio-url`, `--audio`/`--no-audio`, `--watermark`, `--seed`, `--quality-mode std|pro`

### 7. Reference-to-Video (R2V)

Generate video with character/object references (images or videos as actors).

```bash
uv run {baseDir}/scripts/run_multimodal.py --mode r2v --model wan2.6-r2v-flash --prompt "character1 在公园里散步" --reference-urls "https://example.com/person.png" --size "1280*720" --duration 5 --output "walk.mp4"
```

Multi-character example:
```bash
uv run {baseDir}/scripts/run_multimodal.py --mode r2v --model wan2.6-r2v-flash --prompt "character1 对 character2 说你好" --reference-urls "https://example.com/role1.mp4" "https://example.com/role2.png" --shot-type multi --output "dialog.mp4"
```

Models: `wan2.6-r2v-flash`, `wan2.6-r2v`, `pixverse/pixverse-v5.6-r2v`

Options: `--reference-urls` (required, space-separated, up to 5, supports local file paths), `--prompt` (required, use character1/character2 to map references), `--size`, `--duration`, `--shot-type single|multi`, `--negative-prompt`, `--no-audio`, `--watermark`, `--seed`

## Notes

- **本地文件处理**：脚本自动处理本地路径。wan2.7-videoedit 模式会将本地视频/图片上传到 DashScope OSS 获取 `oss://` URL；其他模式（image/i2v/r2v 等）使用 SDK 自带的 `file://` URL 传入。
- PixVerse 与 Kling 模型以 URL 方式提交媒体；脚本已兼容本地路径输入，但若百炼后端对某些模型拒绝 `file://`，请改用可公网访问的 HTTP/HTTPS URL。

## Configuration

API Key 按以下优先级读取：

1. 命令行参数 `--api-key`
2. 环境变量 `DASHSCOPE_API_KEY`
3. 配置文件 `~/.config/bailian-multimodal/api_key.txt`

```bash
# 方式一：环境变量
export DASHSCOPE_API_KEY="sk-..."

# 方式二：配置文件
mkdir -p ~/.config/bailian-multimodal
echo "sk-..." > ~/.config/bailian-multimodal/api_key.txt
```

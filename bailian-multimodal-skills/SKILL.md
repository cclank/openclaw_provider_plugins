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

Generate images, audio, video, and transcribe speech using Aliyun Bailian (Qwen/Wan/CosyVoice) models.

## Features

- **Image Generation**: `z-image-turbo`, `wan2.6-t2i`
- **ASR (Speech-to-Text)**: `qwen3-asr-flash`
- **TTS (Text-to-Speech)**: `qwen3-tts-flash`
- **Text-to-Video**: `wan2.6-t2v`
- **Image-to-Video**: `wan2.6-i2v-flash`, `wan2.6-i2v`
- **Reference-to-Video**: `wan2.6-r2v-flash`, `wan2.6-r2v`

## Usage

### 1. Image Generation

Generate images from text.

```bash
uv run {baseDir}/scripts/run_multimodal.py --mode image --model z-image-turbo --prompt "A futuristic city" --output "city.png"
```

Models: `z-image-turbo`, `wan2.6-t2i`

### 2. ASR (Speech Recognition)

Transcribe audio files or URLs to text.

```bash
uv run {baseDir}/scripts/run_multimodal.py --mode asr --model qwen3-asr-flash --input-audio "https://example.com/audio.mp3"
```

### 3. TTS (Speech Synthesis)

Convert text to speech.

```bash
uv run {baseDir}/scripts/run_multimodal.py --mode tts --model qwen3-tts-flash --text "Hello world" --output "hello.wav"
```

### 4. Text-to-Video (T2V)

Generate video from text prompt. Async task with auto-polling.

```bash
uv run {baseDir}/scripts/run_multimodal.py --mode t2v --model wan2.6-t2v --prompt "一只小猫在月光下奔跑" --duration 10 --size "1280*720" --output "cat.mp4"
```

Models: `wan2.6-t2v`

Options: `--size` (e.g., 1280*720, 1920*1080), `--duration` (2-15s), `--prompt-extend`/`--no-prompt-extend`, `--shot-type single|multi`, `--negative-prompt`, `--audio-url`, `--watermark`, `--seed`

### 5. Image-to-Video (I2V)

Generate video from a reference image (first frame).

```bash
uv run {baseDir}/scripts/run_multimodal.py --mode i2v --model wan2.6-i2v-flash --img-url "https://example.com/cat.png" --prompt "A cat running" --resolution 720P --duration 5 --output "cat_run.mp4"
```

Models: `wan2.6-i2v-flash`, `wan2.6-i2v`

Options: `--img-url` (required, image URL or base64), `--prompt`, `--resolution` (480P/720P/1080P), `--duration`, `--prompt-extend`/`--no-prompt-extend`, `--shot-type single|multi`, `--negative-prompt`, `--audio-url`, `--watermark`, `--seed`

### 6. Reference-to-Video (R2V)

Generate video with character/object references (images or videos as actors).

```bash
uv run {baseDir}/scripts/run_multimodal.py --mode r2v --model wan2.6-r2v-flash --prompt "character1 在公园里散步" --reference-urls "https://example.com/person.png" --size "1280*720" --duration 5 --output "walk.mp4"
```

Multi-character example:
```bash
uv run {baseDir}/scripts/run_multimodal.py --mode r2v --model wan2.6-r2v-flash --prompt "character1 对 character2 说你好" --reference-urls "https://example.com/role1.mp4" "https://example.com/role2.png" --shot-type multi --output "dialog.mp4"
```

Models: `wan2.6-r2v-flash`, `wan2.6-r2v`

Options: `--reference-urls` (required, space-separated, up to 5), `--prompt` (required, use character1/character2 to map references), `--size`, `--duration` (2-10s), `--shot-type single|multi`, `--negative-prompt`, `--no-audio` (silent, r2v-flash only), `--watermark`, `--seed`

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

---
name: zenmux-image-gen-skill
description: Generate images via Zenmux Vertex AI API using Gemini image models.
homepage: https://zenmux.ai
metadata:
  {
    "openclaw":
      {
        "emoji": "🎨",
        "requires": { "bins": ["uv"], "env": ["ZENMUX_API_KEY"] },
        "primaryEnv": "ZENMUX_API_KEY",
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

# Zenmux Image Gen Skill

Generate images using Zenmux's Vertex AI API with Gemini image models.

## Features

- Text-to-image generation
- Single image editing
- Multi-image composition (up to 14 images)
- Multiple Gemini model support

## Generate

```bash
uv run {baseDir}/scripts/generate_image.py --prompt "your image description" --filename "output.png"
```

## Edit (single image)

```bash
uv run {baseDir}/scripts/generate_image.py --prompt "edit instructions" --filename "output.png" -i "/path/to/input.png"
```

## Multi-image composition (up to 14 images)

```bash
uv run {baseDir}/scripts/generate_image.py --prompt "combine these into one scene" --filename "output.png" -i img1.png -i img2.png -i img3.png
```

## Available Models

| Model | Description |
|-------|-------------|
| `google/gemini-3-pro-image-preview` | Default, highest quality |
| `google/gemini-3-pro-image-preview-free` | Free tier, limited quota |
| `google/gemini-2.5-flash-image` | Faster generation |
| `google/gemini-2.5-flash-image-free` | Free tier, faster |

Specify model with `--model`:

```bash
uv run {baseDir}/scripts/generate_image.py --prompt "a sunset over mountains" --filename "sunset.png" --model "google/gemini-2.5-flash-image"
```

## API Key Configuration

1. **Environment variable** (recommended):
   ```bash
   export ZENMUX_API_KEY="your-api-key"
   ```

2. **OpenClaw config**:
   ```bash
   openclaw config set skills.zenmux-image-gen-skill.apiKey "your-api-key"
   ```

3. **Command line argument**:
   ```bash
   uv run {baseDir}/scripts/generate_image.py --api-key "your-api-key" --prompt "..." --filename "..."
   ```

## Notes

- Use timestamps in filenames: `yyyy-mm-dd-hh-mm-ss-name.png`.
- The script prints a `MEDIA:` line with the file path.
- **Important**: To send the image to chat channels (Telegram, Discord, etc.), use the `message` tool with `filePath` parameter. The `MEDIA:` line alone will NOT auto-attach the image as a real attachment.
- Do not read the image back; report the saved path only.

#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "openai>=1.0.0",
#     "dashscope>=1.25.8",
#     "requests>=2.31.0",
#     "pillow>=10.0.0",
# ]
# ///

import argparse
import os
import sys
import json
import base64
import requests
from pathlib import Path

# Fix encoding issues
import locale
locale.setlocale(locale.LC_ALL, 'C.UTF-8')
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# --- Constants ---
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"


def get_api_key(provided_key: str | None) -> str:
    key = provided_key or os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        # Try to read from config file
        config_path = os.path.expanduser("~/.config/bailian-multimodal/api_key.txt")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                key = f.read().strip()
    if not key:
        print("Error: DASHSCOPE_API_KEY not found. Set it in environment, pass --api-key, or create ~/.config/bailian-multimodal/api_key.txt.", file=sys.stderr)
        sys.exit(1)
    return key


# --- Local file helper ---
def _to_file_url(path_or_url: str) -> str:
    """Convert a local file path to file:// URL; leave URLs as-is."""
    if path_or_url.startswith(("http://", "https://", "oss://", "data:", "file://")):
        return path_or_url
    abs_path = os.path.abspath(path_or_url)
    if not os.path.exists(abs_path):
        print(f"Error: File not found: {abs_path}", file=sys.stderr)
        sys.exit(1)
    return f"file://{abs_path}"


# --- Image Generation ---
def generate_image(api_key: str, model: str, prompt: str, output_path: str, size: str = "1024*1024"):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        # "X-DashScope-Async": "enable",  # Commented out because account doesn't support async
    }
    
    # Model-specific parameters
    payload = {
        "model": model,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": prompt}]
                }
            ]
        },
        "parameters": {}
    }

    if model == "z-image-turbo":
         payload["parameters"] = {
             "prompt_extend": False,
             "size": size if size else "1024*1024" # Default for z-image-turbo
         }
    elif model == "wan2.6-t2i":
        payload["parameters"] = {
            "prompt_extend": True,
            "watermark": False,
            "n": 1,
            "size": size if size else "1280*1280" # Default for wan2.6
        }

    print(f"Generating image with {model}...", file=sys.stderr)
    try:
        # Ensure payload uses UTF-8 encoding
        import json as json_module
        payload_bytes = json_module.dumps(payload, ensure_ascii=False).encode('utf-8')
        headers['Content-Type'] = 'application/json; charset=utf-8'
        response = requests.post(DASHSCOPE_API_URL, headers=headers, data=payload_bytes)
        response.raise_for_status()
        result = response.json()
        
        # Check for immediate failure
        if "code" in result and result["code"]:
             print(f"API Error: {result['message']}", file=sys.stderr)
             sys.exit(1)

        # Retrieve image URL
        # Note: wan2.6 might be async? User example shows url in response directly for wan2.6 but z-image-turbo example also looks synchronous-ish or task based.
        # Actually user example for z-image-turbo returns output.choices[0].message.content[0].image
        # User example for wan2.6 returns output.choices[0].message.content[0].image
        
        # Let's handle the response structure as per user example
        if "output" in result and "choices" in result["output"]:
            choices = result["output"]["choices"]
            if choices:
                content = choices[0]["message"]["content"]
                image_url = None
                for item in content:
                    if "image" in item:
                        image_url = item["image"]
                        break
                
                if image_url:
                    print(f"Downloading image from {image_url}...", file=sys.stderr)
                    img_data = requests.get(image_url).content
                    with open(output_path, "wb") as f:
                        f.write(img_data)
                    print(f"Image saved to {output_path}")
                    print(f"MEDIA: {os.path.abspath(output_path)}")
                    return
        
        print(f"Unexpected response format or no image found: {json.dumps(result, indent=2)}", file=sys.stderr)
        sys.exit(1)

    except Exception as e:
        print(f"Error generating image: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


# --- ASR ---
def run_asr(api_key: str, model: str, input_audio: str):
    from openai import OpenAI
    
    # Determine if input is URL or local file
    is_url = input_audio.startswith("http://") or input_audio.startswith("https://")
    
    client = OpenAI(
        api_key=api_key,
        base_url=DEFAULT_BASE_URL,
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": input_audio if is_url else f"data:audio/mp3;base64,{_encode_audio(input_audio)}"
                    }
                }
            ]
        }
    ]
    
    # If using local file and not base64 encoded by user, we need to handle it.
    # However user example puts url in "data": "https://..."
    # If it's local file, OpenAI usually expects base64 encoded string if passing in 'data' field for some custom endpoints,
    # OR we might need to check how qwen3-asr-flash handles local files via OpenAI compatible API.
    # The user example explicitly uses "data": "https://..." URL.
    # If supporting local files, we'd need to upload or base64 encode.
    # Let's assume URL for simplicity as per example, or try base64 for local.
    
    if not is_url:
        # Simple local file check
        if os.path.exists(input_audio):
             with open(input_audio, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
                messages[0]["content"][0]["input_audio"]["data"] = encoded
        else:
            print(f"Error: Input audio file not found: {input_audio}", file=sys.stderr)
            sys.exit(1)

    print(f"Running ASR with {model}...", file=sys.stderr)
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            extra_body={
                "asr_options": {
                    "enable_itn": False
                }
            }
        )
        print(completion.choices[0].message.content)
    except Exception as e:
        print(f"Error running ASR: {e}", file=sys.stderr)
        sys.exit(1)

def _encode_audio(file_path):
    with open(file_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode('utf-8')


# --- TTS ---
def run_tts(api_key: str, model: str, text: str, output_path: str, voice: str = "Cherry"):
    import dashscope
    from dashscope.audio.tts_v2 import SpeechSynthesizer 
    # Note: User example uses dashscope.MultiModalConversation.call which seems unique for qwen3-tts-flash?
    # Actually checking user example:
    # response = dashscope.MultiModalConversation.call(model="qwen3-tts-flash", ...)
    
    dashscope.api_key = api_key
    
    print(f"Running TTS with {model}...", file=sys.stderr)
    try:
        if model == "qwen3-tts-flash":
             # Implementation based on user example
             response = dashscope.MultiModalConversation.call(
                model=model,
                api_key=api_key,
                text=text,
                voice=voice,
                language_type="Chinese" # Defaulting for now
             )
        else:
            # Fallback or other models
            print(f"Model {model} not explicitly supported in this script branch.", file=sys.stderr)
            sys.exit(1)

        if response.status_code == 200:
            if response.output and response.output.audio and response.output.audio.url:
                audio_url = response.output.audio.url
                print(f"Downloading audio from {audio_url}...", file=sys.stderr)
                audio_data = requests.get(audio_url).content
                with open(output_path, "wb") as f:
                    f.write(audio_data)
                print(f"Audio saved to {output_path}")
                print(f"MEDIA: {os.path.abspath(output_path)}")
            else:
                print(f"No audio URL in response: {response}", file=sys.stderr)
                sys.exit(1)
        else:
            print(f"TTS Error: {response.message}", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        print(f"Error running TTS: {e}", file=sys.stderr)
        sys.exit(1)


# --- Video: Shared helper ---
def _download_video(video_url: str, output_path: str):
    """Download video from URL and save to local file."""
    print(f"Downloading video...", file=sys.stderr)
    video_data = requests.get(video_url).content
    with open(output_path, "wb") as f:
        f.write(video_data)
    print(f"Video saved to {output_path}")
    print(f"MEDIA: {os.path.abspath(output_path)}")


# --- Text-to-Video ---
def generate_t2v(api_key: str, model: str, prompt: str, output_path: str,
                 size: str | None = None, duration: int | None = None,
                 prompt_extend: bool = True, shot_type: str = "single",
                 negative_prompt: str | None = None, audio_url: str | None = None,
                 watermark: bool = False, seed: int | None = None):
    from http import HTTPStatus
    from dashscope import VideoSynthesis
    import dashscope
    dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

    print(f"Generating text-to-video with {model}...", file=sys.stderr)
    try:
        kwargs = dict(api_key=api_key, model=model, prompt=prompt,
                      prompt_extend=prompt_extend, watermark=watermark)
        if size:
            kwargs["size"] = size
        if duration is not None:
            kwargs["duration"] = duration
        if shot_type != "single":
            kwargs["shot_type"] = shot_type
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt
        if audio_url:
            kwargs["audio_url"] = audio_url
        if seed is not None:
            kwargs["seed"] = seed

        rsp = VideoSynthesis.call(**kwargs)
        if rsp.status_code == HTTPStatus.OK:
            _download_video(rsp.output.video_url, output_path)
        else:
            print(f"Failed: {rsp.code}, {rsp.message}", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"Error generating t2v: {e}", file=sys.stderr)
        import traceback; traceback.print_exc(file=sys.stderr)
        sys.exit(1)


# --- Image-to-Video ---
def generate_i2v(api_key: str, model: str, img_url: str, output_path: str,
                 prompt: str | None = None, resolution: str | None = None,
                 duration: int | None = None, prompt_extend: bool = True,
                 shot_type: str = "single", negative_prompt: str | None = None,
                 audio_url: str | None = None, watermark: bool = False,
                 seed: int | None = None):
    from http import HTTPStatus
    from dashscope import VideoSynthesis
    import dashscope
    dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

    print(f"Generating image-to-video with {model}...", file=sys.stderr)
    try:
        resolved_img_url = _to_file_url(img_url)

        kwargs = dict(api_key=api_key, model=model, img_url=resolved_img_url,
                      prompt_extend=prompt_extend, watermark=watermark)
        if prompt:
            kwargs["prompt"] = prompt
        if resolution:
            kwargs["resolution"] = resolution
        if duration is not None:
            kwargs["duration"] = duration
        if shot_type != "single":
            kwargs["shot_type"] = shot_type
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt
        if audio_url:
            kwargs["audio_url"] = audio_url
        if seed is not None:
            kwargs["seed"] = seed

        rsp = VideoSynthesis.call(**kwargs)
        if rsp.status_code == HTTPStatus.OK:
            _download_video(rsp.output.video_url, output_path)
        else:
            print(f"Failed: {rsp.code}, {rsp.message}", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"Error generating i2v: {e}", file=sys.stderr)
        import traceback; traceback.print_exc(file=sys.stderr)
        sys.exit(1)


# --- Reference-to-Video ---
def generate_r2v(api_key: str, model: str, prompt: str, reference_urls: list[str],
                 output_path: str, size: str | None = None,
                 duration: int | None = None, shot_type: str = "single",
                 negative_prompt: str | None = None, audio: bool = True,
                 watermark: bool = False, seed: int | None = None):
    from http import HTTPStatus
    from dashscope import VideoSynthesis
    import dashscope
    dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

    print(f"Generating reference-to-video with {model}...", file=sys.stderr)
    try:
        resolved_urls = [_to_file_url(u) for u in reference_urls]

        kwargs = dict(api_key=api_key, model=model, prompt=prompt,
                      reference_urls=resolved_urls, watermark=watermark)
        if size:
            kwargs["size"] = size
        if duration is not None:
            kwargs["duration"] = duration
        if shot_type != "single":
            kwargs["shot_type"] = shot_type
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt
        if not audio:
            kwargs["audio"] = False
        if seed is not None:
            kwargs["seed"] = seed

        rsp = VideoSynthesis.call(**kwargs)
        if rsp.status_code == HTTPStatus.OK:
            _download_video(rsp.output.video_url, output_path)
        else:
            print(f"Failed: {rsp.code}, {rsp.message}", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"Error generating r2v: {e}", file=sys.stderr)
        import traceback; traceback.print_exc(file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run Bailian Multimodal Models")
    parser.add_argument("--mode", required=True, choices=["image", "asr", "tts", "t2v", "i2v", "r2v"], help="Task mode")
    parser.add_argument("--model", required=True, help="Model name (e.g., z-image-turbo, wan2.6-t2v)")
    parser.add_argument("--api-key", help="DashScope API Key")
    
    # Image specific
    parser.add_argument("--prompt", help="Text prompt for image/video generation")
    parser.add_argument("--size", help="Image/video size (e.g., 1024*1024, 1280*720)")
    
    # ASR specific
    parser.add_argument("--input-audio", help="Input audio URL or file path")
    
    # TTS specific
    parser.add_argument("--text", help="Text to synthesize")
    parser.add_argument("--voice", default="Cherry", help="Voice for TTS")
    
    # Video shared
    parser.add_argument("--duration", type=int, help="Video duration in seconds")
    parser.add_argument("--prompt-extend", action=argparse.BooleanOptionalAction, default=True, help="Enable prompt smart rewrite (default: True)")
    parser.add_argument("--shot-type", choices=["single", "multi"], default="single", help="Shot type: single or multi")
    parser.add_argument("--negative-prompt", help="Negative prompt for video generation")
    parser.add_argument("--audio-url", help="Audio URL for video with sound (t2v/i2v)")
    parser.add_argument("--watermark", action="store_true", help="Add AI-generated watermark")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    
    # I2V specific
    parser.add_argument("--img-url", help="Image URL for i2v mode")
    parser.add_argument("--resolution", help="Video resolution for i2v (480P/720P/1080P)")
    
    # R2V specific
    parser.add_argument("--reference-urls", nargs="+", help="Reference URLs (images/videos) for r2v mode")
    parser.add_argument("--no-audio", action="store_true", help="Generate silent video (r2v-flash only)")
    
    # Output
    parser.add_argument("--output", "-o", help="Output file path")

    args = parser.parse_args()
    api_key = get_api_key(args.api_key)
    
    if args.mode == "image":
        if not args.prompt or not args.output:
            print("Error: --prompt and --output are required for image mode.", file=sys.stderr)
            sys.exit(1)
        generate_image(api_key, args.model, args.prompt, args.output, args.size)
        
    elif args.mode == "asr":
        if not args.input_audio:
            print("Error: --input-audio is required for asr mode.", file=sys.stderr)
            sys.exit(1)
        run_asr(api_key, args.model, args.input_audio)
        
    elif args.mode == "tts":
        if not args.text or not args.output:
            print("Error: --text and --output are required for tts mode.", file=sys.stderr)
            sys.exit(1)
        run_tts(api_key, args.model, args.text, args.output, args.voice)

    elif args.mode == "t2v":
        if not args.prompt or not args.output:
            print("Error: --prompt and --output are required for t2v mode.", file=sys.stderr)
            sys.exit(1)
        generate_t2v(api_key, args.model, args.prompt, args.output,
                     size=args.size, duration=args.duration,
                     prompt_extend=args.prompt_extend, shot_type=args.shot_type,
                     negative_prompt=args.negative_prompt, audio_url=args.audio_url,
                     watermark=args.watermark, seed=args.seed)

    elif args.mode == "i2v":
        if not args.img_url or not args.output:
            print("Error: --img-url and --output are required for i2v mode.", file=sys.stderr)
            sys.exit(1)
        generate_i2v(api_key, args.model, args.img_url, args.output,
                     prompt=args.prompt, resolution=args.resolution,
                     duration=args.duration, prompt_extend=args.prompt_extend,
                     shot_type=args.shot_type, negative_prompt=args.negative_prompt,
                     audio_url=args.audio_url, watermark=args.watermark, seed=args.seed)

    elif args.mode == "r2v":
        if not args.prompt or not args.reference_urls or not args.output:
            print("Error: --prompt, --reference-urls and --output are required for r2v mode.", file=sys.stderr)
            sys.exit(1)
        generate_r2v(api_key, args.model, args.prompt, args.reference_urls, args.output,
                     size=args.size, duration=args.duration, shot_type=args.shot_type,
                     negative_prompt=args.negative_prompt, audio=not args.no_audio,
                     watermark=args.watermark, seed=args.seed)

if __name__ == "__main__":
    main()

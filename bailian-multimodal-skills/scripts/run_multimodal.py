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
import time
import requests

# Fix encoding issues
import locale
locale.setlocale(locale.LC_ALL, 'C.UTF-8')
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# --- Constants ---
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
VIDEO_SYNTHESIS_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"
TASK_STATUS_URL = "https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
PIXVERSE_T2V_MODELS = {"pixverse/pixverse-v5.6-t2v"}
PIXVERSE_I2V_MODELS = {"pixverse/pixverse-v5.6-it2v"}
PIXVERSE_R2V_MODELS = {"pixverse/pixverse-v5.6-r2v"}
KLING_VIDEO_MODELS = {"kling/kling-v3-video-generation"}
WAN27_T2V_MODELS = {"wan2.7-t2v"}
WAN27_VIDEOEDIT_MODELS = {"wan2.7-videoedit"}


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
    # URL-encode spaces and special characters in the path
    from urllib.parse import quote
    return "file://" + quote(abs_path, safe="/:@")


def _video_headers(api_key: str) -> dict[str, str]:
    return {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {api_key}",
        "X-DashScope-Async": "enable",
    }


def _video_headers_with_oss(api_key: str) -> dict[str, str]:
    """Headers for requests that use oss:// URLs (e.g. videoedit)."""
    h = _video_headers(api_key)
    h["X-DashScope-OssResourceResolve"] = "enable"
    return h


def _upload_local_file_to_oss(api_key: str, model_name: str, file_path: str) -> str:
    """Upload a local file to Aliyun temp storage and return oss:// URL."""
    from pathlib import Path
    abs_path = os.path.abspath(file_path)
    if not os.path.exists(abs_path):
        print(f"Error: File not found: {abs_path}", file=sys.stderr)
        sys.exit(1)

    # Step 1: get upload policy
    policy_resp = requests.get(
        "https://dashscope.aliyuncs.com/api/v1/uploads",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        params={"action": "getPolicy", "model": model_name},
        timeout=30,
    )
    if not policy_resp.ok:
        raise RuntimeError(f"Failed to get upload policy: {policy_resp.text}")
    data = policy_resp.json()["data"]

    # Step 2: upload to OSS
    file_name = Path(abs_path).name
    key = f"{data['upload_dir']}/{file_name}"
    with open(abs_path, "rb") as f:
        files = {
            "OSSAccessKeyId": (None, data["oss_access_key_id"]),
            "Signature": (None, data["signature"]),
            "policy": (None, data["policy"]),
            "x-oss-object-acl": (None, data["x_oss_object_acl"]),
            "x-oss-forbid-overwrite": (None, data["x_oss_forbid_overwrite"]),
            "key": (None, key),
            "success_action_status": (None, "200"),
            "file": (file_name, f),
        }
        upload_resp = requests.post(data["upload_host"], files=files, timeout=300)
    if upload_resp.status_code != 200:
        raise RuntimeError(f"Failed to upload file to OSS ({upload_resp.status_code}): {upload_resp.text[:200]}")

    oss_url = f"oss://{key}"
    print(f"Uploaded {file_name} -> {oss_url}", file=sys.stderr)
    return oss_url


def _resolve_media_url(api_key: str, model_name: str, path_or_url: str) -> str:
    """Return a URL suitable for server-side download.
    - HTTP/HTTPS/OSS URLs are returned as-is.
    - Local file paths are uploaded to Aliyun temp storage and the oss:// URL is returned.
    """
    if path_or_url.startswith(("http://", "https://", "oss://", "data:")):
        return path_or_url
    # Local path — upload to get oss:// URL
    return _upload_local_file_to_oss(api_key, model_name, path_or_url)


def _post_json(url: str, headers: dict[str, str], payload: dict) -> dict:
    payload_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    response = requests.post(url, headers=headers, data=payload_bytes)
    if not response.ok:
        print(f"HTTP {response.status_code}: {response.text[:500]}", file=sys.stderr)
    response.raise_for_status()
    result = response.json()
    if result.get("code"):
        raise RuntimeError(f"{result['code']}: {result.get('message', 'Unknown error')}")
    return result


def _poll_task(api_key: str, task_id: str, poll_interval: int = 15, timeout_seconds: int = 1800) -> dict:
    headers = {"Authorization": f"Bearer {api_key}"}
    started = time.time()
    consecutive_errors = 0
    while True:
        try:
            response = requests.get(TASK_STATUS_URL.format(task_id=task_id), headers=headers, timeout=60)
            response.raise_for_status()
            result = response.json()
            consecutive_errors = 0
        except requests.RequestException as exc:
            consecutive_errors += 1
            if consecutive_errors >= 3:
                raise RuntimeError(f"Failed to poll task {task_id} after {consecutive_errors} consecutive network errors: {exc}") from exc
            print(
                f"Polling task {task_id} hit transient network error ({consecutive_errors}/3): {exc}. Retrying in 5s...",
                file=sys.stderr,
            )
            time.sleep(5)
            continue

        if result.get("code"):
            raise RuntimeError(f"{result['code']}: {result.get('message', 'Unknown error')}")

        output = result.get("output", {})
        task_status = output.get("task_status")
        if task_status == "SUCCEEDED":
            return result
        if task_status in {"FAILED", "CANCELED", "UNKNOWN"}:
            raise RuntimeError(f"Task {task_id} ended with status {task_status}: {result.get('message', output.get('message', ''))}")
        if time.time() - started > timeout_seconds:
            raise TimeoutError(f"Timed out waiting for task {task_id} after {timeout_seconds} seconds")

        print(f"Task {task_id} status: {task_status or 'UNKNOWN'}, waiting {poll_interval}s...", file=sys.stderr)
        time.sleep(poll_interval)


def _submit_async_video_task(api_key: str, payload: dict) -> dict:
    result = _post_json(VIDEO_SYNTHESIS_URL, _video_headers(api_key), payload)
    output = result.get("output", {})
    task_id = output.get("task_id")
    if not task_id:
        raise RuntimeError(f"Missing task_id in response: {json.dumps(result, ensure_ascii=False)}")
    print(f"Created task {task_id}, polling for result...", file=sys.stderr)
    return _poll_task(api_key, task_id)


def _extract_video_url(task_result: dict) -> str:
    output = task_result.get("output", {})
    video_url = output.get("video_url") or output.get("results", [{}])[0].get("video_url")
    if not video_url:
        raise RuntimeError(f"No video_url found in task result: {json.dumps(task_result, ensure_ascii=False)}")
    return video_url


def _size_to_aspect_ratio(size: str | None) -> str | None:
    if not size:
        return None
    mapping = {
        "1280*720": "16:9",
        "1920*1080": "16:9",
        "1024*576": "16:9",
        "640*360": "16:9",
        "1280*960": "4:3",
        "1920*1440": "4:3",
        "1024*768": "4:3",
        "640*480": "4:3",
        "1280*1280": "1:1",
        "1808*1808": "1:1",
        "1024*1024": "1:1",
        "640*640": "1:1",
        "960*1280": "3:4",
        "1440*1920": "3:4",
        "768*1024": "3:4",
        "480*640": "3:4",
        "720*1280": "9:16",
        "1080*1920": "9:16",
        "576*1024": "9:16",
        "360*640": "9:16",
    }
    if size in mapping:
        return mapping[size]
    try:
        width, height = size.split("*", 1)
        return f"{int(width)}:{int(height)}"
    except (ValueError, TypeError):
        return None


# --- Image Generation ---
def _save_images_from_choices(result: dict, output_path: str):
    """Extract and save image(s) from API response choices."""
    if "output" not in result or "choices" not in result["output"]:
        print(f"Unexpected response format: {json.dumps(result, indent=2)}", file=sys.stderr)
        sys.exit(1)

    choices = result["output"]["choices"]
    if not choices:
        print(f"No choices in response: {json.dumps(result, indent=2)}", file=sys.stderr)
        sys.exit(1)

    content = choices[0]["message"]["content"]
    image_urls = [item["image"] for item in content if "image" in item]

    if not image_urls:
        print(f"No images found in response: {json.dumps(result, indent=2)}", file=sys.stderr)
        sys.exit(1)

    if len(image_urls) == 1:
        print(f"Downloading image...", file=sys.stderr)
        img_data = requests.get(image_urls[0]).content
        with open(output_path, "wb") as f:
            f.write(img_data)
        print(f"Image saved to {output_path}")
        print(f"MEDIA: {os.path.abspath(output_path)}")
    else:
        base, ext = os.path.splitext(output_path)
        for i, url in enumerate(image_urls, 1):
            path = f"{base}_{i}{ext}"
            print(f"Downloading image {i}/{len(image_urls)}...", file=sys.stderr)
            img_data = requests.get(url).content
            with open(path, "wb") as f:
                f.write(img_data)
            print(f"Image {i} saved to {path}")
            print(f"MEDIA: {os.path.abspath(path)}")


def generate_image(api_key: str, model: str, prompt: str, output_path: str, size: str = "1024*1024",
                   n: int = 1, enable_sequential: bool = False, watermark: bool = False):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
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
             "size": size if size else "1024*1024"
         }
    elif model == "wan2.6-t2i":
        payload["parameters"] = {
            "prompt_extend": True,
            "watermark": False,
            "n": 1,
            "size": size if size else "1280*1280"
        }
    elif model == "wan2.7-image-pro":
        payload["parameters"] = {
            "n": n,
            "size": size if size else "2K",
            "watermark": watermark,
        }
        if enable_sequential:
            payload["parameters"]["enable_sequential"] = True

    print(f"Generating image with {model}...", file=sys.stderr)
    try:
        payload_bytes = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        headers['Content-Type'] = 'application/json; charset=utf-8'
        response = requests.post(DASHSCOPE_API_URL, headers=headers, data=payload_bytes)
        response.raise_for_status()
        result = response.json()
        
        if "code" in result and result["code"]:
             print(f"API Error: {result['message']}", file=sys.stderr)
             sys.exit(1)

        _save_images_from_choices(result, output_path)

    except Exception as e:
        print(f"Error generating image: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


# --- Image Editing ---
def generate_image_edit(api_key: str, model: str, prompt: str, input_images: list[str],
                       output_path: str, size: str = "2K", n: int = 1,
                       watermark: bool = False, thinking_mode: bool = False):
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {api_key}",
    }

    content: list[dict] = []
    import base64
    import mimetypes

    for img in input_images:
        # Convert local image to base64 format for API
        mime_type, _ = mimetypes.guess_type(img)
        if not mime_type or not mime_type.startswith('image/'):
            mime_type = 'image/jpeg'  # default to jpeg

        with open(img, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        base64_url = f"data:{mime_type};base64,{encoded_string}"

        content.append({"image": base64_url})
    content.append({"text": prompt})

    payload = {
        "model": model,
        "input": {
            "messages": [{"role": "user", "content": content}]
        },
        "parameters": {
            "size": size if size else "2K",
            "n": n,
            "watermark": watermark,
        },
    }
    if thinking_mode:
        payload["parameters"]["thinking_mode"] = True

    print(f"Editing image with {model}...", file=sys.stderr)
    try:
        payload_bytes = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        response = requests.post(DASHSCOPE_API_URL, headers=headers, data=payload_bytes)
        response.raise_for_status()
        result = response.json()

        if "code" in result and result["code"]:
            print(f"API Error: {result['message']}", file=sys.stderr)
            sys.exit(1)

        _save_images_from_choices(result, output_path)

    except Exception as e:
        print(f"Error editing image: {e}", file=sys.stderr)
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


def _generate_pixverse_t2v(api_key: str, model: str, prompt: str, output_path: str,
                           size: str | None = None, duration: int | None = None,
                           audio: bool = False, watermark: bool = False,
                           seed: int | None = None):
    payload = {
        "model": model,
        "input": {"prompt": prompt},
        "parameters": {
            "size": size or "1280*720",
            "duration": duration or 5,
            "audio": audio,
            "watermark": watermark,
        },
    }
    if seed is not None:
        payload["parameters"]["seed"] = seed

    result = _submit_async_video_task(api_key, payload)
    _download_video(_extract_video_url(result), output_path)


def _generate_pixverse_i2v(api_key: str, model: str, img_url: str, output_path: str,
                           prompt: str | None = None, resolution: str | None = None,
                           duration: int | None = None, audio: bool = False,
                           watermark: bool = False, seed: int | None = None):
    resolution_map = {
        "360P": "640*360",
        "540P": "1024*576",
        "720P": "1280*720",
        "1080P": "1920*1080",
    }
    payload = {
        "model": model,
        "input": {
            "prompt": prompt or "",
            "media": [{"type": "first_frame", "url": _to_file_url(img_url)}],
        },
        "parameters": {
            "resolution": resolution or "720P",
            "size": resolution_map.get((resolution or "720P").upper(), "1280*720"),
            "duration": duration or 5,
            "audio": audio,
            "watermark": watermark,
        },
    }
    if seed is not None:
        payload["parameters"]["seed"] = seed

    result = _submit_async_video_task(api_key, payload)
    _download_video(_extract_video_url(result), output_path)


def _generate_pixverse_r2v(api_key: str, model: str, prompt: str, reference_urls: list[str],
                           output_path: str, size: str | None = None,
                           duration: int | None = None, audio: bool = False,
                           watermark: bool = False, seed: int | None = None):
    payload = {
        "model": model,
        "input": {
            "prompt": prompt,
            "media": [{"type": "refer", "url": _to_file_url(url)} for url in reference_urls],
        },
        "parameters": {
            "size": size or "1280*720",
            "duration": duration or 5,
            "audio": audio,
            "watermark": watermark,
        },
    }
    if seed is not None:
        payload["parameters"]["seed"] = seed

    result = _submit_async_video_task(api_key, payload)
    _download_video(_extract_video_url(result), output_path)


def _generate_kling_t2v(api_key: str, model: str, prompt: str, output_path: str,
                        size: str | None = None, duration: int | None = None,
                        audio: bool = False, watermark: bool = False,
                        quality_mode: str = "std"):
    payload = {
        "model": model,
        "input": {"prompt": prompt},
        "parameters": {
            "mode": quality_mode,
            "aspect_ratio": _size_to_aspect_ratio(size) or "16:9",
            "duration": duration or 5,
            "audio": audio,
            "watermark": watermark,
        },
    }

    result = _submit_async_video_task(api_key, payload)
    _download_video(_extract_video_url(result), output_path)


def _generate_kling_i2v(api_key: str, model: str, img_url: str, output_path: str,
                        prompt: str | None = None, duration: int | None = None,
                        audio: bool = False, watermark: bool = False,
                        quality_mode: str = "std"):
    payload = {
        "model": model,
        "input": {
            "prompt": prompt or "",
            "media": [{"type": "first_frame", "url": _to_file_url(img_url)}],
        },
        "parameters": {
            "mode": quality_mode,
            "duration": duration or 5,
            "audio": audio,
            "watermark": watermark,
        },
    }

    result = _submit_async_video_task(api_key, payload)
    _download_video(_extract_video_url(result), output_path)


# --- wan2.7 Video Edit ---
def _generate_wan27_videoedit(api_key: str, model: str, video_url: str, output_path: str,
                              prompt: str | None = None,
                              negative_prompt: str | None = None,
                              ref_images: list[str] | None = None,
                              resolution: str | None = None,
                              ratio: str | None = None,
                              prompt_extend: bool = True,
                              watermark: bool = False,
                              seed: int | None = None):
    """Edit a video using wan2.7-videoedit (new HTTP protocol, SDK not supported)."""
    # Resolve input video: upload local files to OSS
    resolved_video = _resolve_media_url(api_key, model, video_url)

    media: list[dict] = [{"type": "video", "url": resolved_video}]
    if ref_images:
        for img in ref_images:
            resolved_img = _resolve_media_url(api_key, model, img)
            media.append({"type": "reference_image", "url": resolved_img})

    inp: dict = {"media": media}
    if prompt:
        inp["prompt"] = prompt
    if negative_prompt:
        inp["negative_prompt"] = negative_prompt

    params: dict = {
        "resolution": resolution or "1080P",
        "prompt_extend": prompt_extend,
        "watermark": watermark,
    }
    if ratio:
        params["ratio"] = ratio
    if seed is not None:
        params["seed"] = seed

    payload = {"model": model, "input": inp, "parameters": params}
    # videoedit requires X-DashScope-OssResourceResolve: enable when using oss:// URLs
    result = _post_json(VIDEO_SYNTHESIS_URL, _video_headers_with_oss(api_key), payload)
    output = result.get("output", {})
    task_id = output.get("task_id")
    if not task_id:
        raise RuntimeError(f"Missing task_id in response: {json.dumps(result, ensure_ascii=False)}")
    print(f"Created task {task_id}, polling for result...", file=sys.stderr)
    final_result = _poll_task(api_key, task_id)
    _download_video(_extract_video_url(final_result), output_path)


# --- wan2.7 T2V (new HTTP protocol) ---
def _generate_wan27_t2v(api_key: str, model: str, prompt: str, output_path: str,
                        resolution: str | None = None, ratio: str | None = None,
                        size: str | None = None, duration: int | None = None,
                        prompt_extend: bool = True,
                        negative_prompt: str | None = None,
                        audio_url: str | None = None, audio: bool = False,
                        watermark: bool = False, seed: int | None = None):
    """Submit wan2.7-t2v using the new HTTP API protocol (SDK not supported)."""
    # Derive ratio from size if ratio not explicitly provided
    if not ratio and size:
        ratio = _size_to_aspect_ratio(size)

    inp: dict = {"prompt": prompt}
    if negative_prompt:
        inp["negative_prompt"] = negative_prompt

    params: dict = {
        "resolution": resolution or "1080P",
        "ratio": ratio or "16:9",
        "prompt_extend": prompt_extend,
        "watermark": watermark,
    }
    if duration is not None:
        params["duration"] = duration
    if audio_url:
        params["audio_url"] = audio_url
    if audio:
        params["audio"] = True
    if seed is not None:
        params["seed"] = seed

    payload = {"model": model, "input": inp, "parameters": params}
    result = _submit_async_video_task(api_key, payload)
    _download_video(_extract_video_url(result), output_path)


# --- Text-to-Video ---
def generate_t2v(api_key: str, model: str, prompt: str, output_path: str,
                 size: str | None = None, duration: int | None = None,
                 prompt_extend: bool = True, shot_type: str = "single",
                 negative_prompt: str | None = None, audio_url: str | None = None,
                 watermark: bool = False, seed: int | None = None,
                 audio: bool = False, quality_mode: str = "std",
                 resolution: str | None = None, ratio: str | None = None):
    if model in WAN27_T2V_MODELS:
        print(f"Generating text-to-video with {model} (wan2.7 HTTP API)...", file=sys.stderr)
        try:
            _generate_wan27_t2v(
                api_key, model, prompt, output_path,
                resolution=resolution, ratio=ratio, size=size,
                duration=duration, prompt_extend=prompt_extend,
                negative_prompt=negative_prompt, audio_url=audio_url,
                audio=audio, watermark=watermark, seed=seed,
            )
            return
        except Exception as e:
            print(f"Error generating wan2.7 t2v: {e}", file=sys.stderr)
            import traceback; traceback.print_exc(file=sys.stderr)
            sys.exit(1)

    if model in PIXVERSE_T2V_MODELS:
        print(f"Generating text-to-video with {model} (PixVerse API)...", file=sys.stderr)
        try:
            _generate_pixverse_t2v(
                api_key,
                model,
                prompt,
                output_path,
                size=size,
                duration=duration,
                audio=audio,
                watermark=watermark,
                seed=seed,
            )
            return
        except Exception as e:
            print(f"Error generating pixverse t2v: {e}", file=sys.stderr)
            import traceback; traceback.print_exc(file=sys.stderr)
            sys.exit(1)

    if model in KLING_VIDEO_MODELS:
        print(f"Generating text-to-video with {model} (Kling API)...", file=sys.stderr)
        try:
            _generate_kling_t2v(
                api_key,
                model,
                prompt,
                output_path,
                size=size,
                duration=duration,
                audio=audio,
                watermark=watermark,
                quality_mode=quality_mode,
            )
            return
        except Exception as e:
            print(f"Error generating kling t2v: {e}", file=sys.stderr)
            import traceback; traceback.print_exc(file=sys.stderr)
            sys.exit(1)

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
                 seed: int | None = None, audio: bool = False,
                 quality_mode: str = "std"):
    if model in PIXVERSE_I2V_MODELS:
        print(f"Generating image-to-video with {model} (PixVerse API)...", file=sys.stderr)
        try:
            _generate_pixverse_i2v(
                api_key,
                model,
                img_url,
                output_path,
                prompt=prompt,
                resolution=resolution,
                duration=duration,
                audio=audio,
                watermark=watermark,
                seed=seed,
            )
            return
        except Exception as e:
            print(f"Error generating pixverse i2v: {e}", file=sys.stderr)
            import traceback; traceback.print_exc(file=sys.stderr)
            sys.exit(1)

    if model in KLING_VIDEO_MODELS:
        print(f"Generating image-to-video with {model} (Kling API)...", file=sys.stderr)
        try:
            _generate_kling_i2v(
                api_key,
                model,
                img_url,
                output_path,
                prompt=prompt,
                duration=duration,
                audio=audio,
                watermark=watermark,
                quality_mode=quality_mode,
            )
            return
        except Exception as e:
            print(f"Error generating kling i2v: {e}", file=sys.stderr)
            import traceback; traceback.print_exc(file=sys.stderr)
            sys.exit(1)

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
    if model in PIXVERSE_R2V_MODELS:
        print(f"Generating reference-to-video with {model} (PixVerse API)...", file=sys.stderr)
        try:
            _generate_pixverse_r2v(
                api_key,
                model,
                prompt,
                reference_urls,
                output_path,
                size=size,
                duration=duration,
                audio=audio,
                watermark=watermark,
                seed=seed,
            )
            return
        except Exception as e:
            print(f"Error generating pixverse r2v: {e}", file=sys.stderr)
            import traceback; traceback.print_exc(file=sys.stderr)
            sys.exit(1)

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
    parser.add_argument("--mode", required=True, choices=["image", "image-edit", "asr", "tts", "t2v", "i2v", "r2v", "videoedit"], help="Task mode")
    parser.add_argument("--model", required=True, help="Model name (e.g., z-image-turbo, wan2.6-t2v)")
    parser.add_argument("--api-key", help="DashScope API Key")
    
    # Image specific
    parser.add_argument("--prompt", help="Text prompt for image/video generation")
    parser.add_argument("--size", help="Image/video size (e.g., 1024*1024, 1280*720, 2K)")
    parser.add_argument("--n", type=int, default=1, help="Number of images to generate")
    parser.add_argument("--enable-sequential", action="store_true", help="Enable sequential consistency for multi-image generation")
    parser.add_argument("--thinking-mode", action="store_true", help="Enable thinking mode for image editing")
    parser.add_argument("--input-images", nargs="+", help="Input image URLs or local paths for image editing")
    
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
    parser.add_argument("--audio", dest="audio", action="store_true", help="Enable generated audio when the selected video model supports it")
    parser.add_argument("--no-audio", dest="audio", action="store_false", help="Disable audio when the selected video model supports it")
    parser.add_argument("--watermark", action="store_true", help="Add AI-generated watermark")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--quality-mode", choices=["std", "pro"], default="std", help="Video quality mode for models that support it (for example Kling)")
    parser.set_defaults(audio=None)
    
    # I2V / wan2.7-t2v / videoedit specific
    parser.add_argument("--img-url", help="Image URL for i2v mode")
    parser.add_argument("--resolution", help="Video resolution: 480P/720P/1080P (i2v) or 720P/1080P (wan2.7 models, default 1080P)")
    parser.add_argument("--ratio", choices=["16:9", "9:16", "1:1"], help="Video aspect ratio for wan2.7 models (default 16:9 for t2v; follows input video for videoedit)")

    # videoedit specific
    parser.add_argument("--video-url", help="Input video URL or local path for videoedit mode")
    parser.add_argument("--ref-images", nargs="+", help="Reference image URLs or local paths for videoedit mode (optional)")
    
    # R2V specific
    parser.add_argument("--reference-urls", nargs="+", help="Reference URLs (images/videos) for r2v mode")
    
    # Output
    parser.add_argument("--output", "-o", help="Output file path")

    args = parser.parse_args()
    api_key = get_api_key(args.api_key)
    video_audio = False if args.audio is None else args.audio
    r2v_audio = True if args.audio is None else args.audio
    
    if args.mode == "image":
        if not args.prompt or not args.output:
            print("Error: --prompt and --output are required for image mode.", file=sys.stderr)
            sys.exit(1)
        generate_image(api_key, args.model, args.prompt, args.output, args.size,
                       n=args.n, enable_sequential=args.enable_sequential,
                       watermark=args.watermark)

    elif args.mode == "image-edit":
        if not args.prompt or not args.input_images or not args.output:
            print("Error: --prompt, --input-images and --output are required for image-edit mode.", file=sys.stderr)
            sys.exit(1)
        generate_image_edit(api_key, args.model, args.prompt, args.input_images, args.output,
                            size=args.size or "2K", n=args.n, watermark=args.watermark,
                            thinking_mode=args.thinking_mode)
        
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
                     watermark=args.watermark, seed=args.seed,
                     audio=video_audio, quality_mode=args.quality_mode,
                     resolution=args.resolution, ratio=args.ratio)

    elif args.mode == "i2v":
        if not args.img_url or not args.output:
            print("Error: --img-url and --output are required for i2v mode.", file=sys.stderr)
            sys.exit(1)
        generate_i2v(api_key, args.model, args.img_url, args.output,
                     prompt=args.prompt, resolution=args.resolution,
                     duration=args.duration, prompt_extend=args.prompt_extend,
                     shot_type=args.shot_type, negative_prompt=args.negative_prompt,
                     audio_url=args.audio_url, watermark=args.watermark, seed=args.seed,
                     audio=video_audio, quality_mode=args.quality_mode)

    elif args.mode == "r2v":
        if not args.prompt or not args.reference_urls or not args.output:
            print("Error: --prompt, --reference-urls and --output are required for r2v mode.", file=sys.stderr)
            sys.exit(1)
        generate_r2v(api_key, args.model, args.prompt, args.reference_urls, args.output,
                     size=args.size, duration=args.duration, shot_type=args.shot_type,
                     negative_prompt=args.negative_prompt, audio=r2v_audio,
                     watermark=args.watermark, seed=args.seed)

    elif args.mode == "videoedit":
        if not args.video_url or not args.output:
            print("Error: --video-url and --output are required for videoedit mode.", file=sys.stderr)
            sys.exit(1)
        if args.model not in WAN27_VIDEOEDIT_MODELS:
            print(f"Error: videoedit mode only supports models: {WAN27_VIDEOEDIT_MODELS}", file=sys.stderr)
            sys.exit(1)
        print(f"Editing video with {args.model}...", file=sys.stderr)
        try:
            _generate_wan27_videoedit(
                api_key, args.model, args.video_url, args.output,
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                ref_images=args.ref_images,
                resolution=args.resolution,
                ratio=args.ratio,
                prompt_extend=args.prompt_extend,
                watermark=args.watermark,
                seed=args.seed,
            )
        except Exception as e:
            print(f"Error editing video: {e}", file=sys.stderr)
            import traceback; traceback.print_exc(file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()

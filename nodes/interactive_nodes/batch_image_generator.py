import sys
import os
import io
import time
import hashlib
import json
import threading
import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum

import requests
import numpy as np
import torch
from PIL import Image

from nodes import PreviewImage
from comfy.model_management import InterruptProcessingException
from server import PromptServer
from aiohttp import web

from .image_filter_messaging import send_and_wait, Response, TimeoutResponse

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
if _ROOT not in sys.path: 
    sys.path.insert(0, _ROOT)

_BASE_API_LOADED = False

try:
    import importlib.util as _ilu
    _ga_path = os.path.join(_ROOT, 'nodes', 'api_nodes', 'generative_api_nodes.py')
    _ga_spec = _ilu.spec_from_file_location('omni_gen_api_big', _ga_path)
    _ga_mod  = _ilu.module_from_spec(_ga_spec)
    _ga_spec.loader.exec_module(_ga_mod)
    MODEL_CONFIG = _ga_mod.MODEL_CONFIG
    OMNI_GenerativeAPIBase = _ga_mod.OmniSimulator_GenerativeAPIBase
    _BASE_API_LOADED = True
except Exception as _e:
    print(f"[OmniSimulator] generative_api_nodes import warning: {_e}")
    MODEL_CONFIG = {}
    OMNI_GenerativeAPIBase = object

_ULID_CHARS = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

def generate_ulid() -> str:
    import random
    timestamp_ms = int(time.time() * 1000)
    timestamp_part = ""
    for _ in range(10):
        timestamp_part = _ULID_CHARS[timestamp_ms & 31] + timestamp_part
        timestamp_ms >>= 5

    random_part = "".join(random.choice(_ULID_CHARS) for _ in range(16))
    return timestamp_part + random_part


class JobState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    SUCCESS = "success"
    FAILED = "failed"
    CACHED = "cached"
    RETRYING = "retrying"


class ProgressTracker:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._progress_data = {}
        return cls._instance

    def start_generation(self, node_id: int, jobs: List['GenerationJob']):
        self._progress_data[node_id] = {
            "jobs": [self._job_to_dict(j) for j in jobs],
            "stats": {
                "total": len(jobs),
                "completed": 0,
                "success": 0,
                "failed": 0,
                "cached": 0,
            },
            "is_generating": True,
            "should_stop": False,
            "start_time": time.time(),
        }
        self._emit_event("omni-batch-gen-start", {
            "node_id": node_id,
            "jobs": self._progress_data[node_id]["jobs"],
        })

    def update_job(self, node_id: int, job: 'GenerationJob'):
        if node_id not in self._progress_data:
            return

        data = self._progress_data[node_id]
        for i, j in enumerate(data["jobs"]):
            if j["id"] == job.id:
                data["jobs"][i] = self._job_to_dict(job)
                break

        jobs = data["jobs"]
        data["stats"]["completed"] = sum(1 for j in jobs if j["state"] in ["success", "failed", "cached"])
        data["stats"]["success"] = sum(1 for j in jobs if j["state"] == "success")
        data["stats"]["failed"] = sum(1 for j in jobs if j["state"] == "failed")
        data["stats"]["cached"] = sum(1 for j in jobs if j["state"] == "cached")

        self._emit_event("omni-batch-gen-update", {
            "node_id": node_id,
            "job_id": job.id,
            "state": job.state.value,
            "attempts": job.attempts,
            "error": job.error,
            "generation_time": job.generation_time,
            "image_url": job.result_image_url,
            "image_width": job.result_image_width,
            "image_height": job.result_image_height,
        })

    def complete_generation(self, node_id: int):
        if node_id in self._progress_data:
            self._progress_data[node_id]["is_generating"] = False
            self._emit_event("omni-batch-gen-complete", {
                "node_id": node_id,
                "stats": self._progress_data[node_id]["stats"],
            })

    def get_progress(self, node_id: int) -> Optional[Dict]:
        return self._progress_data.get(node_id)

    def clear_progress(self, node_id: int):
        if node_id in self._progress_data:
            del self._progress_data[node_id]

    def stop_generation(self, node_id: int):
        print(f"[ProgressTracker] stop_generation called for node {node_id}", flush=True)
        if node_id in self._progress_data:
            self._progress_data[node_id]["should_stop"] = True
            self._progress_data[node_id]["is_generating"] = False
            print(f"[ProgressTracker] Set should_stop=True for node {node_id}", flush=True)
        else:
            print(f"[ProgressTracker] WARNING: Node {node_id} not in progress_data!", flush=True)

    def should_stop(self, node_id: int) -> bool:
        if node_id in self._progress_data:
            return self._progress_data[node_id].get("should_stop", False)
        return False

    def _job_to_dict(self, job: 'GenerationJob') -> Dict:
        return {
            "id": job.id,
            "prompt_positive": job.prompt_positive[:100] + "..." if len(job.prompt_positive) > 100 else job.prompt_positive,
            "state": job.state.value,
            "attempts": job.attempts,
            "error": job.error[:100] if job.error else None,
            "generation_time": job.generation_time,
            "cache_hit": job.cache_hit,
            "image_url": job.result_image_url,
            "image_width": job.result_image_width,
            "image_height": job.result_image_height,
            "filename": job.result_filename,
        }

    def _emit_event(self, event_type: str, data: Dict):
        try:
            PromptServer.instance.send_sync(event_type, data)
        except Exception as e:
            print(f"[OmniSimulator] Failed to emit event {event_type}: {e}", flush=True)


progress_tracker = ProgressTracker()


@PromptServer.instance.routes.get('/omni/batch_gen_progress/{node_id}')
async def get_batch_gen_progress(request):
    try:
        node_id = int(request.match_info['node_id'])
        progress = progress_tracker.get_progress(node_id)
        if progress:
            return web.json_response(progress)
        return web.json_response({"jobs": [], "stats": {}, "is_generating": False})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


@PromptServer.instance.routes.post('/omni/batch_gen_retry_failed')
async def retry_failed_jobs(request):
    try:
        data = await request.json()
        return web.json_response({"status": "acknowledged", "message": "Retry not yet implemented"})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


@PromptServer.instance.routes.post('/omni/batch_gen_retry_job')
async def retry_single_job(request):
    try:
        data = await request.json()
        return web.json_response({"status": "acknowledged", "message": "Retry not yet implemented"})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


@PromptServer.instance.routes.post('/omni/batch_gen_stop')
async def stop_generation(request):
    try:
        data = await request.json()
        node_id = data.get("node_id")
        tracker = ProgressTracker()
        tracker.stop_generation(node_id)
        return web.json_response({"status": "stopped", "node_id": node_id})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


@PromptServer.instance.routes.get('/omni/batch_gen_image/{subdir}/{filename}')
async def serve_batch_gen_image(request):
    try:
        import folder_paths
        subdir = request.match_info.get('subdir', 'batch_gen')
        filename = request.match_info.get('filename', '')

        if '..' in subdir or '..' in filename or '/' in filename or '\\' in filename:
            return web.Response(status=403, text="Invalid path")

        comfy_output = folder_paths.get_output_directory()
        filepath = os.path.join(comfy_output, subdir, filename)

        if not os.path.exists(filepath):
            print(f"[OmniSimulator] Image not found: {filepath}", flush=True)
            return web.Response(status=404, text="Image not found")

        ext = os.path.splitext(filename)[1].lower()
        content_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
        }
        content_type = content_types.get(ext, 'application/octet-stream')

        with open(filepath, 'rb') as f:
            data = f.read()

        return web.Response(body=data, content_type=content_type)
    except Exception as e:
        print(f"[OmniSimulator] Error serving image: {e}", flush=True)
        return web.Response(status=500, text=str(e))


def _generate_multi_image(
    engine,
    api_key: str,
    provider: str,
    model: str,
    prompt_positive: str,
    prompt_negative: str,
    aspect_ratio: str,
    resolution: str,
    width: int,
    height: int,
    use_negative_prompt: bool,
    input_image,
    input_image2,
    input_image3,
    input_image4,
    filename_prefix: str,
    job_ids: list,
    use_cache: bool = True,
    cache_keys: list = None,
    node_id: str = None,
) -> dict:
    from ..api_nodes.generative_api_nodes import MODEL_CONFIG
    import base64 as b64

    start_time = time.time()
    MULTI_SUPPORTED_RATIOS = ["3:2", "2:3", "3:4", "4:3"]
    RATIO_MAPPING = {
        "1:1": "3:4",
        "16:9": "3:2",
        "9:16": "2:3",
        "4:5": "3:4",
        "5:4": "4:3",
        "21:9": "3:2",
    }

    original_aspect = aspect_ratio
    if aspect_ratio not in MULTI_SUPPORTED_RATIOS:
        aspect_ratio = RATIO_MAPPING.get(aspect_ratio, "3:2")
        print(f"[OmniSimulator] Multi-image: Aspect ratio '{original_aspect}' not supported, using '{aspect_ratio}' instead", flush=True)

    if use_negative_prompt and prompt_negative.strip():
        if model in ["Nano Banana Pro", "Nano Banana"]:
            full_prompt = f"{prompt_positive}\n\nAvoid: {prompt_negative}"
        else:
            full_prompt = prompt_positive
    else:
        full_prompt = prompt_positive

    model_conf = MODEL_CONFIG.get(model)
    if not model_conf:
        raise Exception(f"Invalid model: {model}")

    provider_conf = model_conf["providers"].get(provider)
    if not provider_conf:
        raise Exception(f"Provider '{provider}' not supported for {model}")

    is_i2i = any(img is not None for img in [input_image, input_image2, input_image3, input_image4])
    base_endpoint = provider_conf["i2i_endpoint"] if is_i2i else provider_conf["t2i_endpoint"]
    endpoint = f"{base_endpoint}-multi"
    build_payload_func = provider_conf["build_payload"]

    kwargs = {
        "prompt": full_prompt,
        "aspect_ratio": aspect_ratio,
        "width": width,
        "height": height,
        "resolution": resolution,
        "num_images": 2,
    }
    if input_image is not None:
        kwargs["image_1"] = input_image
    if input_image2 is not None:
        kwargs["image_2"] = input_image2
    if input_image3 is not None:
        kwargs["image_3"] = input_image3
    if input_image4 is not None:
        kwargs["image_4"] = input_image4

    engine.set_api_key(api_key)
    payload = build_payload_func(engine, **kwargs)

    print(f"[OmniSimulator] Multi-image: Requesting 2 images from {provider}/{endpoint}", flush=True)
    image_urls = engine._submit_wavespeed(endpoint, payload, return_all_outputs=True)

    if not image_urls or len(image_urls) < 2:
        raise Exception(f"Expected 2 images but got {len(image_urls) if image_urls else 0}")

    generation_time = time.time() - start_time
    print(f"[OmniSimulator] Multi-image: Got {len(image_urls)} images in {generation_time:.1f}s", flush=True)

    results = []
    for i, (image_url, job_id) in enumerate(zip(image_urls[:2], job_ids)):
        try:
            response = requests.get(image_url, timeout=120)
            response.raise_for_status()
            img_pil = Image.open(io.BytesIO(response.content)).convert("RGB")
            img_width, img_height = img_pil.size

            img_np = np.array(img_pil).astype(np.float32) / 255.0
            tensor = torch.from_numpy(img_np)

            if use_cache and cache_keys and i < len(cache_keys):
                engine._save_to_cache(cache_keys[i], tensor)
                print(f"[OmniSimulator] Multi-image: Saved image {i+1} to cache", flush=True)

            output_filename, subfolder = engine._save_to_output(tensor, job_id, prompt_positive, filename_prefix, node_id)

            buffer = io.BytesIO()
            img_pil.save(buffer, format="JPEG", quality=85)
            b64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            preview_url = f"data:image/jpeg;base64,{b64_data}"

            results.append({
                "job_id": job_id,
                "status": "success",
                "state": "success",
                "image_url": preview_url,
                "width": img_width,
                "height": img_height,
                "filename": output_filename,
                "subfolder": subfolder,
            })
            print(f"[OmniSimulator] Multi-image: Processed image {i+1} for job #{job_id}", flush=True)
        except Exception as e:
            print(f"[OmniSimulator] Multi-image: Failed to process image {i+1}: {e}", flush=True)
            results.append({
                "job_id": job_id,
                "status": "failed",
                "state": "failed",
                "error": str(e),
            })

    return {
        "status": "success",
        "multi_image": True,
        "generation_time": generation_time,
        "results": results,
        "attempts": 1,
    }


@PromptServer.instance.routes.post('/omni/batch_gen_generate_single')
async def generate_single_image(request):
    try:
        data = await request.json()
        node_id = data.get("node_id")
        job_id = data.get("job_id")
        api_key = data.get("api_key")
        provider = data.get("provider")
        model = data.get("model")
        prompt_positive = data.get("prompt_positive", "")
        prompt_negative = data.get("prompt_negative", "")
        seed = data.get("seed", -1)
        aspect_ratio = data.get("aspect_ratio", "1:1")
        width = data.get("width", 1024)
        height = data.get("height", 1024)
        resolution = data.get("resolution")
        enable_safety = data.get("enable_safety_checker", True)
        use_negative = data.get("use_negative_prompt", True)
        max_retries = data.get("max_retries", 3)
        timeout = data.get("timeout", 600)
        use_cache = data.get("use_cache", True)
        input_image_b64 = data.get("input_image")
        input_image2_b64 = data.get("input_image2")
        input_image3_b64 = data.get("input_image3")
        input_image4_b64 = data.get("input_image4")
        filename_prefix = str(data.get("filename_prefix", "OMNI"))
        multi_image = data.get("multi_image", False)
        second_job_id = data.get("second_job_id")

        if multi_image and provider != "wavespeed.ai":
            print(f"[OmniSimulator] Multi-image mode only works with wavespeed.ai, ignoring for {provider}", flush=True)
            multi_image = False

        engine = BatchGeneratorEngine()
        engine.api_key = api_key

        job = GenerationJob(
            id=job_id,
            prompt_positive=prompt_positive,
            prompt_negative=prompt_negative,
            seed=seed,
        )

        def decode_b64_image(b64_data, label="image"):
            if not b64_data:
                return None
            print(f"[OmniSimulator] I2I mode: Decoding {label} (base64 length: {len(b64_data)})", flush=True)
            img_bytes = base64.b64decode(b64_data)
            pil_img = Image.open(io.BytesIO(img_bytes))
            if pil_img.mode in ["RGBA", "CMYK"]:
                pil_img = pil_img.convert("RGB")
            elif pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            img_array = np.array(pil_img).astype(np.float32) / 255.0
            tensor = torch.from_numpy(img_array)
            print(f"[OmniSimulator] {label} tensor shape: {tensor.shape}", flush=True)
            return tensor

        job.input_image = decode_b64_image(input_image_b64, "image_1")
        job.input_image2 = decode_b64_image(input_image2_b64, "image_2")
        job.input_image3 = decode_b64_image(input_image3_b64, "image_3")
        job.input_image4 = decode_b64_image(input_image4_b64, "image_4")

        if multi_image and second_job_id is not None:
            print(f"[OmniSimulator] MULTI-IMAGE MODE: Generating 2 images for jobs #{job_id} and #{second_job_id}", flush=True)
            input_images = [job.input_image, job.input_image2, job.input_image3, job.input_image4]
            cache_key_0 = engine.compute_multi_cache_key(
                prompt_positive, prompt_negative, seed, model, provider,
                aspect_ratio, resolution or "", input_images, slot=0
            )
            cache_key_1 = engine.compute_multi_cache_key(
                prompt_positive, prompt_negative, seed, model, provider,
                aspect_ratio, resolution or "", input_images, slot=1
            )

            cached_0 = engine._check_cache(cache_key_0) if use_cache else None
            cached_1 = engine._check_cache(cache_key_1) if use_cache else None

            if cached_0 is not None and cached_1 is not None:
                print(f"[OmniSimulator] MULTI-IMAGE CACHE HIT: Both images found in cache!", flush=True)
                results = []
                for i, (cached_tensor, j_id, c_key) in enumerate([(cached_0, job_id, cache_key_0), (cached_1, second_job_id, cache_key_1)]):
                    b64_url = engine._tensor_to_base64(cached_tensor)
                    img_height = cached_tensor.shape[0] if cached_tensor.ndim == 3 else cached_tensor.shape[1]
                    img_width = cached_tensor.shape[1] if cached_tensor.ndim == 3 else cached_tensor.shape[2]

                    output_filename, subfolder = engine._save_to_output(cached_tensor, j_id, prompt_positive, filename_prefix, node_id)

                    results.append({
                        "job_id": j_id,
                        "status": "success",
                        "state": "cached",
                        "image_url": b64_url,
                        "width": img_width,
                        "height": img_height,
                        "filename": output_filename,
                        "subfolder": subfolder,
                        "cache_hit": True,
                    })
                    print(f"[OmniSimulator] Cached image {i+1} for job #{j_id}", flush=True)

                return web.json_response({
                    "status": "success",
                    "multi_image": True,
                    "results": results,
                })

            try:
                result = await asyncio.to_thread(
                    _generate_multi_image,
                    engine=engine,
                    api_key=api_key,
                    provider=provider,
                    model=model,
                    prompt_positive=prompt_positive,
                    prompt_negative=prompt_negative,
                    aspect_ratio=aspect_ratio,
                    resolution=resolution,
                    width=width,
                    height=height,
                    use_negative_prompt=use_negative,
                    input_image=job.input_image,
                    input_image2=job.input_image2,
                    input_image3=job.input_image3,
                    input_image4=job.input_image4,
                    filename_prefix=filename_prefix,
                    job_ids=[job_id, second_job_id],
                    use_cache=use_cache,
                    cache_keys=[cache_key_0, cache_key_1],
                    node_id=node_id,
                )
                return web.json_response(result)
            except Exception as e:
                return web.json_response({
                    "status": "failed",
                    "job_id": job_id,
                    "second_job_id": second_job_id,
                    "error": str(e),
                    "multi_image": True,
                })

        print(f"[OmniSimulator] Job #{job_id} starting API call (parallel execution enabled)", flush=True)
        result_job = await asyncio.to_thread(
            engine.generate_job,
            job=job,
            api_key=api_key,
            provider=provider,
            model=model,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            width=width,
            height=height,
            enable_safety_checker=enable_safety,
            use_negative_prompt=use_negative,
            max_retries=max_retries,
            timeout=timeout,
            use_cache=use_cache,
        )

        if result_job.state in [JobState.SUCCESS, JobState.CACHED]:
            image_url = result_job.result_image_url
            width = result_job.result_image_width or 0
            height = result_job.result_image_height or 0

            output_filename = None
            subfolder = None
            if result_job.result_tensor is not None:
                output_filename, subfolder = engine._save_to_output(
                    result_job.result_tensor,
                    job_id,
                    prompt_positive,
                    filename_prefix,
                    node_id
                )
            return web.json_response({
                "status": "success",
                "job_id": job_id,
                "state": result_job.state.value,
                "image_url": image_url,
                "width": width,
                "height": height,
                "generation_time": result_job.generation_time,
                "cache_hit": result_job.cache_hit,
                "filename": output_filename,
                "subfolder": subfolder,
                "attempts": result_job.attempts,
            })
        else:
            return web.json_response({
                "status": "failed",
                "job_id": job_id,
                "state": result_job.state.value,
                "error": result_job.error or "Generation failed",
                "attempts": result_job.attempts,
            })
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


@dataclass
class GenerationJob:
    id: int
    prompt_positive: str
    prompt_negative: str
    seed: int
    state: JobState = JobState.PENDING
    input_image: Optional[torch.Tensor] = None
    input_image2: Optional[torch.Tensor] = None
    input_image3: Optional[torch.Tensor] = None
    input_image4: Optional[torch.Tensor] = None
    result_tensor: Optional[torch.Tensor] = None
    result_image_url: Optional[str] = None
    result_image_b64: Optional[str] = None
    result_image_width: int = 0
    result_image_height: int = 0
    result_filename: Optional[str] = None
    error: Optional[str] = None
    attempts: int = 0
    generation_time: float = 0.0
    cache_hit: bool = False


HIDDEN = {
    "prompt": "PROMPT",
    "extra_pnginfo": "EXTRA_PNGINFO",
    "uid": "UNIQUE_ID",
    "node_identifier": ("STRING", {"default": ""}),
    "generated_batch_data": ("STRING", {"default": "[]"}),
}


class BatchGeneratorEngine(OMNI_GenerativeAPIBase):
    def __init__(self):
        if OMNI_GenerativeAPIBase is not object:
            super().__init__()
        self._cache_dir = None
        self.api_key = None

    def set_api_key(self, api_key: str):
        if _BASE_API_LOADED and hasattr(super(), 'set_api_key'):
            super().set_api_key(api_key)
        else:
            self.api_key = api_key

    def _submit_wavespeed(self, endpoint: str, payload: dict, return_all_outputs: bool = False):
        if not _BASE_API_LOADED or OMNI_GenerativeAPIBase is object or not hasattr(super(), '_submit_wavespeed'):
            raise RuntimeError("[OmniSimulator] Base API engine is not initialized or missing '_submit_wavespeed'. Check generative_api_nodes.py")
        return super()._submit_wavespeed(endpoint, payload, return_all_outputs=return_all_outputs)

    def _submit_fal_sync(self, endpoint: str, payload: dict, timeout: int) -> str:
        # Если в базовом пакете есть собственная реализация fal, отдаем приоритет ей
        if _BASE_API_LOADED and hasattr(super(), '_submit_fal_sync'):
            return super()._submit_fal_sync(endpoint, payload, timeout)
            
        url = f"https://fal.run/{endpoint}"
        headers = {
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        if not response.ok:
            raise Exception(f"fal.ai error ({response.status_code}): {response.text[:300]}")

        result = response.json()
        if "images" in result and len(result["images"]) > 0:
            return result["images"][0]["url"]
        raise Exception("No image URL in fal.ai response")

    @property
    def cache_dir(self):
        if self._cache_dir is None:
            self._cache_dir = os.path.join(_ROOT, "cache", "batch_gen")
            os.makedirs(self._cache_dir, exist_ok=True)
        return self._cache_dir

    def _compute_cache_key(self, job: GenerationJob, model: str, provider: str,
                          aspect_ratio: str, resolution: str, has_input_images: bool) -> str:
        hasher = hashlib.sha256()
        hasher.update(job.prompt_positive.encode("utf-8"))
        hasher.update(job.prompt_negative.encode("utf-8"))
        hasher.update(str(job.seed).encode("utf-8"))
        hasher.update(model.encode("utf-8"))
        hasher.update(provider.encode("utf-8"))
        hasher.update(aspect_ratio.encode("utf-8"))
        if resolution:
            hasher.update(resolution.encode("utf-8"))
        if has_input_images:
            for i, img in enumerate([job.input_image, job.input_image2, job.input_image3, job.input_image4]):
                if img is not None:
                    hasher.update(f"image_{i+1}:".encode("utf-8"))
                    hasher.update(img.cpu().numpy().tobytes())
        return hasher.hexdigest()

    def compute_multi_cache_key(self, prompt_positive: str, prompt_negative: str, seed: int,
                                model: str, provider: str, aspect_ratio: str, resolution: str,
                                input_images: list, slot: int) -> str:
        hasher = hashlib.sha256()
        hasher.update(prompt_positive.encode("utf-8"))
        hasher.update(prompt_negative.encode("utf-8"))
        hasher.update(str(seed).encode("utf-8"))
        hasher.update(model.encode("utf-8"))
        hasher.update(provider.encode("utf-8"))
        hasher.update(aspect_ratio.encode("utf-8"))
        hasher.update(f"multi_slot:{slot}".encode("utf-8"))
        if resolution:
            hasher.update(resolution.encode("utf-8"))
        for i, img in enumerate(input_images):
            if img is not None:
                hasher.update(f"image_{i+1}:".encode("utf-8"))
                hasher.update(img.cpu().numpy().tobytes())
        return hasher.hexdigest()

    def _check_cache(self, cache_key: str) -> Optional[torch.Tensor]:
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.png")
        if os.path.exists(cache_file):
            try:
                img = Image.open(cache_file).convert("RGB")
                img_np = np.array(img).astype(np.float32) / 255.0
                return torch.from_numpy(img_np)
            except Exception:
                pass
        return None

    def _save_to_cache(self, cache_key: str, tensor: torch.Tensor):
        try:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.png")
            img_np = tensor.cpu().numpy()
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            img_pil.save(cache_file, "PNG")
        except Exception as e:
            print(f"   Cache save failed: {e}", flush=True)

    def _save_temp_preview(self, tensor: torch.Tensor, job_id: int) -> Optional[str]:
        try:
            from folder_paths import get_temp_directory
            temp_dir = os.path.join(get_temp_directory(), "batch_gen")
            os.makedirs(temp_dir, exist_ok=True)

            timestamp = int(time.time() * 1000)
            filename = f"job_{job_id}_{timestamp}.png"
            filepath = os.path.join(temp_dir, filename)

            img_np = tensor.cpu().numpy()
            if img_np.ndim == 4:
                img_np = img_np[0]
            if img_np.max() <= 1.0:
                img_np = (img_np * 255)

            img_np = img_np.astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            img_pil.save(filepath, "PNG", compress_level=4)

            if not os.path.exists(filepath):
                return None
            return filepath
        except Exception as e:
            print(f"   Temp preview save failed: {e}", flush=True)
            return None

    def _tensor_to_base64(self, tensor: torch.Tensor) -> Optional[str]:
        try:
            img_np = tensor.cpu().numpy()
            if img_np.ndim == 4:
                img_np = img_np[0]
            if img_np.max() <= 1.0:
                img_np = (img_np * 255)

            img_np = img_np.astype(np.uint8)
            img_pil = Image.fromarray(img_np)

            buffer = io.BytesIO()
            img_pil.save(buffer, format="JPEG", quality=85)
            b64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return f"data:image/jpeg;base64,{b64_data}"
        except Exception as e:
            print(f"   Base64 conversion failed: {e}", flush=True)
            return None

    def _save_to_output(self, tensor: torch.Tensor, job_id: int, prompt: str, filename_prefix: str = "OMNI", node_id: str = None) -> Tuple[Optional[str], Optional[str]]:
        try:
            import folder_paths
            comfy_output = folder_paths.get_output_directory()

            if "/" in filename_prefix:
                subdir, prefix = filename_prefix.rsplit("/", 1)
                output_dir = os.path.join(comfy_output, subdir)
            else:
                subdir = ""
                prefix = filename_prefix
                output_dir = comfy_output

            os.makedirs(output_dir, exist_ok=True)

            ulid = generate_ulid()
            filename = f"{prefix}_{ulid}.png"
            filepath = os.path.join(output_dir, filename)

            img_np = tensor.cpu().numpy()
            if img_np.ndim == 4:
                img_np = img_np[0]
            if img_np.max() <= 1.0:
                img_np = (img_np * 255)
            img_np = img_np.astype(np.uint8)

            img_pil = Image.fromarray(img_np)
            img_pil.save(filepath, "PNG")
            return filename, subdir
        except Exception as e:
            print(f"   Output save failed: {e}", flush=True)
            return None, None

    def generate_job(
        self,
        job: GenerationJob,
        api_key: str,
        provider: str,
        model: str,
        aspect_ratio: str,
        resolution: Optional[str],
        width: int,
        height: int,
        enable_safety_checker: bool,
        use_negative_prompt: bool,
        max_retries: int,
        timeout: int,
        use_cache: bool,
        on_retry_callback: Optional[callable] = None,
        progress_tracker: Optional['ProgressTracker'] = None,
        node_identifier: Optional[int] = None,
        filename_prefix: str = "OMNI",
    ) -> GenerationJob:
        start_time = time.time()
        job.state = JobState.GENERATING

        has_input = any(img is not None for img in [job.input_image, job.input_image2, job.input_image3, job.input_image4])
        cache_key = self._compute_cache_key(job, model, provider, aspect_ratio, resolution or "", has_input)

        if use_cache:
            cached = self._check_cache(cache_key)
            if cached is not None:
                job.state = JobState.CACHED
                job.result_tensor = cached
                job.cache_hit = True
                job.generation_time = time.time() - start_time

                try:
                    if cached.ndim == 4:
                        job.result_image_height = cached.shape[1]
                        job.result_image_width = cached.shape[2]
                    elif cached.ndim == 3:
                        job.result_image_height = cached.shape[0]
                        job.result_image_width = cached.shape[1]

                    b64_url = self._tensor_to_base64(cached)
                    if b64_url:
                        job.result_image_url = b64_url
                        job.result_image_b64 = b64_url
                except Exception as e:
                    print(f"   Failed to create cached preview: {e}", flush=True)
                return job

        if use_negative_prompt and job.prompt_negative.strip():
            if model in ["Nano Banana Pro", "Nano Banana"]:
                full_prompt = f"{job.prompt_positive}\n\nAvoid: {job.prompt_negative}"
            else:
                full_prompt = job.prompt_positive
        else:
            full_prompt = job.prompt_positive

        kwargs = {
            "api_key": api_key,
            "provider": provider,
            "model": model,
            "prompt": full_prompt,
            "seed": job.seed,
            "aspect_ratio": aspect_ratio,
            "width": width,
            "height": height,
            "enable_safety_checker": enable_safety_checker,
        }
        if resolution:
            kwargs["resolution"] = resolution
        if job.input_image is not None:
            kwargs["image_1"] = job.input_image
        if job.input_image2 is not None:
            kwargs["image_2"] = job.input_image2
        if job.input_image3 is not None:
            kwargs["image_3"] = job.input_image3
        if job.input_image4 is not None:
            kwargs["image_4"] = job.input_image4

        model_conf = MODEL_CONFIG.get(model)
        if not model_conf:
            job.state = JobState.FAILED
            job.error = f"Invalid model: {model}"
            return job

        provider_conf = model_conf["providers"].get(provider)
        if not provider_conf:
            job.state = JobState.FAILED
            job.error = f"Provider '{provider}' not supported for {model}"
            return job

        is_i2i = any(img is not None for img in [job.input_image, job.input_image2, job.input_image3, job.input_image4])
        endpoint = provider_conf["i2i_endpoint"] if is_i2i else provider_conf["t2i_endpoint"]
        build_payload_func = provider_conf["build_payload"]

        self.set_api_key(api_key)
        payload = build_payload_func(self, **kwargs)

        last_error = None
        for attempt in range(max_retries):
            if progress_tracker and node_identifier is not None:
                if progress_tracker.should_stop(node_identifier):
                    job.state = JobState.FAILED
                    job.error = "Stopped by user"
                    return job

            job.attempts = attempt + 1
            try:
                if progress_tracker and node_identifier is not None:
                    if progress_tracker.should_stop(node_identifier):
                        job.state = JobState.FAILED
                        job.error = "Stopped by user"
                        return job

                print(f"   🌐 Job #{job.id} making API call to {provider}...", flush=True)
                if provider == "fal.ai":
                    image_url = self._submit_fal_sync(endpoint, payload, timeout)
                else:
                    image_url = self._submit_wavespeed(endpoint, payload)

                image_response = requests.get(image_url, timeout=120)
                image_response.raise_for_status()
                image_pil = Image.open(io.BytesIO(image_response.content)).convert("RGB")

                job.result_image_width, job.result_image_height = image_pil.size
                image_np = np.array(image_pil).astype(np.float32) / 255.0
                tensor = torch.from_numpy(image_np)

                job.state = JobState.SUCCESS
                job.result_tensor = tensor
                job.generation_time = time.time() - start_time

                if use_cache:
                    self._save_to_cache(cache_key, tensor)

                filename, subfolder = self._save_to_output(tensor, job.id, job.prompt_positive, filename_prefix, node_identifier)
                if filename:
                    job.result_filename = filename
                    job.result_subfolder = subfolder

                b64_url = self._tensor_to_base64(tensor)
                if b64_url:
                    job.result_image_url = b64_url
                    job.result_image_b64 = b64_url
                else:
                    temp_image_path = self._save_temp_preview(tensor, job.id)
                    if temp_image_path:
                        job.result_image_url = f"/view?filename={os.path.basename(temp_image_path)}&subfolder=batch_gen&type=temp"

                return job
            except Exception as e:
                last_error = str(e)
                is_moderation = any(x in last_error.lower() for x in ["422", "content", "policy", "flagged", "safety", "moderation"])
                is_timeout = "timeout" in last_error.lower()
                is_rate_limit = "429" in last_error or "rate" in last_error.lower()

                if attempt < max_retries - 1:
                    job.state = JobState.RETRYING
                    error_msg = "Content policy violation" if is_moderation else "Rate limited" if is_rate_limit else "Timeout" if is_timeout else last_error[:50]
                    job.error = f"Retry {attempt + 1}: {error_msg}"
                    if on_retry_callback:
                        on_retry_callback(job)

                    if is_moderation:
                        wait_time = 0.5 + (attempt * 0.5)
                    elif is_rate_limit:
                        wait_time = 5 + (attempt * 5)
                    elif is_timeout:
                        wait_time = 2 + (attempt * 2)
                    else:
                        wait_time = 2 ** attempt

                    time.sleep(wait_time)
                    job.state = JobState.GENERATING

        job.state = JobState.FAILED
        job.error = last_error
        job.generation_time = time.time() - start_time
        return job


class OMNI_BatchImageGenerator(PreviewImage):
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("images", "prompt_list_positive", "indexes", "count", "failed_count")
    OUTPUT_IS_LIST = (True, True, False, False, False)
    INPUT_IS_LIST = True
    FUNCTION = "generate"
    CATEGORY = "OmniSimulator/Interactive"
    OUTPUT_NODE = False
    DESCRIPTION = "Parallel batch image generator with auto-retry and interactive selection"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"forceInput": True}),
                "provider": ("STRING", {"forceInput": True}),
                "model": ("STRING", {"forceInput": True}),
                "prompt_list_positive": ("STRING", {"forceInput": True}),
                "aspect_ratio": ("STRING", {"forceInput": True}),
                "max_parallel": ("INT", {"default": 5, "min": 1, "max": 100}),
                "max_retries": ("INT", {"default": 3, "min": 1, "max": 10}),
                "timeout": ("INT", {"default": 600, "min": 60, "max": 3600}),
            },
            "optional": {
                "prompt_list_negative": ("STRING", {"forceInput": True}),
                "seed_list": ("INT", {"forceInput": True}),
                "width": ("INT", {"forceInput": True}),
                "height": ("INT", {"forceInput": True}),
                "resolution": ("STRING", {"forceInput": True}),
                "images": ("IMAGE", {"forceInput": True}),
                "images2": ("IMAGE", {"forceInput": True}),
                "images3": ("IMAGE", {"forceInput": True}),
                "images4": ("IMAGE", {"forceInput": True}),
                "enable_img2img": ("BOOLEAN", {"default": False}),
                "use_negative_prompt": ("BOOLEAN", {"default": True}),
                "auto_retry": ("BOOLEAN", {"default": True}),
                "use_cache": ("BOOLEAN", {"default": True}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "multi_image": ("BOOLEAN", {"default": False}),
                "bypass_filter": ("BOOLEAN", {"default": False}),
                "filename_prefix": ("STRING", {"default": "OMNI"}),
            },
            "hidden": HIDDEN,
        }

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("NaN")

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def __init__(self):
        super().__init__()
        self.engine = BatchGeneratorEngine()

    def _extract_val(self, val, default=None):
        if val is None:
            return default
        if isinstance(val, list):
            return val[0] if len(val) > 0 else default
        return val

    def _extract_list(self, val, default_len: int = 0) -> list:
        if val is None:
            return []
        if isinstance(val, list):
            return val
        return [val]

    def _load_generated_images(self, generated_batch: list, node_identifier: str):
        from PIL import Image
        import folder_paths

        comfy_output = folder_paths.get_output_directory()
        loaded_images = []
        loaded_prompts = []
        failed_count = 0

        for idx, job_data in enumerate(generated_batch):
            state = job_data.get("state", "pending")
            filename = job_data.get("filename")
            prompt = job_data.get("prompt_positive", "")
            
            if "subfolder" in job_data:
                subfolder = job_data["subfolder"]
            elif "output_subdir" in job_data:
                subfolder = job_data["output_subdir"]
            else:
                subfolder = ""

            if state in ["success", "cached"] and filename:
                output_folder = os.path.join(comfy_output, subfolder) if subfolder else comfy_output
                filepath = os.path.join(output_folder, filename)

                if os.path.exists(filepath):
                    try:
                        pil_img = Image.open(filepath)
                        img_array = np.array(pil_img).astype(np.float32) / 255.0
                        img_tensor = torch.from_numpy(img_array)
                        loaded_images.append(img_tensor)
                        loaded_prompts.append(prompt)
                    except Exception as e:
                        print(f"[OmniSimulator] Failed to load job {idx}: {e}")
                        failed_count += 1
                else:
                    failed_count += 1
            else:
                if state == "failed":
                    failed_count += 1

        if len(loaded_images) == 0:
            empty_img = torch.zeros((1, 512, 512, 3))
            return ([empty_img], [""], 0, 0, failed_count)

        return (
            [img.unsqueeze(0) for img in loaded_images],
            loaded_prompts,
            len(loaded_images),
            failed_count,
        )

    def generate(self, **kwargs):
        generated_batch_data = self._extract_val(kwargs.get('generated_batch_data'), "[]")
        node_identifier = self._extract_val(kwargs.get('node_identifier'))

        try:
            generated_batch = json.loads(generated_batch_data) if generated_batch_data else []
        except json.JSONDecodeError:
            generated_batch = []

        if generated_batch and len(generated_batch) > 0:
            return self._load_generated_images(generated_batch, node_identifier)

        raise ValueError(
            "No pre-generated images found. "
            "Please click 'Generate All' in the OmniSimulator BIG node UI first, "
            "then run the workflow to output the generated images."
        )


NODE_CLASS_MAPPINGS = {
    "OMNI_BatchImageGenerator": OMNI_BatchImageGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OMNI_BatchImageGenerator": "OMNI Batch Image Generator",
}
"""Bounded subset of the local Ollama protocol used by the navigation clients."""

import base64
import io
import json
import math
from pathlib import Path

from jsonschema import Draft7Validator
from PIL import Image


MAX_BODY = 12 * 1024 * 1024
ALIASES = {"qwen3:4b": ("Qwen3-4B", False),
           "qwen3-vl:4b-instruct": ("Qwen3-VL-4B-Instruct", True)}


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("Non-finite JSON number")

    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)


def model_info(body, root):
    if not isinstance(body, dict) or body.get("model") not in ALIASES:
        raise ValueError("Unknown model; use qwen3:4b or qwen3-vl:4b-instruct")
    name, vision = ALIASES[body["model"]]
    path = root / name
    config = path / "config.json"
    index = path / "model.safetensors.index.json"
    if index.is_file():
        manifest = strict_json(index.read_bytes())
        weights = manifest.get("weight_map", {}) if isinstance(manifest, dict) else {}
        if not isinstance(weights, dict) or not weights:
            raise ValueError("Invalid model shard index")
        shards = set(weights.values())
    else:
        shards = {"model.safetensors"}
    if (not config.is_file() or any(not isinstance(shard, str) or Path(shard).name != shard
                                  or not (path / shard).is_file() for shard in shards)):
        raise ValueError(f"Model files missing: {path}; run model_service.download first")
    return path, vision


def decode_image(value):
    if not isinstance(value, str) or len(value) > 11 * 1024 * 1024:
        raise ValueError("Image must be bounded base64, not a URL")
    try:
        data = base64.b64decode(value, validate=True)
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in ("PNG", "JPEG") or not (0 < image.width <= 1920 and 0 < image.height <= 1080):
                raise ValueError("Image must be PNG/JPEG up to 1920x1080")
            image.load()
            return image.convert("RGB")
    except (OSError, Image.DecompressionBombError) as error:
        raise ValueError("Invalid image") from error


def validate_chat(body, vision):
    if body.get("stream", False) is not False or body.get("think", False) is not False:
        raise ValueError("Only non-streaming, non-thinking requests are supported")
    messages = body.get("messages")
    if not isinstance(messages, list) or not 1 <= len(messages) <= 32:
        raise ValueError("Expected 1 to 32 messages")
    result, image_count, text_count = [], 0, 0
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in ("system", "user", "assistant"):
            raise ValueError("Invalid message role")
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("Message content must be text")
        text_count += len(content)
        images = message.get("images", [])
        if not isinstance(images, list):
            raise ValueError("images must be a list")
        image_count += len(images)
        if image_count > 1 or (images and (not vision or message["role"] != "user")):
            raise ValueError("Only one user image is supported, on the vision model")
        if vision:
            parts = [{"type": "image", "image": decode_image(value)} for value in images]
            parts.append({"type": "text", "text": content})
            content = parts
        result.append({"role": message["role"], "content": content})
    if text_count > 24000:
        raise ValueError("Prompt exceeds text limit")
    options = body.get("options", {})
    if not isinstance(options, dict) or set(options) - {"temperature", "num_ctx", "num_predict"}:
        raise ValueError("Unsupported generation options")
    temperature = options.get("temperature", 0)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature != 0:
        raise ValueError("Only deterministic temperature=0 is supported")
    context, tokens = options.get("num_ctx", 4096), options.get("num_predict", 512)
    if type(context) is not int or not 128 <= context <= 4096 or type(tokens) is not int or not 1 <= tokens <= 1024:
        raise ValueError("Context must be 128..4096; output tokens 1..1024")
    output_schema = body.get("format")
    if output_schema is not None:
        if not isinstance(output_schema, dict) or len(json.dumps(output_schema)) > 16000:
            raise ValueError("format must be a bounded JSON Schema object")
        Draft7Validator.check_schema(output_schema)
    if body.get("keep_alive", 0) != 0:
        raise ValueError("Only keep_alive=0 is supported; use server --keep-loaded for residency")
    return result, output_schema, context, tokens

"""
image_gen.py
Thin wrapper around OpenAI's image generation endpoint.
"""

from __future__ import annotations

import base64
from typing import List

from openai import OpenAI

IMAGE_MODEL = "gpt-image-1"


def generate_images(
    client: OpenAI,
    prompt: str,
    n: int = 1,
    size: str = "1024x1024",
    model: str = IMAGE_MODEL,
) -> List[bytes]:
    """Generate one or more images from a text prompt.

    Returns a list of raw PNG bytes (decoded from base64), ready to be
    displayed or written to disk.
    """
    result = client.images.generate(
        model=model,
        prompt=prompt,
        n=n,
        size=size,
    )

    images = []
    for item in result.data:
        images.append(base64.b64decode(item.b64_json))
    return images

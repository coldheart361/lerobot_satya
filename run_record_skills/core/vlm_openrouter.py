"""
vlm_openrouter.py
=================
Thin wrapper around OpenRouter for VLM queries.

WHY IT EXISTS:
  - Centralize the API config (model name, base URL, key)
  - All other modules import this — swap model in ONE place
  - Handles image encoding to base64 once
"""

import os
import base64
import cv2
import numpy as np
from openai import OpenAI


class VLM:
    def __init__(self, model="google/gemini-3-flash-preview", api_key=None):
        self.model = model
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        self.client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    def query(self, prompt: str, image: np.ndarray = None, max_tokens: int = 3000) -> str:
        """Send prompt (+ optional RGB image) to the VLM. Returns raw text."""
        content = []
        if image is not None:
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            if not ok:
                raise RuntimeError("failed to encode image")
            b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
        content.append({"type": "text", "text": prompt})

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content
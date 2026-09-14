# conceptgraph/utils/vlm_gemini.py
"""
Gemini API adapter for concept-graphs VLM integration using google-genai and OpenCV.
Install SDK with: pip install google-genai

This module exposes backward-compatible functions with the same names
as the original OpenAI-based vlm.py (get_obj_rel_from_image_gpt4v,
get_obj_captions_from_image_gpt4v) so you can import it in place of
conceptgraph.utils.vlm without changing call sites.
"""

import os
import cv2
import logging
from typing import List
from google import genai
from google.genai import types

# Silence verbose third-party loggers (httpx, httpcore, PIL, google)
for logger_name in ("httpx", "httpcore", "google", "google.genai", "PIL", "PngImagePlugin"):
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# Reuse parsers and prompts from the original vlm implementation for
# maximum compatibility.
from .vlm import (
    extract_list_of_tuples,
    vlm_extract_object_captions,
    system_prompt_only_top,
    system_prompt_captions,
    system_prompt_consolidate_captions,
    get_obj_rel_from_image_gpt4v as _openai_get_obj_rel,
    get_obj_captions_from_image_gpt4v as _openai_get_obj_captions,
    consolidate_captions as _openai_consolidate_captions,
)

class GeminiClient:
    """
    Client for calling Gemini multimodal API using the official google-genai SDK.
    """

    def __init__(self, api_key: str = None, model: str = "gemini-3.6-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No Gemini API key provided. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self.client = genai.Client(api_key=self.api_key)
        self.model = model

    def _load_image_bytes(self, image_path: str):
        """Read image via OpenCV and return JPEG byte payload for Gemini API."""
        if not os.path.exists(image_path):
            return None
        
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        success, encoded_img = cv2.imencode(".jpg", img)
        if not success:
            return None
            
        return types.Part.from_bytes(
            data=encoded_img.tobytes(),
            mime_type="image/jpeg",
        )
    
    def chat_with_image_for_rels(self, image_path: str, label_list: List[str]) -> List[tuple]:
        """
        Analyze spatial relationships between objects in an image.
        Keeps the same system/user prompt roles as the original vlm.py.
        """
        image_part = self._load_image_bytes(image_path)
        if image_part is None:
            print(f"[Gemini VLM] Image not found or could not be read: {image_path}")
            return []

        user_query = f"Here is the list of labels for the annotations of the objects in the image: {label_list}. Please describe the spatial relationships between the objects in the image."

        try:
            config = types.GenerateContentConfig(
                system_instruction=system_prompt_only_top
            )
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=[image_part, user_query],
                config=config,
            )
            
            model_text = response.text or ""
            print(f"[Gemini VLM] Relations response: {model_text[:200]}...")
            return extract_list_of_tuples(model_text)
        except Exception as e:
            print(f"[Gemini VLM] Error calling Gemini API for relations: {e}")
            return []


    def chat_with_image_for_captions(self, image_path: str, label_list: List[str]) -> List[dict]:
        """
        Generate captions for objects in an image using the same system prompt
        as the original implementation.
        """
        image_part = self._load_image_bytes(image_path)
        if image_part is None:
            print(f"[Gemini VLM] Image not found or could not be read: {image_path}")
            return []

        user_query = f"Here is the list of labels for the annotations of the objects in the image: {label_list}. Please accurately caption the objects in the image."

        try:
            config = types.GenerateContentConfig(
                system_instruction=system_prompt_captions
            )

            response = self.client.models.generate_content(
                model=self.model,
                contents=[image_part, user_query],
                config=config,
            )

            model_text = response.text or ""
            print(f"[Gemini VLM] Captions response: {model_text[:200]}...")
            return vlm_extract_object_captions(model_text)

        except Exception as e:
            print(f"[Gemini VLM] Error calling Gemini API for captions: {e}")
            return []


    def chat_with_consolidate_captions(self, caption_list: List[str]) -> str:
        """
        Consolidate a list of captions into a single coherent description.
        """
        captions_text = "\n".join([f"{cap['caption']}" for cap in caption_list if cap['caption'] is not None])
        user_query = f"Here are several captions for the same object:\n{captions_text}\n\nPlease consolidate these into a single, clear caption that accurately describes the object."

        try:
            config = types.GenerateContentConfig(
                system_instruction=system_prompt_consolidate_captions
            )

            response = self.client.models.generate_content(
                model=self.model,
                contents=[user_query],
                config=config,
            )

            model_text = response.text or ""
            print(f"[Gemini LLM] Consolidated captions response: {model_text[:200]}...")
            return model_text

        except Exception as e:
            print(f"[Gemini LLM] Error calling Gemini API for consolidated captions: {e}")
            return ""
        

# Backwards-compatible wrapper functions with same signatures as vlm.py
# So the rest of the codebase can import these directly from this module
# instead of conceptgraph.utils.vlm.


def get_obj_rel_from_image_gpt4v(client, image_path: str, label_list: list):
    """
    Backwards-compatible function. If `client` is an instance of GeminiClient
    (or has the method chat_with_image_for_rels), use it. Otherwise, fall back
    to the original OpenAI-based implementation imported from vlm.py.
    """
    if client is None:
        return []
    if hasattr(client, "chat_with_image_for_rels"):
        return client.chat_with_image_for_rels(image_path, label_list)
    # Fallback: assume it's the original OpenAI client
    return _openai_get_obj_rel(client, image_path, label_list)


def get_obj_captions_from_image_gpt4v(client, image_path: str, label_list: list):
    """
    Backwards-compatible captions wrapper.
    """
    if client is None:
        return []
    if hasattr(client, "chat_with_image_for_captions"):
        return client.chat_with_image_for_captions(image_path, label_list)
    return _openai_get_obj_captions(client, image_path, label_list)


def consolidate_captions(client, captions: list):
    """
    Backwards-compatible function to consolidate captions.
    """
    if client is None:
        return ""
    if hasattr(client, "chat_with_consolidate_captions"):
        return client.chat_with_consolidate_captions(captions)
    # Fallback: assume it's the original OpenAI client
    return _openai_consolidate_captions(client, captions)

def get_gemini_client(api_key: str = None, model: str = "gemini-3.6-flash"):
    """
    Factory function to create a Gemini client.
    """
    return GeminiClient(api_key=api_key, model=model)

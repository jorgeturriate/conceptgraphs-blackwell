# conceptgraph/utils/vlm_ollama.py
import os
from PIL import Image
import base64
import logging
from typing import List
from openai import OpenAI
import ollama

# Silence verbose third-party loggers (httpx, httpcore, PIL, google)
for logger_name in ("httpx", "httpcore", "google", "google.genai", "PIL", "PngImagePlugin"):
    logging.getLogger(logger_name).setLevel(logging.WARNING)

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

class OllamaClient:
    """
    Client for calling a local VLM/LLM using the compatible endpoint of OpenAI of Ollama.
    """
    def __init__(self, base_url: str = "http://localhost:11434/v1", model: str = "llava"):
        # We use the OpenAI client library but point it to the local Ollama endpoint.
        self.model = model
        self.base_url = base_url
        self.use_native_ollama = "qwen3.5" in model.lower() or "r1" in model.lower()

        if not self.use_native_ollama:
            self.client = OpenAI(
                base_url=base_url,
                api_key="ollama"
            )


    def _encode_image_base64(self, image_path: str, resize = False, target_size: int=512):
        print(f"Checking if image exists at path: {image_path}")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        if not resize:
            # Open the image
            print(f"Opening image from path: {image_path}")
            with open(image_path, "rb") as img_file:
                encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
                print("Image encoded in base64 format.")
            return encoded_image
        
        print(f"Opening image from path: {image_path}")
        with Image.open(image_path) as img:
            # Determine scaling factor to maintain aspect ratio
            original_width, original_height = img.size
            print(f"Original image dimensions: {original_width} x {original_height}")
            
            if original_width > original_height:
                scale = target_size / original_width
                new_width = target_size
                new_height = int(original_height * scale)
            else:
                scale = target_size / original_height
                new_height = target_size
                new_width = int(original_width * scale)
    
            print(f"Resized image dimensions: {new_width} x {new_height}")
    
            # Resizing the image
            img_resized = img.resize((new_width, new_height), Image.LANCZOS)
            print("Image resized successfully.")
            
            # Convert the image to bytes and encode it in base64
            with open("temp_resized_image.jpg", "wb") as temp_file:
                img_resized.save(temp_file, format="JPEG")
                print("Resized image saved temporarily for encoding.")
            
            # Open the temporarily saved image for base64 encoding
            with open("temp_resized_image.jpg", "rb") as temp_file:
                encoded_image = base64.b64encode(temp_file.read()).decode('utf-8')
                print("Image encoded in base64 format.")
            
            # Clean up the temporary file
            os.remove("temp_resized_image.jpg")
            print("Temporary file removed.")
    
        return encoded_image

    def chat_with_image_for_rels(self, image_path: str, label_list: List[str]) -> List[tuple]:
        base64_img = self._encode_image_base64(image_path)
        if base64_img is None:
            return []

        user_query = f"Here is the list of labels for the annotations of the objects in the image: {label_list}. Please describe the spatial relationships between the objects in the image."

        try:
            if self.use_native_ollama:
                # Qwen3.5 has a thinking native mode that can be disabled with think=False. This is useful for generating more concise responses.
                response = ollama.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt_only_top},
                        {
                            "role": "user",
                            "content": user_query,
                            "images": [base64_img]
                        }
                    ],
                    options={
                        "temperature": 0.1,
                        "num_ctx": 4096
                    },
                    think=False
                )
                model_text = response['message']['content'] or ""
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt_only_top},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_query},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
                                }
                            ]
                        }
                    ],
                    temperature=0.2,
                )
                model_text = response.choices[0].message.content or ""
            return extract_list_of_tuples(model_text)
        except Exception as e:
            print(f"[Ollama VLM] Error in relationships: {e}")
            return []

    def chat_with_image_for_captions(self, image_path: str, label_list: List[str]) -> List[dict]:
        base64_img = self._encode_image_base64(image_path)
        if base64_img is None:
            return []

        user_query = f"Here is the list of labels for the annotations of the objects in the image: {label_list}. Please accurately caption the objects in the image."

        try:
            if self.use_native_ollama:
                response = ollama.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt_captions},
                        {
                            "role": "user",
                            "content": user_query,
                            "images": [base64_img]
                        }
                    ],
                    options={
                        "temperature": 0.1,
                        "num_ctx": 4096
                    },
                    think=False
                )
                model_text = response['message']['content'] or ""
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt_captions},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_query},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
                                }
                            ]
                        }
                    ],
                    temperature=0.2,
                )
                model_text = response.choices[0].message.content or ""
            return vlm_extract_object_captions(model_text)
        except Exception as e:
            print(f"[Ollama VLM] Error in captions: {e}")
            return []

    def chat_with_consolidate_captions(self, caption_list: List[str]) -> str:
        valid_captions = [cap['caption'] for cap in caption_list if cap.get('caption') is not None]
        if not valid_captions:
            return ""

        captions_text = "\n".join(valid_captions)
        user_query = f"Here are several captions for the same object:\n{captions_text}\n\nPlease consolidate these into a single, clear caption that accurately describes the object."

        try:
            if self.use_native_ollama:
                # LLAMADA NATIVA CON OLLAMA
                response = ollama.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt_consolidate_captions},
                        {"role": "user", "content": user_query}
                    ],
                    options={
                        "temperature": 0.1,
                        "num_ctx": 4096
                    },
                    think=False
                )
                return response['message']['content'] or ""
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt_consolidate_captions},
                        {"role": "user", "content": user_query}
                    ],
                    temperature=0.2,
                )
                return response.choices[0].message.content or ""
        except Exception as e:
            print(f"[Ollama LLM] Error in consolidation: {e}")
            return ""

def get_ollama_client(base_url: str = "http://localhost:11434/v1", model: str = "llava"):
    return OllamaClient(base_url=base_url, model=model)

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
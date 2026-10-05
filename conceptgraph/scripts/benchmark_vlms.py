import os
from PIL import Image
import time
import base64
import json
from typing import List
from openai import OpenAI
import ollama

# -----------------------------------------------------------------------------
# 1. Prompts oficiales de ConceptGraphs para la tarea de Captions
# -----------------------------------------------------------------------------
system_prompt_captions = '''
You are an agent specializing in accurate captioning objects in an image.

In the images, each object is annotated with a bright numeric id (i.e. a number) and a corresponding colored contour outline. Your task is to analyze the images and output in a structured format, the captions for the objects.

You will also be given a text list of the numeric ids and names of the objects in the image. The list will be in the format: ["1: name1", "2: name2", "3: name3" ...]

The names were obtained from a simple object detection system and may be inaacurate.

Your response should be in the format of a list of dictionaries, where each dictionary contains the id, name, and caption of an object. Your response will be evaluated as a python list of dictionaries, so make sure to format it correctly. An example of the expected response format is as follows:
[
    {"id": "1", "name": "object1", "caption": "concise description of the object1 in the image"},
    {"id": "2", "name": "object2", "caption": "concise description of the object2 in the image"},
    {"id": "3", "name": "object3", "caption": "concise description of the object3 in the image"}
]

And each caption must be a concise description of the object in the image.

Do NOT provide any reasoning, thinking process, or intro text. Output ONLY the requested JSON list immediately.
'''

system_prompt_only_top = '''
You are an agent specializing in identifying the physical and spatial relationships in annotated images for 3D mapping.

In the images, each object is annotated with a bright numeric id (i.e. a number) and a corresponding colored contour outline. Your task is to analyze the images and output a list of tuples describing the physical relationships between objects. Format your response as follows: [("1", "relation type", "2"), ...]. When uncertain, return an empty list.

Note that you are describing the **physical relationships** between the **objects inside** the image.

You will also be given a text list of the numeric ids of the objects in the image. The list will be in the format: ["1: name1", "2: name2", "3: name3" ...], only output the physical relationships between the objects in the list.

The relation types you must report are:
- phyically placed on top of: ("object x", "on top of", "object y") 
- phyically placed underneath: ("object x", "under", "object y") 

An illustrative example of the expected response format might look like this:
[("object 1", "on top of", "object 2"), ("object 3", "under", "object 2"), ("object 4", "on top of", "object 3")]. Do not put the names of the objects in your response, only the numeric ids.

Do not include any other information in your response. Only output a parsable list of tuples describing the given physical relationships between objects in the image.
'''

def encode_image_base64(image_path: str, resize = False, target_size: int=512) -> str:
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


def benchmark_model(model_name: str, image_path: str, label_list: List[str], base_url: str = "http://localhost:11434/v1"):
    """
    Ejecuta el benchmark de inferencia para un VLM en Ollama midiendo tiempos de respuesta.
    """
    print(f"\n==================================================")
    print(f" Iniciando prueba para el modelo: [{model_name}]")
    print(f"==================================================")

    # Inicializar cliente compatible con OpenAI apuntando a Ollama
    client = OpenAI(base_url=base_url, api_key="ollama")

    base64_img = encode_image_base64(image_path)
    user_query = f"Here is the list of labels for the annotations of the objects in the image: {label_list}. Please accurately caption the objects in the image."

    # Cronometrar tiempo total transcurrido (Latencia)
    start_time = time.time()

    try:
        response = client.chat.completions.create(
            model=model_name,
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
            temperature=0.1,
            extra_body= {
                "options": {"think": False, "num_ctx": 7000}
            }
        )
        elapsed_time = time.time() - start_time
        model_output = response.choices[0].message.content or ""

        # Extraer información de tokens si la API la expone
        usage = getattr(response, 'usage', None)
        completion_tokens = usage.completion_tokens if usage else 0
        tok_per_sec = (completion_tokens / elapsed_time) if elapsed_time > 0 and completion_tokens else 0.0

        print(f"\n--- [RESULTADOS DEL BENCHMARK] ---")
        print(f"Modelo:                  {model_name}")
        print(f"Tiempo total por frame:  {elapsed_time:.2f} segundos")
        if tok_per_sec > 0:
            print(f"Velocidad de generación: {tok_per_sec:.2f} tokens/seg (Tokens generados: {completion_tokens})")
        
        print("\n--- [RESPUESTA DEL VLM] ---")
        print(model_output)
        print("--------------------------------------------------\n")

    except Exception as e:
        print(f"[Error de ejecución en {model_name}]: {e}")


def benchmark_relation(model_name: str, image_path: str, label_list: List[str], base_url: str = "http://localhost:11434/v1"):
    """
    Ejecuta el benchmark de inferencia de relaciones para un VLM en Ollama midiendo tiempos de respuesta.
    """
    print(f"\n==================================================")
    print(f" Iniciando prueba para el modelo: [{model_name}]")
    print(f"==================================================")

    # Inicializar cliente compatible con OpenAI apuntando a Ollama
    client = OpenAI(base_url=base_url, api_key="ollama")

    base64_img = encode_image_base64(image_path)
    user_query = f"Here is the list of labels for the annotations of the objects in the image: {label_list}. Please describe the spatial relationships between the objects in the image."

    # Cronometrar tiempo total transcurrido (Latencia)
    start_time = time.time()

    try:
        response = client.chat.completions.create(
            model=model_name,
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
            temperature=0.1,
            extra_body= {
                "options": {"think": False}
            }
        )
        elapsed_time = time.time() - start_time
        model_output = response.choices[0].message.content or ""

        # Extraer información de tokens si la API la expone
        usage = getattr(response, 'usage', None)
        completion_tokens = usage.completion_tokens if usage else 0
        tok_per_sec = (completion_tokens / elapsed_time) if elapsed_time > 0 and completion_tokens else 0.0

        print(f"\n--- [RESULTADOS DEL BENCHMARK] ---")
        print(f"Modelo:                  {model_name}")
        print(f"Tiempo total por frame:  {elapsed_time:.2f} segundos")
        if tok_per_sec > 0:
            print(f"Velocidad de generación: {tok_per_sec:.2f} tokens/seg (Tokens generados: {completion_tokens})")
        
        print("\n--- [RESPUESTA DEL VLM] ---")
        print(model_output)
        print("--------------------------------------------------\n")

    except Exception as e:
        print(f"[Error de ejecución en {model_name}]: {e}")

def benchmark_qwen_native(image_path: str, user_query: str, system_prompt: str):
    # Cargar imagen a base64
    base64_img = encode_image_base64(image_path)

    start_time = time.time()

    # Llamada directa a la API nativa de Ollama pasando la opción think=False
    response = ollama.chat(
        #model="qwen3.5:9b",
        model="qwen3-vl:8b",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_query,
                "images": [base64_img]
            }
        ],
        #options={
        #    "temperature": 0.1,
        #    "think": False,       # Forzar la desactivación del Thinking Process
        #    "num_ctx": 4096       # Mantener el contexto controlado para la GPU
        #}
        think=False
    )

    elapsed_time = time.time() - start_time
    content = response['message']['content']

    # --- CÁLCULO DE TOKENS POR SEGUNDO ---
    eval_count = response.get('eval_count', 0)        # Tokens generados
    eval_duration = response.get('eval_duration', 1)  # Duración en nanosegundos
    
    # Convertir nanosegundos a segundos
    tok_per_sec = (eval_count / (eval_duration / 1e9)) if eval_duration > 0 else 0.0

    print(f"--- [RESULTADOS BENCHMARK QWEN3.5] ---")
    print(f"Tiempo total por frame:  {elapsed_time:.2f} segundos")
    print(f"Velocidad de generación: {tok_per_sec:.2f} tok/s (Tokens generados: {eval_count})")
    print("\nSalida del VLM:")
    print(content)


if __name__ == "__main__":
    # RUTA DE EJEMPLO: Reemplaza por una imagen anotada válida de tu dataset (Replica o Stonefish)
    IMAGE_PATH = "/home/jorgeturriate/ConceptGraphsTesting/datasets/Replica/room0/exps/s_detections_stride10/vis/frame000000annotated_for_vlm.jpg"
    
    # LISTA DE OBJETOS SIMULADA DE PRUEBA
    TEST_LABELS = ["0: sofa chair", "1: stool", "2: pillow", "3: coffee kettle", "5: stool", "8: closet door", "9: window", "10: tissue box", "11: power outlet", "12: end table", "13: coffee table", "14: light switch", "16: plate", "17: shelf", "18: poster", "20: cabinet", "24: closet door", "33: coffee table"]

    # Modelos multimodales a comparar (asegúrate de haberlos descargado con 'ollama pull <modelo>')
    models_to_test = ["qwen3.5:9b"] #" qwen3-vl:8b qwen2.5vl llava gemma4"

    #for model in models_to_test:
    #    benchmark_model(model_name=model, image_path=IMAGE_PATH, label_list=TEST_LABELS)
    user_query1 = f"Here is the list of labels for the annotations of the objects in the image: {TEST_LABELS}. Please accurately caption the objects in the image."
    user_query2 = f"Here is the list of labels for the annotations of the objects in the image: {TEST_LABELS}. Please describe the spatial relationships between the objects in the image."
    benchmark_qwen_native(image_path=IMAGE_PATH, user_query=user_query2, system_prompt=system_prompt_only_top)
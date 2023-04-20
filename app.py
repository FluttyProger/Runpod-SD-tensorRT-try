import os
import torch
from torch import autocast
import base64
from io import BytesIO
from transformers import pipeline
from diffusers import DDIMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    
    model_name = os.getenv("MODEL_NAME")
    scheduler = DDIMScheduler.from_pretrained(model_name, subfolder="scheduler")

    model = StableDiffusionPipeline.from_pretrained(model_name,
                                                    custom_pipeline="stable_diffusion_tensorrt_txt2img",
                                                    torch_dtype=torch.float16,
                                                    scheduler=scheduler).set_cached_folder(model_name).to("cuda")


def inference(model_inputs:dict):
    global model

    prompt = model_inputs.get('prompt', None)
    heigth = model_inputs.get('heigth', 768)
    width = model_inputs.get('width', 768)
    steps = model_inputs.get('steps', 20)
    guidance_scale = model_inputs.get('guidance_scale', 9)
    seed = model_inputs.get('seed', None)

    if not prompt: return {'message': 'No prompt was provided'}
    
    generator = None
    if seed: generator = torch.Generator("cuda").manual_seed(seed)
    
    with autocast("cuda"):
        image = model(prompt, guidance_scale=guidance_scale, heigth=heigth, width=width, num_inference_steps=steps, generator=generator).images[0]
    
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return {'image_base64': image_base64}

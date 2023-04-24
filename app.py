import os
import torch
from torch import autocast
import base64
from io import BytesIO
from transformers import pipeline
from diffusers import DDIMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline

os.environ["CUDA_MODULE_LOADING"] = "LAZY"

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    model_name = os.getenv("MODEL_NAME")
    model_rev = os.getenv("MODEL_REV")
    scheduler = DDIMScheduler.from_pretrained(model_name,
                                                subfolder="scheduler")
    
    model = StableDiffusionPipeline.from_pretrained(model_name,
                                                    custom_pipeline="stable_diffusion_tensorrt_txt2img_ee",
                                                    revision=model_rev,
                                                    torch_dtype=torch.float16,
                                                    scheduler=scheduler)
    
    # re-use cached folder to save ONNX models and TensorRT Engines
    model.set_cached_folder(model_name, revision=model_rev)
    
    model = model.to("cuda")
    model.enable_model_cpu_offload()
    model.enable_attention_slicing(1)
    model.enable_xformers_memory_efficient_attention()

def inference(model_inputs:dict):
    global model

    prompt = model_inputs.get('prompt', None)
    height = model_inputs.get('height', 768)
    negative = model_inputs.get('negative_prompt', None)
    num_images_per_prompt = model_inputs.get('num_images_per_prompt', 1)
    width = model_inputs.get('width', 768)
    steps = model_inputs.get('steps', 20)
    guidance_scale = model_inputs.get('guidance_scale', 9)
    seed = model_inputs.get('seed', None)

    if not prompt: return {'message': 'No prompt was provided'}
    
    generator = None
    if seed: generator = torch.Generator("cuda").manual_seed(seed)
    
    #with autocast("cuda"):
    image = model(prompt, negative_prompt=negative, guidance_scale=guidance_scale, height=height, width=width, num_inference_steps=steps, generator=generator)
    
    buffered = BytesIO()
    image.images[0].save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    print(image)
    return {'image_base64': image_base64}


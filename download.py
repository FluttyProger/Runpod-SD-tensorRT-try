# In this file, we define download_model
# It runs during container build time to get model weights built into the container
import os
import torch
from torch import autocast
import base64
from io import BytesIO
from transformers import pipeline
from diffusers import DDIMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline

def download_model():
    model_name = os.getenv("MODEL_NAME")
    scheduler = DDIMScheduler.from_pretrained("stabilityai/stable-diffusion-2-1",
                                                subfolder="scheduler")

    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1",
                                                    custom_pipeline="stable_diffusion_tensorrt_txt2img",
                                                    revision='fp16',
                                                    torch_dtype=torch.float16,
                                                    scheduler=scheduler,)

    # re-use cached folder to save ONNX models and TensorRT Engines
    pipe.set_cached_folder("stabilityai/stable-diffusion-2-1", revision='fp16',)
    

if __name__ == "__main__":
    download_model()

# High cost - FP32

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

model_id = "dreambooth-concept"
device = "cuda"


pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
pipe = pipe.to(device)

sample_num = 4 # Number of generated images
lst = []
prompt = 'A 1950s Style Comic-Like Drawing of myprompt, grainy, realistic, hyperrealistic, very realistic, very very realistic, highly detailed, very detailed, extremely detailed, detailed, digital art, trending on artstation, detailed face, very detailed face, very detailed face, realism, HD Quality, 8k resolution, intricate details, body and head in frame, drawing, inked drawing, comic drawing, neat drawing, 1950s, 50s'
for i in range(sample_num):
    with autocast("cuda"):
        a = pipe(prompt, guidance_scale=10, height=512, width=512,
                 num_inference_steps=50, seed='random', scheduler='LMSDiscreteScheduler')["sample"][0]
        lst.append(a)
        a.save(f'outputs/gen-image-{i}.png')

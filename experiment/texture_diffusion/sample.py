import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ["http_proxy"] = "http://10.10.115.8:7890"
os.environ["https_proxy"] = "http://10.10.115.8:7890"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
import torch
import timm 
from PIL import Image
import  torchvision.transforms as transforms
from model import img_embedding
import  torch.nn as  nn 
import pickle
from train_utils import TrainingConfig,train_loop
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import StableDiffusionPipeline,DDIMScheduler,DDIMInverseScheduler,UNet2DConditionModel,AutoencoderKL
from transformers import CLIPTokenizer,CLIPTextModel,Dinov2Model,CLIPVisionModelWithProjection,CLIPImageProcessor
from datasets import load_dataset
from torch.utils.data import DataLoader
import wandb
import numpy as np
from collections import OrderedDict
from .custom_pipeline import StableDiffusionXLCustomPipeline

if __name__=="__main__":
    device=torch.device("cuda")
    image=Image.open("/home/x_lv/Dataset/output/generator_output/transformation_translation/0.png").convert("RGB")

    id="stabilityai/stable-diffusion-xl-base-1.0"    
      
    vae=AutoencoderKL.from_pretrained(id,subfolder='vae',torch_dtype=torch.float16).to(device)
    vae.requires_grad_(False)
    scheduler=DDIMScheduler.from_pretrained(id,subfolder="scheduler")
    scheduler.set_timesteps(50)
    unet=UNet2DConditionModel.from_pretrained(id,subfolder="unet",torch_dtype=torch.float16).to(device)
    unet.requires_grad_(False)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter", subfolder="models/image_encoder",torch_dtype=torch.float16).to(device)
    image_encoder.requires_grad_(False)
    img_emb_model=img_embedding( unet)  
    clip_image_processor=CLIPImageProcessor()

    pipeline=StableDiffusionXLCustomPipeline(vae,unet,scheduler,image_encoder,clip_image_processor)

    out_img=pipeline(image,device)
    
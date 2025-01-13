from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from diffusers import StableDiffusionXLPipeline
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import rescale_noise_cfg
from transformers import CLIPImageProcessor
from .model import img_embedding
import torchvision.transforms as transforms
import torchvision
from tqdm import tqdm
 
class StableDiffusionXLCustomPipeline(StableDiffusionXLPipeline):
     
    def __init__(
        self,
        vae,
        unet,
        scheduler,
        image_encoder,
        clip_image_processor:CLIPImageProcessor,
        image_embed_model:img_embedding
        
    ):
        super().__init__(vae,None,None,None,None,unet,scheduler,image_encoder)
        self.sd_transformation=transforms.Compose([
            transforms.Resize((512,512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),])
        self.clip_image_processor=clip_image_processor
        self.img_emb_model=image_embed_model


    @torch.no_grad()
    def __call__(  # noqa: C901
       self,image1,device,dtype,
       original_size=(512,512),
       target_size=(512,512),
        uncond_embeddings=None,
       crops_coords_top_left=(0.0,0.0),
       num_inference_steps=50,
       height=512,
       width=512,
       img_emb_model=None
    ):
        if img_emb_model is not None:
            self.img_emb_model=img_emb_model
        batch_size=1
        text_encoder_projection_dim=2048
        image_ti2=self.sd_transformation(image1).to(device,dtype=dtype).unsqueeze(0)
        
        image_addition=torch.FloatTensor(self.clip_image_processor(image1)["pixel_values"]).to(device,dtype=dtype)
         
        self.vae.to(torch.float32)
        timesteps = self.scheduler.timesteps

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
         
        '''add_text_embeds=uncond_embeddings
        add_text_embeds = add_text_embeds.to(device)'''
        
        latent_inv=torch.randn((batch_size,4,height//8,width//8)).to(device,dtype)
        with torch.no_grad():
                image_embeds=self.image_encoder(image_addition).image_embeds

        for t in tqdm(timesteps):
            added_cond_kwargs = {"time_ids": add_time_ids} #"text_embeds": add_text_embeds,
            predict = self.scheduler.scale_model_input(latent_inv, timestep=t)
            t=t.repeat(batch_size).to(device)
           
            noise_pred = self.img_emb_model(self.unet,image_ti2,predict, t, image_embeds).to(dtype)
            #ip_adapter(predict, t, img_embedding[0].to(weight_dtype), image_embeds)
            latent_inv = self.scheduler.step(noise_pred, t, latent_inv).prev_sample.to(dtype)

             

        out_img=self.decode_img(self.vae,latent_inv)

        return out_img
    
    def decode_img(self,vae,latents):
        # bath of latents -> list of images
        latents = (1 /  vae.config.scaling_factor) * latents
        with torch.no_grad():
            image = vae.decode(latents.to(torch.float32)).sample #.to(torch.float32)
        image =  (image / 2 + 0.5).clamp(0, 1)
        image = image.detach()
        image = image.cpu().squeeze(0)
        image= torchvision.transforms.functional.to_pil_image(image)
    
        pil_images = image
        return [pil_images]
    

    def _get_add_time_ids(
        self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
    ):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids
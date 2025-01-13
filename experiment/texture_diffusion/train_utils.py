import torch
from dataclasses import dataclass
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
import torchvision.transforms.functional
from tqdm.auto import tqdm
from pathlib import Path
import os
import torch.nn.functional as F
from PIL import Image
import wandb
import torchvision
from accelerate import DistributedDataParallelKwargs
import itertools
from skimage.metrics import structural_similarity as ssim
import numpy as np
import torch.nn as nn 
import random
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
 

@dataclass
class TrainingConfig:
    def __init__(self):
        self.image_size = 512  # the generated image resolution
        self.train_batch_size =8
        self.eval_batch_size = 1  # how many images to sample during evaluation
        self.num_epochs = 150
        self.gradient_accumulation_steps =1
        self.learning_rate = 1e-4
        self.lr_warmup_steps = 0
        self.save_image_epochs = 200
        self.save_image_steps = 2000
        self.save_model_epochs = 10000
        self.mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
        self.output_dir = "/home/x_lv/texture/experiment/texture_diffusion/out-diff"  # the model name locally and on the HF Hub
        
        
        self.overwrite_output_dir = True  # overwrite the old model when re-running the notebook
        self.seed = 0


def encode_img(vae,input_img,b_cat=False):
    # Single image -> single latent in a batch (so size 1, 4, 64, 64)
    type=input_img.dtype
    latents = vae.encode(input_img.to(torch.float32)).latent_dist.sample() #.to(torch.float32)
    latents = latents * vae.config.scaling_factor
    latents = latents.to(dtype=type)
    return latents
    '''if len(input_img.shape)<4:
        input_img = input_img.unsqueeze(0)
    with torch.no_grad():
        latent = vae.encode(input_img) # Note scaling
         
    return  latent.latent_dist.sample() if b_cat else 0.18215 * latent.latent_dist.sample()'''



def decode_img(vae,latents):
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


def evaluate(config, epoch,test_dataloader, unet, img_emb_model,vae, noise_scheduler,accelerator,weight_dtype,x,ip_adapter,image_encoder):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
        unet.eval()
        img_emb_model.eval()
        vae.eval()
        img_list=[]
        ssim_score=0
        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                '''add_time_ids = _get_add_time_ids(
                (512,512),
                (0.0,0.0),
                (512,512),
                dtype=weight_dtype,
                text_encoder_projection_dim=0,
        )
                added_cond_kwargs = {"time_ids": add_time_ids}'''

                input_img=batch["conditional_image"].to(unet.device,weight_dtype)
                label_img=batch["label_image"].to(unet.device,weight_dtype)
                cond_latents=batch["additional_image"].to(unet.device,weight_dtype)
                #type_cond=batch["text"]
                #cond_latents=encode_img(vae,cond_latents,True)

                #img_embedding=img_emb_model(cond_latents)
                
                 
                bs=cond_latents.shape[0]
                latent_inv=torch.randn((bs,4,config.image_size//8,config.image_size//8)).to(unet.device,weight_dtype)
                
                with torch.no_grad():
                        image_embeds = image_encoder(cond_latents).image_embeds#output_hidden_states=True).hidden_states[-2]'''
                        
                for t in tqdm(noise_scheduler.timesteps):
                    #predict =  torch.cat((latent_inv,cond_latents),dim=1)
                    predict = noise_scheduler.scale_model_input(latent_inv, timestep=t)
                    t=t.repeat(bs).to(unet.device)

                    
                    '''noise = unet(predict, t,encoder_hidden_states=img_embedding[0].to(weight_dtype),down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in img_embedding[1]
                    ]).sample''' 

                    
                    noise_pred = img_emb_model(unet,input_img,predict, t, image_embeds).to(weight_dtype)
                    #ip_adapter(predict, t, img_embedding[0].to(weight_dtype), image_embeds)

                    latent_inv = noise_scheduler.step(noise_pred, t, latent_inv).prev_sample.to(weight_dtype)
                
                out_img=decode_img(vae,latent_inv.to(weight_dtype))
                
                for ori,pred,label in zip(input_img,out_img,label_img):
                    img_list.append(wandb.Image(ori,caption="ori image"))
                    img_list.append(wandb.Image(pred,caption="pred image"))
                    img_list.append(wandb.Image(label,caption="label image"))
                    x1=np.array(pred)
                    y2=np.array(torchvision.transforms.functional.to_pil_image(label))
                    ssim_score=ssim(x1 ,y2,data_range=255,channel_axis=-1)
            accelerator.log({"test":img_list})

 
            
        return ssim_score 

def pyramid_noise_like(x, discount=0.9):
  b, c, w, h = x.shape # EDIT: w and h get over-written, rename for a different variant!
  u = nn.Upsample(size=(w, h), mode='bilinear')
  noise = torch.randn_like(x)
  for i in range(10):
    r = random.random()*2+2 # Rather than always going 2x, 
    w, h = max(1, int(w/(r**i))), max(1, int(h/(r**i)))
    noise += u(torch.randn(b, c, w, h).to(x)) * discount**i
    if w==1 or h==1: break # Lowest resolution is 1x1
  return noise/noise.std() # Scaled back to roughly unit variance


def train_loop(config, unet, img_emb_model,vae, noise_scheduler, optimizer, train_dataloader,test_loader, lr_scheduler,x,ip_adapter,image_encoder,node_type):
    
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="wandb",
        project_dir=os.path.join(config.output_dir, "logs"),
        #mixed_precision="fp16",
        kwargs_handlers=[ddp_kwargs]
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)

        
        wandb.init(project="texture_exp",config={"node_type":node_type,"base_model":"sdxl","adapter":"ti2","addition":config.__dict__,"description":"add faceadapter with ti2 sample + sdxl sample + pyramid noise 0.5"})
         
        accelerator.init_trackers("train_example")

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
         
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
         
    

    # Move text_encode and vae to gpu and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=torch.float32)   #dtype=torch.float32  sdxl
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    x=torch.randn((1,2048)).to(accelerator.device)
 
    #img_emb_model.to(accelerator.device,dtype=weight_dtype)
    total=len(train_dataloader)

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    optimizer, train_dataloader,test_loader, lr_scheduler,img_emb_model = accelerator.prepare(
        optimizer, train_dataloader, test_loader,lr_scheduler,img_emb_model
    )

  

    global_step = 0
    ssim_score=0
    ssim_all=0.0
    evaluate_step=0
    # Now you train the model
    
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=total, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            
            with accelerator.accumulate(img_emb_model):
                fisrt_images = batch["additional_image"].to(weight_dtype)
                cond_latents=batch["conditional_image"].to(weight_dtype)
                cond_images=batch["label_image"].to(weight_dtype)
                label=batch["label"]
                #type_cond=batch["text"]
                latents=encode_img(vae,cond_images,False).to(weight_dtype)
                #cond_latents=encode_img(vae,cond_latents,True)

                #img_emb=img_emb_model(fisrt_images).to(unet.device,weight_dtype)
                # Sample noise to add to the images
                '''noise = torch.randn_like(latents, device=latents.device)
                noise += 0.05 * torch.randn((latents.shape[0], latents.shape[1], 1, 1)).to(accelerator.device, dtype=weight_dtype)'''
                noise=pyramid_noise_like(latents,0.5).to(accelerator.device, dtype=weight_dtype)
                #noise = torch.randn_like(latents, device=accelerator.device)
                '''x3=decode_img(vae,noise)
                x3[0].save("/home/x_lv/texture/experiment/texture_diffusion/3.png")'''
                bs = latents.shape[0]
                '''add_time_ids = _get_add_time_ids(
                (512,512),
                (0.0,0.0),
                (512,512),
                dtype=weight_dtype,
                text_encoder_projection_dim=0,
        )
                added_cond_kwargs = {"time_ids": add_time_ids}'''
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bs,), device=latents.device,
                    dtype=torch.int64
                )
                timesteps = torch.rand((bs, ), device=latents.device)
                timesteps = (1 - timesteps**3) * (noise_scheduler.config.num_train_timesteps-1)
                timesteps = timesteps.long()
                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_images = noise_scheduler.add_noise(latents, noise, timesteps) 
                #noisy_images=torch.cat((noisy_images,cond_latents),dim=1)
                
                #img_embedding=img_emb_model(fisrt_images)
                '''if bs >1:
                    x1=x.unsqueeze(0).repeat(bs,1,1).to(weight_dtype)
                else:
                    x1=x.unsqueeze(0).to(weight_dtype)'''
                # Predict the noise residual
                '''noise_pred = unet(noisy_images, timesteps, encoder_hidden_states=img_embedding[0].to(weight_dtype),down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in img_embedding[1]
                    ] ).sample '''
                with torch.no_grad():
                    image_embeds = image_encoder(fisrt_images).image_embeds#output_hidden_states=True).hidden_states[-2]'''
                noise_pred =img_emb_model(unet,cond_latents,noisy_images, timesteps,image_embeds).to(weight_dtype)
                #ip_adapter(noisy_images, timesteps, img_embedding[0].to(weight_dtype), image_embeds).to(weight_dtype)

                loss = F.mse_loss(noise_pred.float(), noise.float())
                accelerator.backward(loss)

                '''img_emb_model.ip_adapter.image_proj_model.to(torch.float32)
                img_emb_model.ip_adapter.adapter_modules.to(torch.float32)
                 
                params_to_opt = itertools.chain(img_emb_model.ip_adapter.image_proj_model.parameters(),  img_emb_model.ip_adapter.adapter_modules.parameters())'''
                accelerator.clip_grad_norm_(img_emb_model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                '''img_emb_model.ip_adapter.image_proj_model.to(torch.float16)
                img_emb_model.ip_adapter.adapter_modules.to(torch.float16)'''
                 

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step,"epoch":epoch}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1
            
        # After each epoch you optionally sample some demo images with evaluate() and save the model
            if accelerator.is_main_process:
                evaluate_step+=1

                if (global_step + 1) % config.save_image_steps == 0 :#or epoch == config.num_epochs - 1:
                    
                    ssim_=evaluate(config, epoch,test_loader,unet,img_emb_model,vae,noise_scheduler,accelerator,weight_dtype,None,None,image_encoder )

                    print(f"ssim_score:{ssim_:.2f}_steps:{global_step}")
                    #if ssim_>ssim_score: 
                    inter_model=None
                    inter_model=accelerator.unwrap_model(img_emb_model).state_dict()
                    save_dict={}
                    for k,v in inter_model.items():
                        if "t2i_adapters" in k or "img2token" in k:#"image_proj_model" in k or "adapter_modules" in k:
                            save_dict.update({k:v})
                        
                    ssim_all+=ssim_
                    ckpt_path=Path(f'/home/x_lv/texture/experiment/texture_diffusion/ckpt/img_emb_model/{label[0]}')
                    
                    if not ckpt_path.exists():
                        ckpt_path.mkdir()
                    
                    torch.save(save_dict,f"{ckpt_path}/_ssim_avg:{(ssim_all/evaluate_step):.2f}_steps:{global_step}.pth")
                    
                    ssim_score=ssim_



'''    def  metric(img1,img2):
        img1 = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2, cv2.IMREAD_GRAYSCALE)

        # 初始化 SIFT 检测器
        sift = cv2.SIFT_create()

        # 检测特征点和计算描述符
        keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
        if descriptors1 is not None and descriptors2 is not None:
        # 使用 BFMatcher 进行特征点匹配
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches = bf.match(descriptors1, descriptors2)

            # 计算匹配距离的平均值作为相似性分数
            match_distances = [m.distance for m in matches]
            similarity_score = sum(match_distances) / len(match_distances)

            print(f"Similarity score (SIFT + BFMatcher): {similarity_score}")
        else:
            x1=np.array(pred)
            y2=np.array(torchvision.transforms.functional.to_pil_image(label))
            ssim_score=ssim(x1 ,y2,data_range=255,channel_axis=-1)'''


def _get_add_time_ids(
        original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
    ):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

 
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids
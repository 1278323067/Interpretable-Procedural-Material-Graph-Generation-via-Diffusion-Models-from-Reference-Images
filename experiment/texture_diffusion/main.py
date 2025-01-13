import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1,3"
os.environ["http_proxy"] = "http://10.10.115.11:7897"
os.environ["https_proxy"] = "http://10.10.115.11:7897"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
import torch
import timm 
from PIL import Image
import  torchvision.transforms as transforms
from model import img_embedding,IPAdapter,ImageProjModel
import  torch.nn as  nn 
import pickle
from train_utils import TrainingConfig,train_loop
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import StableDiffusionPipeline,DDIMScheduler,DDIMInverseScheduler,UNet2DConditionModel,AutoencoderKL
from transformers import CLIPTokenizer,CLIPTextModel,Dinov2Model,CLIPImageProcessor
from datasets import load_dataset
from torch.utils.data import DataLoader
import wandb
import numpy as np
import sys
from ipadapter.resampler import Resampler
from ipadapter.attention_processor import IPAttnProcessor2_0,AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection,CLIPVisionModel
import itertools
config = TrainingConfig()
import random
#local_rank=int(os.environ["LOCAL_RANK"])
def set_seed(seed):
    # Python 内置的随机数生成器
    random.seed(seed)
    
    # NumPy 的随机数生成器
    np.random.seed(seed)
    
    # PyTorch 的随机数生成器
    torch.manual_seed(seed)
    
    # 如果你使用的是多个 GPU
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)
set_seed(0)

if __name__=="__main__":
    #wandb.init(project="dd")
    #pretrained swinv2
    
    #feature_model=Dinov2Model.from_pretrained("facebook/dinov2-large")

    #feature_model=timm.create_model("swinv2_base_window12to24_192to384.ms_in22k_ft_in1k") 
    '''in_features=feature_model.head.fc.in_features
    feature_model.head.fc=nn.Linear(in_features,35,bias=True) 
    feature_model.to(device)
    from collections import OrderedDict
    state_dict=torch.load("/home/x_lv/texture/experiment/swinv2_base_window12to24_192to384.ms_in22k_ft_in1k_type:0_False_epoch:25_88.99617416651485.pth")["net"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    feature_model.load_state_dict(new_state_dict)'''
    #feature_model.requires_grad_(False)

    #custom model
  
    '''for i in img_emb_model.parameters():
        i.requires_grad=False'''
    #test image
    '''img=Image.open("/home/x_lv/Dataset/output/generator_valid/mirror.png").convert("RGB")
    transform_cfg=timm.data.resolve_data_config(feature_model.pretrained_cfg)
    mean,std,size=transform_cfg["mean"],transform_cfg["std"],transform_cfg["input_size"][-1]
    _transforms=transforms.Compose([transforms.Resize([size,size]),transforms.ToTensor(),transforms.Normalize(mean,std)])
    img=_transforms(img).to(device)

    image_emb=model(img.unsqueeze(0))
    print(image_emb.shape)'''


    #stable diffusion
   
    id="stabilityai/stable-diffusion-xl-base-1.0"  #"stabilityai/stable-diffusion-xl-base-1.0"  
      
    vae=AutoencoderKL.from_pretrained(id,subfolder='vae',torch_dtype=torch.float16)
    vae.requires_grad_(False)
    scheduler=DDIMScheduler.from_pretrained(id,subfolder="scheduler")
    scheduler.set_timesteps(50)

    unet=UNet2DConditionModel.from_pretrained(id,subfolder="unet",torch_dtype=torch.float16)
    unet.requires_grad_(False)

    '''tokenizer = CLIPTokenizer.from_pretrained(id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(id, subfolder="text_encoder")'''
    image_encoder = CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter", subfolder="models/image_encoder",torch_dtype=torch.float16)
    #text_encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)
    '''uent_config = UNet2DConditionModel.load_config(id, subfolder="unet")
    uent_config["in_channels"]=8
    
    unet = UNet2DConditionModel.from_config(uent_config)'''
    #conv_in=unet.conv_in
    #unet.conv_in=nn.Conv2d(conv_in.in_channels*2,conv_in.out_channels,conv_in.kernel_size,conv_in.stride,conv_in.padding)
    '''for i in unet.parameters():
        i.requires_grad=False'''

    '''image_proj_model = ImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim=image_encoder.config.projection_dim,
        clip_extra_context_tokens=4,
    )'''
  
    # init adapter modules
    attn_procs = {}
    '''unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor2_0()
        else:
            layer_name = name.split(".processor")[0]
             
            weights = {
                f"to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                f"to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = IPAttnProcessor2_0(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
            attn_procs[name].load_state_dict(weights)
    unet.set_attn_processor(attn_procs)
   
 
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())'''
    
   
    img_emb_model=img_embedding( )  #image_proj_model, adapter_modules
    '''z=torch.load("/home/x_lv/texture/experiment/texture_diffusion/ckpt/img_emb_model/inpaint:input0/_ssim_avg:0.00_steps:14999.pth")
    img_emb_model.load_state_dict(z)'''
    #train
    sd_trans=transforms.Compose([
            transforms.Resize((512,512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),])
    
    ''' transform_cfg=timm.data.resolve_data_config(feature_model.pretrained_cfg)
    mean,std,size=transform_cfg["mean"],transform_cfg["std"],transform_cfg["input_size"][-1]
    cond_trans=transforms.Compose([transforms.Resize([size,size]),transforms.ToTensor(),transforms.Normalize(mean,std)])'''#transforms.Compose([transforms.Resize([224,224]),transforms.ToTensor(),transforms.Normalize(( 0.485,0.456,0.406),( 0.229,0.224,0.225))])
    #transforms.Compose([transforms.Resize([size,size]),transforms.ToTensor(),transforms.Normalize(mean,std)])
    clip_image_processor=CLIPImageProcessor()
    def sd_transforms(examples):
        #del examples["text"]
        for k,v in examples.items():
            if k != "label":
                for i,z in enumerate(v):
                    if z.mode=="I;16" or z.mode == "I":
                        image=np.array(z)
                        scaled_tensor = image/65535
                        x=(scaled_tensor*255.0).astype(np.uint8)
                        v[i]= Image.fromarray(x, mode='L').convert("RGB")
                    else:
                        v[i] =z.convert("RGB")

            if k=="additional_image":
                examples.update({k:clip_image_processor(v)["pixel_values"]})

                #examples.update({k:torch.stack([cond_trans(i)  for i in v])})
                #examples.update({"conditioning_latent":sd_trans(v[0]).unsqueeze(0)})
                #examples.update({"conditioning_image":sd_trans(v[0]).unsqueeze(0)})
            elif k=="conditional_image":
                examples.update({k:torch.stack([sd_trans(i)  for i in v])})   #clip_image_processor(v)["pixel_values"]
                '''elif k=="text":
                examples.update({"text":v})'''
            elif k=="label_image":
                examples.update({k:torch.stack([sd_trans(i)  for i in v])})
        return examples

    data_path="/home/x_lv/.cache/huggingface/datasets/distance_2/generator/default-3afc4de49bce0eed"    
    dataset=load_dataset(data_path,split="train")
    
    dataset_type=list(set(dataset["label"]))
    dataset_index=0
    if len(dataset_type) >1:
        zzz=['blend_blending_mode_max:input0', 'blend_blending_mode_max:input1', 'blend_blending_mode_subtract:input0', 'blend_blending_mode_add:input1', 'blend_blending_mode_multiply:input0', 'blend_blending_mode_multiply:input1', 'blend_blending_mode_subtract:input1', 'blend_blending_mode_add:input0']
    
        print(dataset_type)
        dataset=dataset.filter(lambda x : "distance:input0" in x["label"] )
    node_type=dataset[0]["label"]
    dataset.set_transform(sd_transforms)
    dataset=dataset.train_test_split(test_size=0.004,shuffle=True,generator=np.random.default_rng(seed=1))

    train_dataset=dataset["train"]
    test_dataset=dataset ["test"]
    train_dataloader=DataLoader(train_dataset,batch_size=config.train_batch_size)
    test_dataloader=DataLoader(test_dataset,batch_size=config.eval_batch_size)
   
    params_to_opt = itertools.chain(img_emb_model.t2i_adapters.parameters(),img_emb_model.img2token.parameters()) #img_emb_model.ip_adapter.image_proj_model.parameters(),  img_emb_model.ip_adapter.adapter_modules.parameters()
    optimizer = torch.optim.AdamW(params_to_opt, lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)
    #x=torch.randn((1,1024)).to(device)
    
    train_loop(config=config,unet=unet,img_emb_model=img_emb_model,vae=vae,noise_scheduler=scheduler,optimizer=optimizer,train_dataloader=train_dataloader,test_loader=test_dataloader,lr_scheduler=lr_scheduler,x=None,ip_adapter=None,image_encoder=image_encoder,node_type=node_type)

    '''feature_model.eval()
    logits=feature_model(img.unsqueeze(0))
    _,preds=torch.max(logits,1)
    print(preds)
    with open("/home/x_lv/texture/RADAM/RADAM/lable_pair.pkl","rb") as f:
            lable_pair=pickle.load(f)
            print(lable_pair)'''


 
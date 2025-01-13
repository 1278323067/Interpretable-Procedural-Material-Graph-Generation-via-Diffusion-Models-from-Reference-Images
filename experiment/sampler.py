
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
os.environ["http_proxy"] = "http://10.10.115.13:7897"
os.environ["https_proxy"] = "http://10.10.115.13:7897"
import timm 
import argparse 
import pickle
import torch.nn as nn
import torch
from pathlib import Path
from PIL import  Image
import torchvision.transforms as transforms
from texture_diffusion.model import img_embedding
from diffusers import UNet2DConditionModel,AutoencoderKL,DDIMScheduler,StableDiffusionXLInpaintPipeline
from transformers import CLIPVisionModelWithProjection,CLIPImageProcessor
from texture_diffusion.custom_pipeline import StableDiffusionXLCustomPipeline
from safetensors.torch import load_file
import subprocess
import numpy as np 
import json 
import queue
import yaml
import io
import requests
import sys
sys.path.append("/home/x_lv/texture/diffmat")
from  diffmat.translator.util import  (NODE_CATEGORY_LUT,CONFIG_DIR)
from diffmat.core.material import functional as F, noise as N
import  torchvision.transforms as transform
Dir="/home/x_lv/texture/" 

class Node():
    def __init__(self,image,utils,node_type_=None,is_refiner=True) -> None:
        self.image=output_refiner(image.convert("RGB")) if is_refiner else image.convert("RGB")
        self.node_type_with_param=node_type_
        self.node_type=self.get_node_type(utils)  if node_type_ is None else node_type_
         
        self.node_params=None
        self.parents_index=None
        self.param_idx=None
        self.list_index=None
        self.child_list=self._resolve_node_type(self.node_type)

    def update_node_type(self,node_type):
        self.node_type = node_type
        self.node_type_with_param=node_type
        self.child_list = self._resolve_node_type(self.node_type)

    def _resolve_node_type(self, node_type: str):
        if node_type == "bitmap" :
            return
        elif "blend" in node_type:
            node_type="blend"
        elif "edge_detect" in node_type:
            node_type="edge_detect"
        elif "slope_blur" in node_type :
            node_type  = "slope_blur"
        elif "splatter_filter" in node_type:
            node_type = "splatter"
        if node_type not in NODE_CATEGORY_LUT :
            raise NotImplementedError(f'Unsupported node type: {node_type}')
             
        else:
            node_category = NODE_CATEGORY_LUT.get(node_type, 'generator')
        is_generator = node_category in ('dual', 'generator')

        load_config_mode = 'generator' if is_generator else 'node'
        dir_name = f'{load_config_mode}s' if load_config_mode in ('node', 'function', 'generator') else load_config_mode
        node_config_path = Path(CONFIG_DIR / dir_name / f'{node_type}.yml')
        if not node_config_path.exists():
            raise FileNotFoundError(f'Configuration file not found for {load_config_mode} type: '
                                    f'{node_type}')
        with open(node_config_path,'r') as f:
            xml_file=yaml.safe_load(f)
        if xml_file["input"] is not None:
            img_list=[None for _ in xml_file['input']]
        else:
            img_list=None
        return  img_list
     

    def set_params(self,params):
        self.node_params=params

    def set_child(self,idx,child_node):
        self.child_list[idx]=child_node.list_index
        child_node.set_param_idx(idx)

    def set_parents_index(self,index):
        self.parents_index=index

    def set_param_idx(self,idx):
        self.param_idx=idx

    def get_node_type(self,utils):
        image_=utils.transforms(self.image)
        node_type=self.cls(args,utils.model_classifier,utils.node_dict,image_)
        return node_type
    
    def set_list_index(self,index):
        self.list_index=index

    def cls(self,args,cls_model,node_dict,images):
        images=images.unsqueeze(0).to("cuda")
        logits=cls_model(images)
        #_,preds=torch.max(logits,1)
        _, pred_k = logits.topk(3, 1, True, True)
        node_type=[k for i in pred_k.tolist()[0] for k,v in node_dict.items() if i==v] # [k for k,v in node_dict.items() if v==pred_k.item()]
        print(node_type)
        self.list_node_type = node_type
        '''if node_type[0]=="blend_mask_one_noise_blending_mode_copy":
         
            self.node_type_with_param=node_type[1]
        else:'''
        self.node_type_with_param=node_type[0]
        if any( i in self.node_type_with_param for i in ["blend_","edge_detect"]):
            return self.node_type_with_param
        elif "slope_blur" in self.node_type_with_param:
            return "slope_blur"
        else :
            path="/home/x_lv/texture/diffmat/diffmat/config/node_list.yml"
            with open(path,"r") as f:
                content=yaml.safe_load(f)
             
            node_type=[i  for k,v in content.items() for i in v if (   self.node_type_with_param.startswith(i) and not isinstance(v,str))]
            node_type=max(node_type,key=len)
                
        return node_type

def output_refiner(image):
    image_=image

    image_np = np.array(image)
    for i in range(image_np.shape[0]):
        for j in range(image_np.shape[1]):
            image_np[i,j,:]= [max(image_np[i,j,:]),max(image_np[i,j,:]),max(image_np[i,j,:])] if  (max(image_np[i,j,:])-min(image_np[i,j,:])) <5 else image_np[i,j,:]
     # 获取 R, G, B 通道
    r, g, b = image_np[:,:,0], image_np[:,:,1], image_np[:,:,2]
 
    mask = (r <= 10) & (g <= 10) & (b <= 10)

    image_np[mask] = [0, 0, 0]

    mask_ = (r >= 250) & (g >= 250) & (b >= 250)

    image_np[mask_] = [255,255,255]

    # 将 NumPy 数组转换回图片
    new_image = Image.fromarray(image_np.astype("uint8"))
    '''if node.parents_index is not None: 
        parent_node_type=node_list[node.parents_index].node_type
        if "blend_mask_two_noise" in parent_node_type:
            pass'''
    img_tensor=transform.PILToTensor()(new_image.convert("L")).unsqueeze(0)
    image_1=F.auto_levels(img_tensor/255)
    new_image=transform.ToPILImage()(image_1.squeeze(0)).convert("RGB")
        
    return new_image

def rgb_to_sorted_grayscale(rgb_image):
     
    rgb_image = rgb_image
    pred_image=rgb_image.convert("L")
  
    #pred_image=pred_image.convert("L")
    x=np.array(pred_image)
    print(x.max())
     
    width, height = rgb_image.size
    pixels = []
    zz=set()
    for x in range(width):
        for y in range(height):
            gray_value = pred_image.getpixel((x, y))
            rgb_value = rgb_image.getpixel((x, y))
            if gray_value not in zz:
                pixels.append((gray_value, rgb_value))
                zz.add(gray_value)

    sorted_pixels = sorted(pixels, key=lambda p: p[0])

    sorted_image = Image.new('RGB', (256, 256))
    for y in range(256):
        for i in range(len(sorted_pixels)):
            sorted_image.putpixel((sorted_pixels[i][0],y),sorted_pixels[i][1])
    sorted_image=sorted_image.resize((512,512),Image.NEAREST)
    return sorted_image


def mask_seg(mask):
    img=mask
    width, height = img.size
    sub_width = width // 2
    sub_height = height // 2
    sort_list=[]
    img_=[]
    for i in range(20):
        x1=np.random.randint(0,256)
        y1=np.random.randint(0,256)
        img_.append(img.crop((x1,y1,x1+sub_width,y1+sub_height)))
    for i in range(20):
        zero_pixels_count=0    
        for y in range(sub_height):
            for x in range(sub_width):
                pixel_value =sum( img_[i].getpixel((y, x)) )
                if pixel_value <10:
                    zero_pixels_count += 1
        sort_list.append((zero_pixels_count,img_[i]))
    out_list=sorted(sort_list,key=lambda x:x[0])
    for i in range(5):
        sub_img=out_list[i][1]
        sub_img=sub_img.resize((512,512),resample=Image.Resampling.BILINEAR)
        sub_img.save("/home/x_lv/texture/experiment/mask_emb/"+str(i)+".png")
        
def blend_mask_copy_legacy(init_image,mask_image):
    init_image=init_image.convert("RGB")
    mask_image=mask_image.convert("L")
   
    init_=np.array(init_image)
    mask_image_=np.array(mask_image) 
    mask_image_=np.expand_dims(mask_image_, axis=-1)
    new_image=(init_*(1-mask_image_/255))
    new_image=Image.fromarray(np.uint8(new_image))
    mask_seg(new_image)
    '''new_image=np.array(new_image)
    colored_value=np.sum(new_image,axis=(0,1))
    color_mask=0
    for x in range(new_image.shape[0]):
        for y in range(new_image.shape[1]):
            color_mask += 1 if sum(new_image[x,y,:]) >100 else 0
    avg_value=[c/color_mask  for i,c in enumerate(colored_value) ]
    new_image_=new_image.copy()
    for x in range(new_image.shape[0]):
        for y in range(new_image.shape[1]):
            
                new_image_[x,y,:] = avg_value if sum(new_image[x,y,:]) <100  else new_image[x,y,:]
    new_image=Image.fromarray(np.uint8(new_image_))'''
    '''image=Image.open("/home/x_lv/texture/experiment/mask_emb/0.png")
    new_image= Image.composite (image,new_image,mask_image)
    new_image.save('test.png')'''
     
    commands=['python','-u',"/home/x_lv/texture/diffusers/examples/textual_inversion/textual_inversion_sdxl.py",
              "--pretrained_model_name_or_path","stabilityai/stable-diffusion-xl-base-1.0",
              "--train_data_dir","/home/x_lv/texture/experiment/mask_emb",
              "--learnable_property","object" ,
              "--placeholder_token","lxy" ,
                "--initializer_token","pattern" ,
                '--save_steps',"500",
                "--resolution","512" ,
                "--train_batch_size","1" ,
                "--gradient_accumulation_steps","1" ,
                "--max_train_steps","500" ,
                "--learning_rate","5.0e-04" ,
                "--lr_scheduler","constant" ,
                "--lr_warmup_steps","0" ,
                "--output_dir","/home/x_lv/texture/experiment/textual_inversion_img_emb" ,
                "--report_to","wandb","--checkpointing_steps","1000"
                ]
    process = subprocess.Popen(commands,shell=False,bufsize=1 ,text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    while process.poll() is None:
            line = process.stderr.readline()
            process.stderr.flush() # 刷新缓存，防止缓存过多造成卡死
            #line = line.decode("utf8") 
            print(line)
        
    error=process.wait()
    print(f"subprocess exit: {error}")
    '''stdout, stderr = process.communicate()
    
    # 打印输出和错误（如果有的话）
    print(stdout.decode())
    print(stderr.decode())'''

    embed = load_file("/home/x_lv/texture/experiment/textual_inversion_img_emb/learned_embeds.safetensors")
    embeds_2=load_file("/home/x_lv/texture/experiment/textual_inversion_img_emb/learned_embeds_2.safetensors")
    
    pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16
    ).to("cuda")

    '''pipeline.load_textual_inversion(embeds_2, token="lxy", text_encoder=pipeline.text_encoder_2, tokenizer=pipeline.tokenizer_2)
    pipeline.load_textual_inversion(embed, token="lxy", text_encoder=pipeline.text_encoder, tokenizer=pipeline.tokenizer)'''
     
    new_image.save("1.png")
    mask_image.save("2.png")
     
    inpainted_image = pipeline( prompt="lxy",image=new_image, mask_image=mask_image,guidance_scale=8,strength=0.99,num_inference_steps=20,target_size=(512,512))[0]
    
    return inpainted_image[0]

def blend_mask_copy(init_image,mask_image):
    init_image=init_image.convert("RGB")
    mask_image=mask_image.convert("L")

    init_byte_arr = io.BytesIO()
    init_image.save(init_byte_arr, format='png')
    init_value = init_byte_arr.getvalue()

    mask_byte_arr = io.BytesIO()
    mask_image.save(mask_byte_arr, format='png')
    mask_value = mask_byte_arr.getvalue()

    r = requests.post('https://clipdrop-api.co/cleanup/v1',
    files = {
        'image_file': ('image.jpg', init_value, 'image/jpeg'),
        'mask_file': ('mask.png', mask_value, 'image/png'),
        #'mode': ("mode","quality","text/plain")
        },
    data = { "mode": "quality" },
    headers = { 'x-api-key': '4eecc429d6e3f1ee5a96943d071cccb2c372592e24251d7cf1a2960bcf822a118691f165f9c2ee98e934265fbcf7099d'}

    )
    if (r.ok):
    # r.content contains the bytes of the returned image
        pass
    else:
        r.raise_for_status()
  
    image_stream = io.BytesIO(r.content)
    result=Image.open(image_stream)
    return result

def mask_proportion(mask):
    img=np.array(mask)[0]
    high_pro= sum(1 for i in range(img.shape[1]) for j in range(img.shape[2]) if img[0][i][j] > (0.5) )/512**2
    threshold=(high_pro)*0.7+(1.0-high_pro)*0.3
    #(high_pro-0.1)/(0.9-0.1)*0.7+(0.9-high_pro)/(0.9-0.1)*0.3
    print(threshold)
    return threshold

def image_inqueue(node_queue,node_list,node,node_parent=None):
    if not any(item in node.node_type for item in ["shape","bitmap","clouds_2","bnw_spots_3","brick_generator"]):
        node_queue.put(node)
    node.set_list_index(len(node_list))
    node_list.append(node)  
    if node_parent is not None:
        node.set_parents_index(node_parent.list_index)
    
    node.image.save(Dir+"experiment/output/"+f"{len(node_list)-1}_{str(node.node_type_with_param)}_{node.parents_index}_{node.param_idx}.png")

def node_process(args,utils,pipe,node_queue,node_list):
    
    node=node_queue.get()
    # without image input
    
  

    if ("blend_mask_two_noise" not in node.node_type   and node.parents_index is None  and "normal" not in node.node_type ) : #or(node.parents_index is not None and "blend_mask_two_noise" in node_list[node.parents_index].node_type and node.list_index in [2,3] ):
        node.update_node_type("dyngradient")
        image_gray=node.image.convert("L")
 
        node_1=Node(image_gray,utils,is_refiner=False)
        node.set_child(0,node_1)
        image_inqueue(node_queue,node_list,node_1,node)
        image_lut=rgb_to_sorted_grayscale(node.image)
        node_2=Node(image_lut,utils,"bitmap",is_refiner=False)
        node.set_child(1,node_2)
        image_inqueue(node_queue,node_list,node_2,node)
        
    else:
        if node.parents_index is not None:
            if  "blend_mask_two_noise" in node_list[node.parents_index].node_type and node.node_type  == 'blend_mask_two_noise_blending_mode_copy' :
                node.update_node_type(node.list_node_type[1] if "slope_blur" not in node.list_node_type[1] else "slope_blur")
        '''if "blend_mask_two_noise" in node_list[node.parents_index].node_type and node.node_type  == 'slope_blur' :
            new_image=node.image
            img_tensor=transform.PILToTensor()(new_image.convert("L")).unsqueeze(0)
            image_1=F.auto_levels(img_tensor/255)
            new_image=transform.ToPILImage()(image_1.squeeze(0)).convert("RGB")
            new_image.save("/home/x_lv/texture/experiment/output/test.png")
            node.image=new_image'''

        ada_type=[file for file in utils.ada_list if  file.name.startswith(node.node_type)]
        if node.node_type == "splatter" or node.node_type == "transformation":
            ada_type=[ada_type[1]]

        if len(ada_type) > 0:

            ada_path = [Path(args.adapter_path)/ada_ for ada_ in ada_type]
            for input_idx,ada in enumerate(ada_path):        
                ada_model=list(ada.rglob("*"))[0]
                model_=torch.load(ada_model)
                img_emb_model.load_state_dict(model_)
                image_output = pipe(node.image,device,dtype,img_emb_model=img_emb_model)[0]
                node_=Node(image_output,utils)
                
                

                if "blend_mask_two_noise" in node.node_type:
                    node.set_child(2,node_)
                    image_inqueue(node_queue,node_list,node_,node)

                    img_tensor=transform.PILToTensor()(image_output.convert("L")).unsqueeze(0)
                    
                    image_1=F.blur_hq(img_tensor/255,intensity=2.0)
                    threshold=mask_proportion(image_1)
                    image_1=transform.ToPILImage()(image_1.squeeze(0))
                    image_1=image_1.point(lambda x : 0 if x < 20 else 255)
                    #
                    image_1.save(Dir+"experiment/output/1_mask.png")
                    
                    inpainted_image=blend_mask_copy(node.image,image_1)

                    node_1=Node(inpainted_image,utils,is_refiner=False)
                    node.set_child(0,node_1)
                    image_inqueue(node_queue,node_list,node_1,node)
                    

                    reverse_mask=image_output.point(lambda x :  int((1-x/255)*255) )
                    img_tensor=transform.PILToTensor()(reverse_mask.convert("L")).unsqueeze(0)
                    image_2=F.blur_hq(img_tensor/255,intensity=2.0)
                    threshold=mask_proportion(image_2)
                    image_2=transform.ToPILImage()(image_2.squeeze(0))
                    
                    image_2=image_2.point(lambda x : 0 if x<20 else 255)
                    image_2.save(Dir+"experiment/output/2_mask.png")
                    inpainted_image_=blend_mask_copy(node.image,image_2)
                    
                    node_2=Node(inpainted_image_,utils,is_refiner=False)
                    node.set_child(1,node_2)
                    image_inqueue(node_queue,node_list,node_2,node)
                    print("----------------------------------------------")
                     

                else:
                    node.set_child(input_idx,node_)
                    image_inqueue(node_queue,node_list,node_,node)
                    

        else:
            if node.node_type == "dyngradient":
                image_gray=node.image.convert("L")
                node_1=Node(image_gray,utils,is_refiner=False)
                node.set_child(0,node_1)
                image_inqueue(node_queue,node_list,node_1,node)

                image_lut=rgb_to_sorted_grayscale(node.image)
                node_2=Node(image_lut,utils,"bitmap",is_refiner=False)
                node.set_child(1,node_2)
                image_inqueue(node_queue,node_list,node_2,node)
            elif node.node_type == "invert":
                image_array=np.array(node.image)/255
                image_array = (np.clip(1-image_array ,0,1)*255).astype("uint8")
                image = Image.fromarray(image_array)
                node_ = Node(image,utils)
                node.set_child(0,node_)
                image_inqueue(node_queue,node_list,node_,node)
            elif node.node_type == "cartesian_to_polar" :
                img_tensor=transform.PILToTensor()(node.image.convert("L")).unsqueeze(0)
                    
                image=F.pol2car(img_tensor/255)
                image=transform.ToPILImage()(image.squeeze(0))
                node_ = Node(image,utils)
                node.set_child(0,node_)
                image_inqueue(node_queue,node_list,node_,node)
            elif node.node_type == "polar_to_cartesian" :
                img_tensor=transform.PILToTensor()(image_output.convert("L")).unsqueeze(0)
                    
                image=F.car2pol(img_tensor/255)
                image=transform.ToPILImage()(image.squeeze(0))
                node_ = Node(image,utils)
                node.set_child(0,node_)
                image_inqueue(node_queue,node_list,node_,node)

    return
 
if __name__== "__main__":
    device="cuda"
    dtype=torch.float16
    parse=argparse.ArgumentParser(description="")
    parse.add_argument("--adapter_path",type=str,default="/work/imc_lab/x_lv/ckpt/img_emb_model")
    parse.add_argument("--cls_model",type=str,default="swinv2_base_window12to24_192to384.ms_in22k_ft_in1k")  #resnet50 #swinv2_base_window12to24_192to384.ms_in22k_ft_in1k #convnextv2_base.fcmae_ft_in22k_in1k_384
    parse.add_argument("--cls_model_pretrained",type=str,default="/home/x_lv/texture/experiment/classfier_ckptswinv2_base_window12to24_192to384.ms_in22k_ft_in1k_type:4_False_epoch:3_96.93457097577542.pth")
    parse.add_argument("--node_type_path",type=str,default="/home/x_lv/texture/node_type.pkl")
    parse.add_argument("--image_input",type=str,default="/work/imc_lab/x_lv/output/final_dataset/origin.png")
   
    args=parse.parse_args()
    if not os.path.exists("/home/x_lv/texture/experiment/mask_emb/"):
        os.mkdir("/home/x_lv/texture/experiment/mask_emb/")
    if not os.path.exists("/home/x_lv/texture/experiment/output/"):
        os.mkdir("/home/x_lv/texture/experiment/output/")
    
    node_queue=queue.Queue()
    node_list=[]

    class utils_():
        def __init__(self,device) -> None:
           
            transform_cfg=timm.data.resolve_data_config(timm.create_model(args.cls_model).pretrained_cfg)
            mean,std,size=transform_cfg["mean"],transform_cfg["std"],transform_cfg["input_size"][-1]
            self.transforms=transforms.Compose([transforms.Resize([size,size]),transforms.ToTensor(),transforms.Normalize(mean,std)])
            with open(args.node_type_path,"rb") as f:
                self.node_dict=pickle.load(f)
            self.num_node_type=len(self.node_dict)
            self.ada_list=[file for file in Path(args.adapter_path).rglob("*") if file.is_dir()]
            model_classifier=timm.create_model(args.cls_model,pretrained=True).to(device)
            
            in_features=model_classifier.head.fc.in_features
            model_classifier.head.fc=nn.Linear(in_features,self.num_node_type,bias=True)
             
            model_classifier.load_state_dict(torch.load(args.cls_model_pretrained))
            self.model_classifier=model_classifier.to(device)
    utils=utils_(device)

    id="stabilityai/stable-diffusion-xl-base-1.0"  #"stabilityai/stable-diffusion-xl-base-1.0"  
    unet=UNet2DConditionModel.from_pretrained(id,subfolder="unet",torch_dtype=dtype).to(device)
    unet.requires_grad_(False)
    vae=AutoencoderKL.from_pretrained(id,subfolder='vae',torch_dtype=torch.float32).to(device)
    vae.requires_grad_(False)
    scheduler=DDIMScheduler.from_pretrained(id,subfolder="scheduler")
    scheduler.set_timesteps(50)

    image_encoder = CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter", subfolder="models/image_encoder",torch_dtype=dtype).to(device)
    #text_encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)
    clip_image_processor=CLIPImageProcessor()
    pipe=StableDiffusionXLCustomPipeline(vae,unet,scheduler,image_encoder,clip_image_processor,None) 
    img_emb_model=img_embedding().to(device,dtype)   


    image_=Image.open(args.image_input) 
    if image_.mode=="I;16" or image_.mode == "I":
                        image=np.array(image_)
                        scaled_tensor = image/65535
                        x=(scaled_tensor*255.0).astype(np.uint8)
                        image_= Image.fromarray(x, mode='L').convert("RGB")
    else:
        image_ =image_.convert("RGB")
    node_first=Node(image_,utils,"dyngradient",is_refiner=False)
    image_inqueue(node_queue,node_list,node_first)
    

    while not node_queue.empty():
         
        node_process(args,utils,pipe,node_queue,node_list)


    
    
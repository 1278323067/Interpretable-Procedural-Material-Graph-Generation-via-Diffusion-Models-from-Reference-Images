import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ["http_proxy"] = "http://10.10.115.8:7890"
os.environ["https_proxy"] = "http://10.10.115.8:7890"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
import timm
from model import GPT2LMHeadModel_custom,GPT2LMHeadModel
from transformers import GPT2Tokenizer ,AutoConfig
import json
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from transformers import Trainer,TrainingArguments
#import wandb
from utils import vocab_T,T_D,custom_Datacollator,custom1_Datacollator,vocab_T1,decode_image,Sample
from PIL import Image
from pathlib import Path

def sample(dir:str):
    with open(dir,"rb") as f:
        node_params=json.load(f)
     
def  decode(codes,tokenizer:GPT2Tokenizer):
    _codes=codes[0].tolist()
    out_token=[]
    
    for i,code in enumerate(_codes):
        if "input" in tokenizer.convert_ids_to_tokens(code) and _codes[i+1] != tokenizer.convert_tokens_to_ids("<Null>"):
            print(tokenizer.convert_ids_to_tokens(code))
            print(i)
            out_token.append(_codes[i+1:i+1+256])
        
    return out_token


torch.manual_seed(1)
ck_point=Path("/home/x_lv/texture/experiment/texture_transformer/test_output/lbs/checkpoint-25000/")
 
tokenizer=GPT2Tokenizer.from_pretrained("/home/x_lv/texture/experiment/texture_transformer/My_tokenizer")
_config=AutoConfig.from_pretrained("gpt2")
_config.vq_vocab_size=16384
_config.vocab_size=len(tokenizer)
vocab=vocab_T1(16384)
model=GPT2LMHeadModel_custom.from_pretrained(ck_point)
#model.resize_token_embeddings(len(tokenizer))
#model.resize_token_embeddings(len(vocab)) 


data_collator = custom_Datacollator(tokenizer=tokenizer)
 
transform=transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
                  
 
device="cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
'''dataset=T_D("/home/x_lv/Dataset/output/generator_output1",model_dir="/home/x_lv/texture/rq_vae_transformer/output/Texture_D-rqvae-8x8x4/16042024_172856/epoch10_model.pt",device=device,transforms=transform,tokenizer=tokenizer)
 
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                            [train_size, test_size])
test_loader=DataLoader(train_dataset,batch_size=1,shuffle=False ,collate_fn=data_collator)

data=next(iter(test_loader)) '''
#print(data["input_ids"])
path="/home/x_lv/Dataset/output/generator_output1/dyngradient/2"
data=Sample(path,transform=transform,model_dir="/home/x_lv/texture/rq_vae_transformer/output/Texture_D-rqvae-8x8x4/16042024_172856/epoch10_model.pt",tokenizer=tokenizer).parse() 
 
output=model.generate(data[0].to(device),num_beams=1,do_sample=False,max_length=1025,texture_type=data[1].to(device),vq_emb_mask=[[0]],pad_token_id=tokenizer.eos_token_id)#pad_token_id=tokenizer.pad_token) 
 
 
 
out_token=decode(output,tokenizer)
 
for index,i in enumerate(out_token):
    codes=tokenizer.convert_ids_to_tokens(i)
    codes=[ int(i[1:-1]) for i in codes]
    print(codes[::4])
    code=torch.tensor(codes)
    image=decode_image("/home/x_lv/texture/rq_vae_transformer/output/Texture_D-rqvae-8x8x4/16042024_172856/epoch10_model.pt",code.reshape(8,8,4).unsqueeze(0).to(device))
    image.save(f"test{index}.png")
import os
import pickle
import sys
sys.path.append("/home/x_lv/texture/")
import torch 
from torch.nn.utils.rnn import pad_sequence
from  rq_vae_transformer.rqvae import create_model
from rq_vae_transformer.rqvae import RQVAE
from typing import NewType
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import json 
from pathlib import Path
import numpy as np
#from datasets import Dataset
import tokenizers
from rq_vae_transformer.rqvae import load_config, augment_arch_defaults

def load_model(path, ema=False):

    model_config = os.path.join(os.path.dirname(path), 'config.yaml')
    config = load_config(model_config)
    config.arch = augment_arch_defaults(config.arch)

    model, _ = create_model(config.arch, ema=False)
    #path='/home/x_lv/texture/rq-vae-transformer/imagenet_rqvae.pt'
    ckpt = torch.load(path)['state_dict_ema'] if ema else torch.load(path)['state_dict']
    model.load_state_dict(ckpt)

    return model, config


def decode_image(dir:str,code,  device="cuda"):
    model=load_model(dir)[0].to(device)
    image=model.decode_code(code)
    image*=0.5+0.5
    image=torch.clamp(image,0,1)
    image=F.to_pil_image(image[0].cpu() )
    return image


RQVAE=NewType("RQVAE",RQVAE)
#/home/x_lv/Dataset/output/generator_output/
 

def getListOfTextures(dirName):
    # create a list of all files in a root dir
    if dirName not in ["gradient"]:
        listOfFile = os.listdir(dirName)
        allFiles = list()
        for entry in listOfFile:


                fullPath = os.path.join(dirName, entry)
                
                if os.path.isdir(fullPath):
                    allFiles = allFiles + getListOfTextures(fullPath)
                else:
                    if os.path.splitext(entry)[1] ==".json":
                        pathname = os.path.join(dirName,os.path.splitext(entry)[0])
                        allFiles.append(pathname)


    return allFiles


class vocab_T():
    def __init__(self) -> None:
        self.special_tokens=[]
        for i in range(10):
            self.special_tokens.append("<input"+str(i)+">")
        self.special_tokens.extend(["<int>","<float>","<list1d>","<list2d>","<list3d>","<bool>","<str>","<Null>","<placeholder>"])
        for i in range(16384):
            self.special_tokens.append(f"<{i}>")
class vocab_T1():
    def __init__(self,vocab_size) -> None:
        self.special_tokens={}
        self.special_tokens_list=[]
        self.special_param_list=[]
        self.vocab_size=vocab_size

    

        for i in range(10):
            self.add_special_token("<input"+str(i)+">",is_params=True)
        
        self.add_special_token("<Null>")
        self.add_special_token("<pad>")
      

    def get_ids(self,name:str):
        return self.special_tokens[name]
    
    def add_special_token(self,name,is_params=False):
        self.special_tokens.update({name:self.vocab_size})
        self.special_tokens_list.append(self.vocab_size)
        if is_params:
            self.special_param_list.append(self.vocab_size)
        self.vocab_size+=1

    def __len__(self):
        return self.vocab_size
    
    def  decode(self,codes):
        _codes=codes[0]
        out_token=[]
        for i,code in enumerate(_codes):
            if code in self.special_param_list and _codes[i+1] != self.get_ids("<Null>"):
                out_token.append(_codes[i+1:i+1+256])
         
        return out_token
                  

class Sample():
    def __init__(self,image_dir:str,model_dir:str,transform,tokenizer,device="cuda") -> None:
        self.img_dir=Path(image_dir)
        self.model=load_model(model_dir)[0]
        self.tokenizer=tokenizer
        self.transform=transform
        self.device=device
        with open("/home/x_lv/texture/RADAM/RADAM/lable_pair.pkl","rb") as f:
            self.lable_pair=pickle.load(f)
    def parse(self):
        
        with open(self.img_dir.with_suffix(".json"),"rb") as f:
            params=json.load(f)
        img_type=self.lable_pair.get(list(params.keys())[0])
        img_type=torch.tensor(img_type,device=self.device).unsqueeze(0)
        image=Image.open(self.img_dir.with_suffix(".png")).convert('RGB')
         
        image=self.transform(image)
        self.model=self.model.to(self.device)
        image=image.to(self.device).unsqueeze(0)
        codes=self.model(image)[-1].reshape(-1).tolist()
        codes=[f"<{i}>" for i in codes]
        #codes=codes.unsqueeze(0).to(self.device)
        token=self.tokenizer(codes)["input_ids"]
        token=[ j for i in token for j in i]
        return torch.tensor(token).unsqueeze(0),img_type
    
    def re_parse(self,ids):
        codes=self.tokenizer.convert_ids_to_tokens(ids)
        codes=[ int(i[2:-2]) for i in codes]
        return codes

class T_D(Dataset):
    def __init__(self,root:str,model_dir:str,device:str,transforms,tokenizer=None,vocab:vocab_T1=None):
        self.dir=root
        self.image_file=getListOfTextures(self.dir)
        self.transform=transforms
        self.model=load_model(model_dir)[0]
        self.device=device
        self.tokenizer=tokenizer
        with open("/home/x_lv/texture/RADAM/RADAM/lable_pair.pkl","rb") as f:
            self.lable_pair=pickle.load(f)
        self.vocab=vocab
    def __len__(self):
        return len(self.image_file)
    
    def get_code(self,img_path):
        
        image=Image.open(img_path.with_suffix(".png")).convert('RGB')
        image=self.transform(image)
        self.model=self.model.to(self.device)
        image=image.to(self.device).unsqueeze(0)
        codes=self.model(image)[-1]
        return codes.reshape(-1).tolist()
    
    def __getitem__(self,idx):
        
        path=Path(self.image_file[idx])

        codes=self.get_code(path)
        codes=[ f"<{i}>" for i in codes]
        place_code=[]
        vq_emb_mask=[0]
        #place_code.extend(codes)
        place_code.extend(codes)
        with open(path.with_suffix(".json"),"r") as f:
            content=json.load(f)
            param=list(content.values())[0]
            texture_type=self.lable_pair.get(list(content.keys())[0])

       
        '''for k, v in param.items():
           if k=="input":
                for index,i in enumerate(param[k]):
                     
                    place_code.append(self.vocab.get_ids("<input"+str(index)+">"))
                    if i is None:
                        place_code.append(self.vocab.get_ids("<Null>"))
                    else:
                        n_path=(Path(path).parents[2])/(Path(i).as_posix().replace("\\","/"))
                        n_codes=self.get_code(n_path)
                        place_code.extend(n_codes)'''
                        
        
        #params.append("<placeholder>")
        for k, v in param.items():
            if k=="input":
                for index,i in enumerate(param[k]):
                    if i is not  None:
                        
                        place_code.append("<input"+str(index)+">")
                        vq_emb_mask.append(len(place_code))
                        n_path=(Path(path).parents[2])/(Path(i).as_posix().replace("\\","/"))
                        n_codes=self.get_code(n_path)
                        n_codes=[f"<{i}>" for i in n_codes]
                        place_code.extend(n_codes)
                        #params.append("<placeholder>")
            elif type(v) == int or type(v) == float or type(v) == str or type(v) == bool:
                place_code.append(f"<{str(type(v))[8:-2]}>")
                place_code.append(str(v))
            elif type(v) == list:
                place_code.append(f"<list{np.array(v).ndim}d>")
                place_code.append(str(v))
            else:
                raise ValueError(f"not implemented type{type(v)}") 
        token=self.tokenizer(place_code)["input_ids"]
        token=[ j for i in token for j in i]
       
        '''place_ids=self.tokenizer.encode("<placeholder>")[0]
        token=self.tokenizer(str(tuple(params)))['input_ids']
        place_code_i=0
        for i, ids in enumerate(token):
            if ids==place_ids:
                token[i:i+1]=place_code[place_code_i]
                place_code_i+=1
                vq_emb_mask.append(i)'''

        #re_data={"input_ids":token,"texture_type":label,"vq_emb_mask":vq_emb_mask}

        re_data={"input_ids":token,"texture_type":texture_type,"vq_emb_mask":vq_emb_mask}
        
        

        return  re_data # torch.tensor(place_code),torch.tensor(label)  #token,label,vq_emb_mask 
    
class custom1_Datacollator:
    def __init__(self,vocab:vocab_T1):
         self.pad=vocab.get_ids("<pad>")
        
    def __call__(self, batch):
        input_ids=[]
        label=[]
        for i,l in batch:
            input_ids+=[i]
            label+=[l]
        batch = [item.t() for item in input_ids]
        ids=pad_sequence(batch,batch_first=True,padding_value=self.pad)
        #ids=ids.permute(0,2,1)
        labels=torch.stack(label)
        return {"input_ids":ids,"texture_type":labels,"labels":ids}

class custom_Datacollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, examples):
        vq_emb_mask=[]
        for feature in examples:
            vq_emb_mask.append(feature["vq_emb_mask"])
            del feature["vq_emb_mask"]
        
        batch = self.tokenizer.pad(examples,padding="max_length",return_tensors="pt" )
        batch.update({"labels":batch["input_ids"]})
        batch.update({"vq_emb_mask":vq_emb_mask})
        return batch
    




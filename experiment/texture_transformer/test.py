import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3"
os.environ["http_proxy"] = "http://10.10.115.8:7890"
os.environ["https_proxy"] = "http://10.10.115.8:7890"
os.environ["NCCL_P2P_DISABLE"]="1"
os.environ["NCCL_IB_DISABLE"]="1"
import torch

import timm
import torch.distributed as dist
from model import GPT2LMHeadModel_custom,GPT2LMHeadModel
from transformers import GPT2Tokenizer,AutoConfig
from utils import vocab_T,T_D,custom_Datacollator,custom1_Datacollator,vocab_T1

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from transformers import Trainer,TrainingArguments,DataCollatorWithPadding,DataCollatorForSeq2Seq,get_linear_schedule_with_warmup
import wandb

from tqdm import tqdm
import argparse
import torch.nn as nn
import deepspeed

local_rank=int(os.environ["LOCAL_RANK"])



def train(model,train_loader,loss_fn,optimizer,args,epoch,device):
    model.train()
    train_loss=0
    train_acc=0
    total_train_acc=0
    data_size=0
    for i,data in tqdm(enumerate(train_loader)):
            input_ids=data["input_ids"].to(device)
            labels=data["labels"].to(device)
            texture_type=data["texture_type"].to(device)
            vq_emb_mask=data["vq_emb_mask"]
            optimizer.zero_grad()
            output=model(input_ids=input_ids,labels=labels,texture_type=texture_type,vq_emb_mask=vq_emb_mask)
            logits=output.logits
            loss=output.loss
            loss.backward()
            optimizer.step()
            
            train_loss+=loss.item()*len(labels)
            _,preds=torch.max(logits,1)
            train_acc=(preds==labels[..., 1:]).sum().item()
            total_train_acc+=train_acc
            data_size+=len(labels)
            if i% 10 ==9:
                wandb.log({"train_loss":loss.item(),"train_acc":100*(train_acc/len(labels)),"epoch":epoch})
                print({"train_loss":loss.item(),"train_acc":100*(train_acc/len(labels)),"epoch":epoch})
    return {"total_train_loss":train_loss/data_size,"total_train_acc":100*(total_train_acc/data_size)}

def test(model,test_loader,loss_fn,device):
    model.eval()
    test_loss=0
    test_acc=0
    top_k=0
    data_size=0
    with torch.no_grad():
        for i,data  in tqdm(enumerate(test_loader)):
            input_ids=data["input_ids"].to(device)
            labels=data["labels"].to(device)
            texture_type=data["texture_type"].to(device)
            vq_emb_mask=data["vq_emb_mask"]
             
            logits=model(input_ids=input_ids,labels=labels,texture_type=texture_type,vq_emb_mask=vq_emb_mask)
            loss=loss_fn(logits,labels)

            test_loss+=loss.item()*len(labels)
            _,preds=torch.max(logits,1)
            test_acc+=(preds==labels[..., 1:]).sum().item()
            
            '''_, pred_k = logits.topk(3, 1, True, True)
            pred_k = pred_k.t()
            correct = pred_k.eq(labels.view(1, -1).expand_as(pred_k))
            top_k += correct[:3].reshape(-1).float().sum(0, keepdim=True).item()'''
            data_size+=len(labels)


    return {"test_loss":test_loss/data_size,"test_acc":100*(test_acc/data_size)}


 
#torch.distributed.init_process_group(backend='nccl')
deepspeed.init_distributed()
parse=argparse.ArgumentParser(description="")

parse.add_argument("--batch_size",type=int,default=1)
parse.add_argument("--lr",type=float,default=1e-4)
parse.add_argument("--epoch",type=int,default=3)
parse.add_argument("--local_rank", type=int, default=0)
args=parse.parse_args()
g_config=AutoConfig.from_pretrained("gpt2")
g_config.vq_vocab_size=16384
tokenizer=GPT2Tokenizer.from_pretrained("/home/x_lv/texture/experiment/texture_transformer/My_tokenizer")
'''TOKEN_DICT={"additional_special_tokens":vocab_T().special_tokens}
tokenizer.add_special_tokens(TOKEN_DICT)
tokenizer.pad_token = tokenizer.eos_token 
tokenizer.save_pretrained("./My_tokenizer")'''
model=GPT2LMHeadModel_custom.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

'''g_config=AutoConfig.from_pretrained("gpt2")
vocab=vocab_T1(16384)
g_config.vq_vocab_size=len(vocab)
model=GPT2LMHeadModel_custom(g_config)
 '''
 
transform=transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
data_collator = custom_Datacollator(tokenizer=tokenizer)  ##############
        

device=torch.device('cuda',local_rank)
dataset=T_D("/home/x_lv/Dataset/output/generator_output1",model_dir="/home/x_lv/texture/rq_vae_transformer/output/Texture_D-rqvae-8x8x4/16042024_172856/epoch10_model.pt",device=device,transforms=transform,tokenizer=tokenizer)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                            [train_size, test_size],generator=torch.Generator().manual_seed(1))

wandb.init(
    project="transformer_test"
)

'''
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
train_loader=DataLoader(train_dataset,batch_size=args.batch_size,shuffle=False,sampler=train_sampler,collate_fn=data_collator)
test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
test_loader=DataLoader(test_dataset,batch_size=args.batch_size,shuffle=False,sampler=test_sampler,collate_fn=data_collator)

wandb.init(project="transformer_test",config={"epoch":args.epoch,"lr":args.lr,"batch_size":args.batch_size},group="DDP")

model.to(device)
model=nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

loss_fn=nn.CrossEntropyLoss(label_smoothing=0.0)
optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

best_acc=0.0
best_epoch=0
best_name=None
for i in tqdm((range(args.epoch))):
    train_loader.sampler.set_epoch(i)
    test_loader.sampler.set_epoch(i)
    train_metrics=train(model,train_loader,loss_fn,optimizer,args,i,device=device)
    test_metrics=test(model,test_loader,loss_fn,args,device=device)

    exp_lr_scheduler.step()

    acc=test_metrics["test_acc"]
    if acc>best_acc and torch.distributed.get_rank() == 0:
        
        best_acc=acc
        best_epoch=i
        if best_name is not None:
            os.remove(best_name)
        best_name=f"{args.model}_type:{args.type}_{args.finetune_all}_epoch:{i}_{acc}.pth"
        
        
        torch.save(model.module.state_dict(),best_name)

    if (acc<best_acc) and (i-best_epoch)>=3:
        print("##################################")
        print(f"early stop at epoch{i}")
        print("##################################")
        break
    


    print({**train_metrics,**test_metrics})
    wandb.log({**train_metrics,**test_metrics})
    '''
def hook_fn(m, i, o):
  print(m)

'''for i in model.children():
    i.register_backward_hook(hook_fn)'''

training_args=TrainingArguments(
    output_dir="/home/x_lv/texture/experiment/texture_transformer/test_output/lbs",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    #fp16=True,
    #deepspeed="deepspeed_config.json",
    learning_rate=args.lr,
    lr_scheduler_type ="cosine",
    #auto_find_batch_size=True,
    #load_best_model_at_end=True,
    report_to = "wandb",
    logging_steps = 10,
    #save_total_limit =1,
    save_strategy="steps",
    evaluation_strategy="steps",
    eval_steps=5001,
    save_steps=5000,
    #label_smoothing_factor=0.1,
    #ddp_find_unused_parameters=False
)    

trainer=Trainer(
    model=model,
    args=training_args,
    #tokenizer=tokenizer,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset =test_dataset
)


trainer.train()


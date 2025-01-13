import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
os.environ["http_proxy"] = "http://10.10.115.11:7890"
os.environ["https_proxy"] = "http://10.10.115.11:7890"
os.environ["NCCL_P2P_DISABLE"]="1"
os.environ["NCCL_IB_DISABLE"]="1"
import torch.distributed
#os.environ["WANDB_MODE"]="offline"
from transformers import TrainingArguments,Trainer,get_cosine_schedule_with_warmup
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from tqdm import tqdm
import wandb
import numpy as np 
import sys
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
import pickle
sys.path.append('/home/x_lv/texture/RADAM/RADAM')
from diffusers.optimization import get_cosine_schedule_with_warmup

from feature_extraction import extract_features
from datasets_ import Texture_D
from accelerate import Accelerator,DistributedDataParallelKwargs

#local_rank=int(os.environ["LOCAL_RANK"])

class TrainingConfig:
    def __init__(self):
        self.image_size = 512  # the generated image resolution
        self.train_batch_size =8
        self.eval_batch_size = 1  # how many images to sample during evaluation
        self.num_epochs = 10
        self.gradient_accumulation_steps =1
        self.learning_rate = 1e-4
        self.lr_warmup_steps = 0
        self.save_model_epochs = 10000
        self.mixed_precision = "no"  # `no` for float32, `fp16` for automatic mixed precision
        self.output_dir = "/home/x_lv/texture/experiment/texture_diffusion/out-diff"  # the model name locally and on the HF Hub
        
        
        self.overwrite_output_dir = True  # overwrite the old model when re-running the notebook
        self.seed = 0


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
 
        loss = (  self.alpha[targets]* (1 - pt) ** self.gamma * ce_loss).mean()
        return loss
    

def train(model,train_loader,loss_fn,optimizer,args,epoch,device,lr_scheduler):
    model.train()
    train_loss=0
    train_acc=0
    total_train_acc=0
    data_size=0
    for i,(images,lables) in tqdm(enumerate(train_loader)):
            images=images.to(device)
            lables=lables.to(device)
            optimizer.zero_grad()
            logits=model(images)
            loss=loss_fn(logits,lables)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            train_loss+=loss.item()*len(lables)
            _,preds=torch.max(logits,1)
            train_acc=(preds==lables).sum().item()
            total_train_acc+=train_acc
            data_size+=len(lables)
            if i% 50 ==49:
                wandb.log({"train_loss":loss.item(),"train_acc":100*(train_acc/len(lables)),"epoch":epoch})
                print({"train_loss":loss.item(),"train_acc":100*(train_acc/len(lables)),"epoch":epoch,"lr_rate":lr_scheduler.get_lr()[0]})
    return {"total_train_loss":train_loss/data_size,"total_train_acc":100*(total_train_acc/data_size)}

def test(model,test_loader,loss_fn,args,device):
    model.eval()
    test_loss=0
    test_acc=0
    top_k=0
    data_size=0
    with torch.no_grad():
        for i,(images,lables) in tqdm(enumerate(test_loader)):
            images=images.to(device)
            lables=lables.to(device)

            logits=model(images)
            loss=loss_fn(logits,lables)


            test_loss+=loss.item()*len(lables)
            _,preds=torch.max(logits,1)
            test_acc+=(preds==lables).sum().item()
            
            _, pred_k = logits.topk(3, 1, True, True)
            pred_k = pred_k.t()
            correct = pred_k.eq(lables.view(1, -1).expand_as(pred_k))
            top_k += correct[:3].reshape(-1).float().sum(0, keepdim=True).item()
            data_size+=len(lables)


    return {"test_loss":test_loss/data_size,"test_acc":100*(test_acc/data_size),"top_k":100*top_k/data_size}

def origin_test(model):
    pass


if  __name__=="__main__":
    #torch.distributed.init_process_group(backend='nccl')
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    parse=argparse.ArgumentParser(description="")
    parse.add_argument("--data_path",type=str,default="/work/imc_lab/x_lv/output/generator_output")
    parse.add_argument("--model",type=str,default="swinv2_base_window12to24_192to384.ms_in22k_ft_in1k")  #resnet50 #swinv2_base_window12to24_192to384.ms_in22k_ft_in1k #convnextv2_base.fcmae_ft_in22k_in1k_384
    parse.add_argument("--node_type_path",type=str,default="/home/x_lv/texture/node_type.pkl")
    parse.add_argument("--type",type=int,default=4,help="0:fine tune last layer   1:RADAM   2:visualize table data  3: " )
    parse.add_argument("--finetune_all" ,default=False, action="store_true")
    parse.add_argument("--batch_size",type=int,default=8)
    parse.add_argument("--lr",type=float,default=5e-4)
    parse.add_argument("--epoch",type=int,default=5)
    args=parse.parse_args()

    config = TrainingConfig()
     
    #device = torch.device('cuda', local_rank)


    transform_cfg=timm.data.resolve_data_config(timm.create_model(args.model).pretrained_cfg)
    mean,std,size=transform_cfg["mean"],transform_cfg["std"],transform_cfg["input_size"][-1]


    _transforms=transforms.Compose([transforms.Resize([size,size]),transforms.ToTensor(),transforms.Normalize(mean,std)])
    dataset=Texture_D(args.data_path,_transforms,args.node_type_path)
    train_size = int(0.85 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                [train_size, test_size],generator=torch.Generator().manual_seed(0))
    print(len(train_dataset))
    print(len(test_dataset))
    train_dataloader=DataLoader(train_dataset,batch_size=config.train_batch_size)
    test_dataloader=DataLoader(test_dataset,batch_size=config.eval_batch_size)
    '''train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader=DataLoader(train_dataset,batch_size=args.batch_size,shuffle=False,sampler=train_sampler)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader=DataLoader(test_dataset,batch_size=args.batch_size,shuffle=False,sampler=test_sampler)

    wandb.init(project=str(args.type),config={"epoch":args.epoch,"lr":args.lr,"batch_size":args.batch_size,"optimizer":"adamw"},group="DDP")'''

    '''all_labels=dict()
    for _, label in  train_dataset:
        if str(label) not in all_labels:
            all_labels.setdefault(str(label),1)
        else:
            all_labels[str(label)]+=1
    print(all_labels)
    all_labels=sorted(all_labels.items(),key=lambda x :x[0])
    print(all_labels)
    class_counts = [k[1] for k in all_labels]
    print(class_counts)'''
    
    if args.type==0:

        model=timm.create_model(args.model,pretrained=True)

        '''if not  args.finetune_all:
            for p in  model.parameters():
                p.requires_grad=False'''
        
        #in_features=model.fc.in_features    resnet50
        in_features=model.head.fc.in_features
        model.head.fc=nn.Linear(in_features,35,bias=True)
        model.to(device)
        model=nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        
        loss_fn=nn.CrossEntropyLoss(label_smoothing=0.0)
        optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.01)
        exp_lr_scheduler = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=0,num_training_steps=len(train_loader)*args.epoch)

        best_acc=0.0
        best_epoch=0

        checkpoint = torch.load('/home/x_lv/texture/experiment/swinv2_base_window12to24_192to384.ms_in22k_ft_in1k_type:0_False_epoch:16_84.18048217647416.pth')
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        exp_lr_scheduler.load_state_dict(checkpoint["scheduler"])

        best_name=None
        for i in tqdm((range(start_epoch,args.epoch))):
            train_loader.sampler.set_epoch(i)
            test_loader.sampler.set_epoch(i)
            train_metrics=train(model,train_loader,loss_fn,optimizer,args,i,device=device,lr_scheduler=exp_lr_scheduler)
            test_metrics=test(model,test_loader,loss_fn,args,device=device)

            

            acc=test_metrics["test_acc"]
            if acc>best_acc and torch.distributed.get_rank() == 0:
                
                best_acc=acc
                best_epoch=i
                if best_name is not None:
                    os.remove(best_name)
                best_name=f"{args.model}_type:{args.type}_{args.finetune_all}_epoch:{i}_{acc}.pth"
                
                checkpoint = {
                        "net": model.state_dict(),
                        'optimizer':optimizer.state_dict(),
                        "epoch": i,
                        'scheduler': exp_lr_scheduler.state_dict(),
                    }
                torch.save(checkpoint,best_name)

            if (acc<best_acc) and (i-best_epoch)>=3:
                print("##################################")
                print(f"early stop at epoch{i}")
                print("##################################")
                break
            


            print({**train_metrics,**test_metrics})
            wandb.log({**train_metrics,**test_metrics})

    elif args.type==1:
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'

        file="./output/RADAM_features_"+args.model
        print("~~~~~~~~~~~~~~~~"+file)
        model=nn.Sequential(
        nn.Linear(3904,35,bias=True) 
        ).to(device)
        model=nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        
        loss_fn=nn.CrossEntropyLoss(label_smoothing=0.0)
        optimizer=torch.optim.SGD(model.parameters(),lr=args.lr,momentum=0.9)
        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        if os.path.isfile(file):
            with open(file,"rb") as f:
                X_train,Y_train,X_test,Y_test=pickle.load(f)
                print(f"feature size: {X_train.shape}")
        else:
            X_train, Y_train = extract_features(args.model, train_dataset, depth="all",
                                                    pooling="AvgPool2d", M=1,
                                                    batch_size=args.batch_size, multigpu=True,
                                                    seed=666999,local_rank=local_rank)
            X_test, Y_test = extract_features(args.model, test_dataset, depth="all",
                                                    pooling="AvgPool2d", M=1,
                                                    batch_size=args.batch_size, multigpu=True,
                                                    seed=666999,local_rank=local_rank)
       
            with open(file,"wb") as f:
                pickle.dump([X_train,Y_train,X_test,Y_test],f)
                print(f"feature size: {X_train.shape}")

        class feature(Dataset):
            def __init__(self,Input,Lable) -> None:
                super().__init__()
                self.data=torch.from_numpy(Input).to(dtype=torch.float32)
                self.label=Lable 
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, index) :
                    
                return self.data[index],self.label[index]
    
        train_dataset1=feature(X_train,Y_train)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset1)
        train_loader_=DataLoader(train_dataset1,batch_size=args.batch_size,sampler=train_sampler)

        test_dataset1=feature(X_test,Y_test)
        train_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset1)
        test_loader_=DataLoader(test_dataset1,batch_size=args.batch_size,sampler=test_sampler)
        best_acc=0.0
        best_epoch=0
        best_name=None
        for i in tqdm((range(args.epoch))):
            train_loader_.sampler.set_epoch(i)
            test_loader_.sampler.set_epoch(i)
            train_metrics=train(model,train_loader_,loss_fn,optimizer,args,i,device=device)
            test_metrics=test(model,test_loader_,loss_fn,args,device=device)

            exp_lr_scheduler.step()

            acc=test_metrics["test_acc"]
            if acc>best_acc and torch.distributed.get_rank() == 0:
                
                best_acc=acc
                best_epoch=i
                if best_name is not None:
                    os.remove(best_name)
                best_name=f"{args.model}_type_{args.type}_epoch_{i}_{acc}.pth"
                torch.save(model.state_dict(),best_name)

            if (acc<best_acc) and (i-best_epoch)>=3:
                print("##################################")
                print(f"early stop at epoch{i}")
                print("##################################")
                break
            


            print({**train_metrics,**test_metrics})
            wandb.log({**train_metrics,**test_metrics})
    elif args.type==2:
         
        '''pretrained_cfg=timm.create_model(args.model).default_cfg
        pretrained_cfg['file']="resnet50_type_1_epoch_14_80.82566359877285.pth"
        model=timm.create_model(args.model,pretrained=True,pretrained_cfg=pretrained_cfg)'''
        model=nn.Sequential(
        nn.Linear(3904,35,bias=True) 
        ).to(device)
        state_dict = torch.load('resnet50_type_1_epoch_10_80.90569561157797.pth')
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)
        columns=["image","pred","truth"]
        table=wandb.Table(columns=columns)
        with open("../RADAM/RADAM/lable_pair.pkl","rb") as f:
            lable_pair=pickle.load(f)
        print(lable_pair)
     
        loss_fn=nn.CrossEntropyLoss()
        model.eval()
        X_test, Y_test,images = extract_features(args.model, test_dataset, depth="all",
                                                    pooling="AvgPool2d", M=1,
                                                    batch_size=args.batch_size, multigpu=True,
                                                    seed=666999,local_rank=local_rank)
        
        class feature(Dataset):
            def __init__(self,Input,Lable,Images) -> None:
                super().__init__()
                self.data=torch.from_numpy(Input).to(dtype=torch.float32)
                self.label=Lable 
                self.images=torch.from_numpy(Images)
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, index) :
                    
                return self.data[index],self.label[index],self.images[index]
    


        test_dataset1=feature(X_test,Y_test,images)
        train_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset1)
        test_loader_=DataLoader(test_dataset1,batch_size=args.batch_size,sampler=test_sampler)
        
        with torch.no_grad():
            for i,(data,lables,images) in tqdm(enumerate(test_loader_)):
                 
                test_loader.sampler.set_epoch(i)
                data=data.to(device)
                lables=lables.to(device)
                
       
                logits=model(data)
                loss=loss_fn(logits,lables)
                 
                _,preds=torch.max(logits,1)
                
                images=images.permute(0,2,3,1).cpu().numpy()
                lables=lables.cpu().numpy()
                preds=preds.cpu().numpy()

                for image,lable,pred in zip(images,lables,preds):
                    table.add_data(wandb.Image(image),lable,pred)
            wandb.log({"test_predictions":table})
    elif args.type==3:
        model=timm.create_model(args.model,pretrained=True)
        training_args=TrainingArguments(
        output_dir="/home/x_lv/texture/experiment/texture_transformer/test_output/classifier",
        overwrite_output_dir=True,
        num_train_epochs=args.epoch,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        #fp16=True,
        #deepspeed="deepspeed_config.json",
        learning_rate=args.lr,
        lr_scheduler_type ="cosine",
        #auto_find_batch_size=True,
        #load_best_model_at_end=True,
        report_to = "wandb",
        logging_steps = 100,
        #save_total_limit =1,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        #eval_steps=5001,
        #save_steps=5000,
        #label_smoothing_factor=0.1,
        #ddp_find_unused_parameters=FalseSS
    )    

        trainer=Trainer(
            model=model,
            args=training_args,
            #tokenizer=tokenizer,
            #data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset =test_dataset
        )
        trainer.train()
    elif args.type == 4:
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
            wandb.init(project="texture_classification",config={})
            accelerator.init_trackers("train_example",config=config)

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
            
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
            
        model=timm.create_model(args.model,pretrained=True)
        '''if not  args.finetune_all:
            for p in  model.parameters():
                p.requires_grad=False'''
        with open(args.node_type_path,"rb") as f:
            num_node_type=len(pickle.load(f))
         #in_features=model.fc.in_features    resnet50
        in_features=model.head.fc.in_features
        model.head.fc=nn.Linear(in_features,num_node_type,bias=True)
        
        #model.load_state_dict(torch.load("/home/x_lv/texture/experiment/classfier_ckptswinv2_base_window12to24_192to384.ms_in22k_ft_in1k_type:4_False_epoch:3_96.34448574969021.pth"))
        
        class_counts=[319, 10464, 253, 1298, 943, 229, 1, 5001, 18885, 5230, 6548, 8450, 32674, 29, 9, 5, 1, 27, 18, 22, 17, 4895, 1286, 7724, 14285, 7529, 4225, 149, 391, 1493, 541, 1,9743, 867, 6727, 8261, 792, 2355, 1289, 2618, 161, 124, 123, 207, 122, 5982, 294, 107, 111, 116, 127, 309, 136, 120, 729, 95, 3405, 123, 2296, 3443, 4447, 13, 2172, 10, 622, 63, 54, 2754, 119, 650, 3880, 20, 14258, 6397, 8345, 874, 33996, 2134, 6118, 675, 893]
        num_classes = len(class_counts)
        total_samples = len(train_dataset)

        class_weights = []
        for count in class_counts:
            weight = 1/(count / total_samples)
            class_weights.append(weight)
        class_weights=torch.tensor(class_weights).to(accelerator.device)
        #loss_fn=nn.CrossEntropyLoss(label_smoothing=0.0)
        loss_fn=FocalLoss(alpha=class_weights, gamma=2)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,weight_decay=0.01)
        lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )
        
        optimizer, train_dataloader,test_dataloader, lr_scheduler,model = accelerator.prepare(
        optimizer, train_dataloader, test_dataloader,lr_scheduler,model
    )
        model.to(accelerator.device,weight_dtype)
        best_acc=0.0
        best_epoch=0
        best_name=None
        for epoch in range(config.num_epochs):
             
            progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")
            global_step=0
            model.train()
            train_loss=0
            train_acc=0
            total_train_acc=0
            data_size=0
            for step, (images,lables) in enumerate(train_dataloader):
                
                with accelerator.accumulate(model):
               
               
                    images=images.to(accelerator.device,weight_dtype)
                    lables=lables 
                   
                    logits=model(images)
                    loss=loss_fn(logits,lables)
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    train_loss+=loss.item()*len(lables)
                    _,preds=torch.max(logits,1)
                    train_acc=(preds==lables).sum().item()
                    total_train_acc+=train_acc
                    data_size+=len(lables)
                    
                    if step% 50 ==49:
                        global_step+=1
                        accelerator.log({"train_loss":loss.item(),"train_acc":100*(train_acc/len(lables)),"epoch":epoch,"lr": lr_scheduler.get_last_lr()[0],"step":global_step})
                     
                        progress_bar.update(50)
                        progress_bar.set_postfix({"train_loss":loss.item(),"train_acc":100*(train_acc/len(lables)),"epoch":epoch,"lr_rate":lr_scheduler.get_lr()[0]})
            accelerator.log({"total_train_loss":train_loss/data_size,"total_train_acc":100*(total_train_acc/data_size)})
            model.eval()
            test_loss=0
            test_acc=0
            top_k=0
            data_size=0
            with torch.no_grad():
                for i,(images,lables) in tqdm(enumerate(test_dataloader)):
                    images=images.to(accelerator.device)
                    lables=lables

                    logits=model(images)
                    loss=loss_fn(logits,lables)


                    test_loss+=loss.item()*len(lables)
                    _,preds=torch.max(logits,1)
                    test_acc+=(preds==lables).sum().item()
                    
                    _, pred_k = logits.topk(3, 1, True, True)
                    pred_k = pred_k.t()
                    correct = pred_k.eq(lables.view(1, -1).expand_as(pred_k))
                    top_k += correct[:3].reshape(-1).float().sum(0, keepdim=True).item()
                    data_size+=len(lables)

            accelerator.log({"test_loss":test_loss/data_size,"test_acc":100*(test_acc/data_size),"top_k":100*top_k/data_size})
                
                
            acc=100*(test_acc/data_size)
            if acc>best_acc and accelerator.is_main_process:
                
                best_acc=acc
                best_epoch=epoch
                if best_name is not None:
                    os.remove("/home/x_lv/texture/experiment/classfier_ckpt"+best_name)
                best_name=f"{args.model}_type:{args.type}_{args.finetune_all}_epoch:{epoch}_{acc}.pth"
                
                inter_model=accelerator.unwrap_model(model).state_dict()
                torch.save(inter_model,"/home/x_lv/texture/experiment/classfier_ckpt"+best_name)

            if (acc<best_acc) and (i-best_epoch)>=3:
                print("##################################")
                print(f"early stop at epoch{i}")
                print("##################################")
                break
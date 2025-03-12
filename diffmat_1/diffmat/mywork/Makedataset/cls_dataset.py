import os
import  shutil
import yaml
import json
import pickle

with open(r"../../config/node_list.yml", "r") as f:
    node_list = yaml.safe_load(f)
num=0
lable={}
for nodes in node_list.values():
    for node in nodes:
        lable.update({node:num})
        num+=1

with open(r"D:\AI\texture\utils\lable_pair.pkl", "wb") as f:
    pickle.dump(lable,f)
'''if __name__=="__main__":
    root=r"D:\AI\diffmat"
    ori_dir=os.path.join(root,"output")
    out_dir=os.path.join(root,"dataset")
    for file in os.listdir(ori_dir):
        d_dir=os.path.join(ori_dir,file)
        for file1 in os.listdir(d_dir):
            pre,suf=os.path.splitext(file1)
            if suf==".png":
                with open(os.path.join(d_dir,pre+'.json')) as f:
                    node_type=list(json.load(f).keys())[0]
                target=lable[node_type]
                shutil.copyfile(os.path.join(d_dir,file1),os.path.join(out_dir,str(target)+".png"))
                pass

'''
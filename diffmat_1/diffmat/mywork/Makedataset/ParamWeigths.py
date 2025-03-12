import os
import yaml

class ParamWeights():
    def __init__(self):
        self.param_weights={}
        self.folder_name=["generators","nodes"]
        self.path=r"D:\AI\diffmat\diffmat\config"
        self.init_node_param()


    def init_node_param(self):
        '''
        get all nodes' initial parameters
        '''
        for name in self.folder_name:
            full_path = os.path.join(self.path,name)
            for file in os.listdir(full_path):
                with open(os.path.join(full_path,file),'r') as  f:
                    content = yaml.safe_load(f)

                self.param_weights[file[:-4]]={}
                if content.get("param"):
                    default_param = content["param"]
                    for param in default_param:
                        param_name=param["name"]
                        self.param_weights[file[:-4]][param_name]={}



    def update(self,node_name,new_param):
        for key1, value1 in new_param.items():
            print(key1)
            if type(value1) != list:
                if  value1 in  self.param_weights[node_name][key1]:
                    self.param_weights[node_name][key1][value1]+=1
                else:
                    self.param_weights[node_name][key1].update({value1:1})
            elif node_name!="levels":
                for i, val in enumerate(value1):
                    if self.param_weights[node_name][key1].get(i):
                        if val in self.param_weights[node_name][key1][i]:
                            self.param_weights[node_name][key1][i][val] += 1
                        else:
                            self.param_weights[node_name][key1][i].update({val: 1})
                    else:
                        self.param_weights[node_name][key1].update({i:{val: 1}})
            else:
                if value1[0] in self.param_weights[node_name][key1]:
                    self.param_weights[node_name][key1][value1[0]] += 1
                else:
                    self.param_weights[node_name][key1].update({value1[0]: 1})


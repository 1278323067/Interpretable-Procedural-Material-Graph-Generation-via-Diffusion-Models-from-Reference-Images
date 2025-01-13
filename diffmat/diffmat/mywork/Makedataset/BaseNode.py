import os.path

from diffmat.core.material import  functional,noise
import pickle

class BaseNode():
    def __init__(self,name : str =''   ):
        self.name=name
        self.function=None

        self._init_node_function()
        self._init_param()

    def _init_node_function(self):
        if hasattr(functional,self.name):
            self.function=getattr(functional,self.name)
        elif hasattr(noise,self.name):
            self.function=getattr(noise,self.name)
        else:
            raise  RuntimeError("cannot find node function ")

    def _init_param(self):
        path=os.path.dirname(__file__)
        with open(os.path.join(path,"paramOutput"),"rb") as f:
            content=pickle.load(f)
        param=content[self.name]
        print(param)
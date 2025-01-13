from   BaseGraph import BaseGraph
from ParamWeigths import ParamWeights
import argparse
import os
from xml.etree import  ElementTree as ET
import pickle

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--sbs_global_path",type=str ,default=r"D:\AI\diffmat\test\sbs\match_v1\1\1")
    args=parser.parse_args()
    init_weights=ParamWeights()
    for file in os.listdir(args.sbs_global_path):
        path=os.path.join(args.sbs_global_path,file)
        if os.path.isfile(path):
            print(file+"##############")
            root=ET.parse(path).getroot()
            BaseGraph(root,init_weights,device="cpu")
    with open("paramOutput","wb") as f:
        pickle.dump(init_weights.param_weights,f)

if __name__ == "__main__":
    main()
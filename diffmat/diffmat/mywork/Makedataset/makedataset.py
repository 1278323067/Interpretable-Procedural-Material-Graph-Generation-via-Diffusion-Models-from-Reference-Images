import argparse
import os
from diffmat import MaterialGraphTranslator as MGT, config_logger
from diffmat.optim import Optimizer
from diffmat.core.io import read_image
from   BaseGraph import BaseGraph
from ParamWeigths import ParamWeights

from xml.etree import  ElementTree as ET
import pickle

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--MF_origin",type=bool, default= False)
    parser.add_argument("--outdir", type=str,default=r"D:\AI\diffmat\outtrash")

    args=parser.parse_args()
    if args.MF_origin:
        #D:\AI\diffmat\test\sbs\match_v1\1\1
        origin_path=r"D:\AI\diffmat\test\sbs\match_v1\1\1"
        sbs_num=0
        for foldername, folderlist, filelist in os.walk(origin_path):
             if ".autosave" not in foldername and "dependencies" not in foldername:
                 for file in filelist:
                    name,postfix=os.path.splitext(file)
                    if postfix==".sbs":

                        filepath=os.path.join(foldername,file)
                        print('---------------------------------------')
                        print(filepath)
                        print('---------------------------------------')
                        translator = MGT(filepath , res=8, external_noise=False)
                        graph = translator.translate(external_input_folder="", device='cuda')
                        graph.compile()

                        with graph.timer('Forward'):
                            outdir = os.path.join(args.outdir, str(sbs_num))

                            os.makedirs(outdir)
                            global_num = graph.custom_evaluate(benchmarking=False, output_path=outdir,filepath=filepath)
                            sbs_num+=1
                            print(f"sbs_num:{sbs_num}")
    else:
        origin_path = r"D:\material"
        init_weights = ParamWeights()

        for foldername, folderlist, filelist in os.walk(origin_path):
            if ".autosave" not in foldername and "dependencies" not in foldername:
                for file in filelist:
                    name, postfix = os.path.splitext(file)
                    if postfix == ".sbs":
                        filepath = os.path.join(foldername, file)
                        print('---------------------------------------')
                        print(filepath)
                        print('---------------------------------------')
                        root = ET.parse(filepath).getroot()
                        BaseGraph(root, init_weights, device="cpu")
        with open("paramOutput", "wb") as f:
            pickle.dump(init_weights.param_weights, f)


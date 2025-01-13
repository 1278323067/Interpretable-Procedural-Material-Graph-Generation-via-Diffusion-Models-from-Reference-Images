from xml.etree import  ElementTree as ET
from BaseNode import  BaseNode
from diffmat.translator.util import (
    is_image, is_optimizable, is_integer_optimizable, get_value, get_param_value,
    find_connections, NODE_CATEGORY_LUT, FACTORY_LUT,load_node_config_custom
)
from BaseParam import BaseParam
from ParamWeigths import  ParamWeights
import json
from diffmat.translator import param_trans
from diffmat.translator.base import BaseNodeTranslator, BaseParamTranslator
from xml.etree import ElementTree as ET
from typing import Dict, List, Tuple, Sequence, Optional, Type, Union
from diffmat.translator.types import NodeFunction, NodeConfig, Constant
from diffmat.core.base import BaseParameter
from diffmat.translator.function_trans import FunctionGraphTranslator

class BaseGraph():
    '''
    Only parse discrete values(int,str)
    param_weights : initial node parameters
    '''
    def __init__(self,root,init_weigths,device="cpu"):
        self.root=root
        self.init_weights=init_weigths
        self.param_translator = []
        self.node_config = {}
        self.device=device

        self.init_graph()



    def init_graph(self):

        for node in self.root.iter("compImplementation"):
            node_name = ""
            node = node[0]
            if node.tag == 'compInputBridge' or node.tag== 'compOutputBridge':
               continue
            elif node.tag == 'compFilter':
                node_type = node.find('filter').get('v')
            else:
                path = get_value(node.find('path'))
                node_type = path[path.rfind('/') + 1: path.rfind('?')]
            node_name=node_type
            print(node_name)

            if node_type not in NODE_CATEGORY_LUT or node_type in ["fxmaps", "bitmap", "svg", "pixelprocessor",
                                                                   "output"]:
                continue

            #nodes that  do not  need to collect parameters
            if node_type in ["gradient","curve"]:
                continue

            self.node_config=self._resolve_node_type(node_type)
            if self.node_config is None:
                continue
            self._init_param_translators(node)

            kwargs = {'device': self.device}
            for pt in self.param_translator:
                if isinstance(pt.function_trans, FunctionGraphTranslator):
                    pt.function_trans=None
                param= pt.translate_custom(**kwargs)
                new_param={pt.name:param}
                self.init_weights.update(node_name,new_param)

        ''' if node_config is not None and node_config.get("param"):
                params=self._init_weights(node_config)
                for param_et in node.iterfind('parameters/parameter'):
                    par_name=param_et.find('name').get('v')
                    if params.get(par_name) and param_et.find('.//dynamicValue') is  None:
                        baseparam=BaseParam(param_et,*params.get(par_name).values())
                        value=baseparam._calc_value()
                        for key ,value1 in params.get(par_name).items():
                            params[par_name][key]=value

                for param_et in node.iterfind('paramsArrays/paramsArray'):
                    par_name = param_et.find('name').get('v')
                    if params.get(par_name) and param_et.find('.//dynamicValue') is None:
                        baseparam = BaseParam(param_et, *params.get(par_name).values())

                        value = baseparam._calc_value()
                        for key, value1 in params.get(par_name).items():
                            params[par_name][key] = value

                for key,value in params.items():
                    self.init_weights.update(node_name,value)
            else:
                continue
'''


    def _resolve_node_type(self, node_type: str):
        '''
        get node yml config
        '''


        # log not implemented node
        if node_type not in NODE_CATEGORY_LUT :
            with open("./output.json","r+") as f:
                content=json.load(f)
                if node_type not in content["not_impl"]:
                    content["not_impl"].append(node_type)
                    f.seek(0)
                    json.dump(content,f)
            return None
        else:
            node_category = NODE_CATEGORY_LUT.get(node_type, 'generator')
        is_generator = node_category in ('dual', 'generator')
        is_fxmap_pp = node_category in ('pixelprocessor', 'fxmap')
        load_config_mode = 'generator' if is_generator else 'node'
        node_config = load_node_config_custom(node_type, mode=load_config_mode)    #get xml  file content
        return  node_config

    def _init_weights(self,node_config):
        '''
        get node initial weights
        '''
        params={}
        default_param = node_config["param"]
        for param in default_param:
            if  param.get("default") is not None or param.get("sbs_default") is not None:
                value = param["default"] if "default" in param else param["sbs_default"]
                params.update({param["sbs_name"]: {param["name"]: value}})
            else:
                params.update({param["sbs_name"]: {param["name"]: []}})
        return params

    """ if isinstance(value, bool) or isinstance(value, int) or isinstance(value, str):
                    params.update({param["sbs_name"]: {param["name"]: value}})
                elif isinstance(value, list) and len(value) in (2, 3, 4):
                    if isinstance(value[0], int):
                        params.update({param["sbs_name"]: {param["name"]: value}})
                else:
                    continue
                """   # whether or not to  preserve float type data


    def _init_param_translators(self,node_root):
        """Create parameter translators according to node configuration info.
        """
        self.param_translator.clear()

        # Skip if the node configuration doesn't specify any parameter info
        if not self.node_config.get('param'):
            return

        # Get node implementation element
        node_imp_et =node_root

        # Build lookup dictionary for parameter XML elements
        param_et_dict: Dict[str, ET.Element] = {}

        for param_et in node_imp_et.iterfind('parameters/parameter'):
            param_et_dict[param_et.find('name').get('v')] = param_et
        for param_et in node_imp_et.iterfind('paramsArrays/paramsArray'):
            param_et_dict[param_et.find('name').get('v')] = param_et

        # Read parameter translator configurations and construct translator objects
        for config in self.node_config['param']:
            #print(config["sbs_name"])
            # Get translator type and look up the factory for class name
            trans_type = config.get('type')
            if trans_type is None:
                trans_type = "default"
            root = None
            # ----------------------------------------------------------------------------------------------------
            if self.node_config['func'] == "tile_generator":
                import json
                with open(r"D:\AI\diffmat\diffmat\mywork\Makedataset\legacy_tile_generator.json") as f:
                    data = json.load(f)

                if not param_et_dict.get(config['sbs_name']):
                    root = param_et_dict.get(data.get(config['sbs_name']))
                else:
                    root = param_et_dict.get(config['sbs_name'])
            # ----------------------------------------------------------------------------------------------------
            else:
                root = param_et_dict.get(config['sbs_name'])
            # Delete irrelevant entries in parameter config
            kwargs: Dict[str, Constant] = config.copy()
            del kwargs['type']

            if trans_type == 'integer':

                kwargs.pop('quantize', None)
                if trans_type == 'constant':
                    kwargs.pop('scale', None)

            # Create the parameter translator
            trans_class: Type[BaseParamTranslator] = \
                getattr(param_trans, FACTORY_LUT['param_trans'][trans_type])
            self.param_translator.append(trans_class(root, **kwargs))



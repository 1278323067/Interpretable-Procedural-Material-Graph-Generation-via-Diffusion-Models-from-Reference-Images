from diffmat.translator.util import lookup_value_type,get_param_value,to_constant_custom,get_param_type
from diffmat.translator import types as tp

class BaseParam():
    def __init__(self,root,default_value,param_name=None):
        self.root=root
        self.param_name=param_name
        self.type = self.type = get_param_type(root) if root else \
                    lookup_value_type(default_value) if default_value is not None else \
                    lookup_value_type(default_value) if default_value is not None else 0

    def _calc_value(self):
        value_str = get_param_value(self.root, check_dynamic=False)
        if self.param_name == "gradientrgba":
            value=self._to_literal_gradient(value_str)
        elif self.param_name == "curveluminance":
            pass
        else:

           # if value_str is None:
            value = to_constant_custom(value_str, self.type)
        return value


    def _to_literal_gradient(self, value_str) :

        if not value_str:
            raise ValueError('The input cell array must not be empty')
        positions =  [to_constant_custom(cell['position'], tp.FLOAT) for cell in value_str]
        colors =  [to_constant_custom(cell['value'], tp.FLOAT4) for cell in value_str]
        return None
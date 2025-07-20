"""

"""

from collections import OrderedDict
import yaml
import copy
import sys

DEFAULT_CONFIG_PATH = None #'configs/default.yaml'

NoneType = type(None)
class ConfigProcessor:
    """
    Builds configuration from default config [configs/default.yaml] and command line arguments, including (optionally) a config file.
    """
    VALID_SCALAR_TYPES = (int, float, str, bool, type(None))
    NONALLOWED_KEYS = ('init', '__getitem__', '__str__', '__repr__', '__getattr__')
    
    def __init__(self):
        self._dict = None
        self._default_config_path = None
        self._params = None
        
    def init(self, default_config_path=DEFAULT_CONFIG_PATH, params=None, allow_new_keys=False, verbose=True):
        """
        If param is None, loads sys.argv[1:] as params.
        """
        assert self._dict is None, "Config is already initialized"
        self._default_config_path = default_config_path
        self._params = params
        self._allow_new_keys = allow_new_keys
        self._verbose = verbose
        if default_config_path:
            default_config_dict = self._load_config_file(default_config_path)
            self._dict = self._load_elements(default_config_dict, template_element=None, key_prefix='')._dict   
        config_dict = self._param_to_dict(params)
        self._dict = self._load_elements(config_dict, template_element=self._dict, key_prefix='')._dict
        self._remove_wildcard_keys(self._dict)
        return self
    
    def _safe_get_dict(self):
        assert self._dict is not None, "Config is not initialized"
        return self._dict
                
    def __contains__(self, key):
        return key in self._safe_get_dict()
    
    def __getitem__(self, key):
        """
        Allow using config as a dict: config[key] to get the value
        """
        return self._safe_get_dict()[key]

    def __getattr__(self, key):
        """
        Allow using config as a dict: config.key to get the value
        """
        
        if key in self.__dict__:
            return self.__dict__[key]
        if key != '_dict' and key in self._safe_get_dict():
            return self._dict[key]
        raise AttributeError(f"Key {key} not found in config")
    
    def __setattr__(self, key, value):
        if hasattr(self, "_dict") and self._dict and key in self._dict:
            raise TypeError(f"Attributes of ConfigProcessor can not be reassigned. Attempt to assign attribute '{key}'")
        else:
            super().__setattr__(key, value)
    
    def get(self, key, default=None):  # TODO убрать, когда вычищу обращения к get()
        if key in self._safe_get_dict():
            return self._dict[key]
        return default
        
    def keys(self):
        return self._safe_get_dict().keys()
    
    def values(self):
        return self._safe_get_dict().values()
    
    def items(self):
        return self._safe_get_dict().items()
    
    def _param_to_dict(self, param):
        if param is None:
            return self._param_list_to_dict(sys.argv[1:])
        elif isinstance(param, list):
            return self._param_list_to_dict(param)
        elif isinstance(param, str):
            return self._param_list_to_dict(param.split(' '))
        elif isinstance(param, dict):
            return param
        raise ValueError(f"Invalid parameter: {param}")
    
    def _param_list_to_dict(self, param_list):
        """
        Convert command line arguments to a dictionary, including nested keys.
        """
        if self._verbose:
            print('ConfigProcessor._param_list_to_dict: loading', param_list)
        config_dict = {}
        waiting_for_value = False
        target_dict = None
        target_key = None
        for param in param_list:
            if param == '': # it can happen if there are no arguments or if there is more than one consecutive space in the command line.
                continue
            if waiting_for_value and not param.startswith('--'):
                assert target_dict is not None and target_key is not None, f"Value '{param}' is expected after key starting with '--'"
                value = param
                waiting_for_value = False
            else:
                waiting_for_value = False
                if param.startswith('--'):
                    waiting_for_value = True
                    param = param[2:]
                if '=' in param:
                    waiting_for_value = False
                    key, value = param.split('=')
                else:
                    key = param
                    value = 'True'
                key = key.strip().replace('-', '_')
                if ':' in key:
                    key, key_last = key.split(':',1)
                    keys = key.split('.') + [key_last]
                else:
                    keys = key.split('.')
                target_dict = config_dict
                for k in keys[:-1]:
                    if k not in target_dict:
                        target_dict[k] = {}
                    target_dict = target_dict[k]
                target_key = keys[-1]
            target_dict[target_key] = yaml.safe_load(value) # interpret value types similar to file loading
        return config_dict

    def _load_elements(self, config_element, template_element, key_prefix=''):
        """
        template_element: element to check keys of new element. Is used to check keys for list of dicts.
        key_prefix: full path to the key in the config. Is used only for error messages.
        """
        if isinstance(config_element, (dict, ConfigProcessor)):
            return self._load_elements_dict(config_element, template_element, key_prefix)
        elif isinstance(config_element, (list, tuple)):
            return self._load_elements_list(config_element, template_element, key_prefix)
        else:
            return self._load_elements_scalar(config_element, template_element, key_prefix)
    
    def _load_elements_dict(self, config_dict, template_element, key_prefix=''):
        """
        template_element: element to check keys of new element. Is used to check keys for list of dicts.
        key_prefix: full path to the key in the config. Is used only for error messages.
        """
        assert isinstance(template_element, (NoneType, dict, ConfigProcessor)), f"Invalid config type: {type(config_dict)} for key: {key_prefix}. {type(template_element)} is expected"
        if isinstance(template_element, ConfigProcessor):
            template_element = template_element._dict
        assert isinstance(config_dict, (dict, ConfigProcessor)), f"Invalid config type: {type(config_dict)} for key: {key_prefix}. Dict is expected"
        if isinstance(config_dict, ConfigProcessor):
            config_dict = config_dict._dict
        if isinstance(config_dict, dict) and 'config' in config_dict:
            base_config_path = config_dict['config']
            base_config_dict = self._load_config_file(base_config_path)
            result = self._load_elements(base_config_dict, template_element, key_prefix)
            template_element = result._dict
        else:
            result = ConfigProcessor() 
            result._dict = dict()

        if template_element:
            for key, template_value in template_element.items():
                value = config_dict.get(key, template_value)
                result._dict[key] = self._load_elements(value, template_value, key_prefix=f"{key_prefix}.{key}")
        # Check new keys.
        for key, value in config_dict.items():
            assert key not in self.NONALLOWED_KEYS, f"Key {key} is not allowed"
            if key not in result._dict.keys():
                template_value = None
                if template_element and key != 'config':
                    if '...' in template_element.keys():
                        template_value = template_element['...']
                    else:
                        assert self._allow_new_keys,  f"Key {key_prefix}.{key} not found in default config"
                result._dict[key] = self._load_elements(value, template_value, key_prefix=f"{key_prefix}.{key}")
        return result
    
    def _load_elements_list(self, config_list, template_element, key_prefix=''):
        """
        template_element: element to check keys of new element. Is used to check keys for list of dicts.
        key_prefix: full path to the key in the config. Is used only for error messages.
        """
        if template_element is not None:
            assert isinstance(template_element, list), f"Invalid config type: {type(config_list)} for key: {key_prefix}. {type(template_element)} is expected"
            template_element = None # TODO: add support for templates for list elements
        result = list()

        for i, v in enumerate(config_list):
            new_element = self._load_elements(v, template_element=template_element, key_prefix=f"{key_prefix}[{i}].")
            result.append(new_element)
        return result

    def _load_elements_scalar(self, config_value, template_element, key_prefix=''):
        """
        template_element: element to check keys of new element. Is used to check keys for list of dicts.
        key_prefix: full path to the key in the config. Is used only for error messages.
        """
        if template_element is not None:
            assert isinstance(template_element, self.VALID_SCALAR_TYPES), f"Invalid config type: {type(config_value)} for key: {key_prefix}. {type(template_element)} is expected"
        assert isinstance(config_value, self.VALID_SCALAR_TYPES), f"Invalid type: {type(config_value)} for key: {key_prefix}. Scalar is expected"
        return config_value
                
    def _remove_wildcard_keys(self, dict_to_process=None):
        if isinstance(dict_to_process, ConfigProcessor):
            dict_to_process = dict_to_process._dict
        if isinstance(dict_to_process, dict):
            dict_to_process.pop('...', None)
            for v in dict_to_process.values():
                self._remove_wildcard_keys(v)
        elif isinstance(dict_to_process, (list, tuple)):
            for v in dict_to_process:
                self._remove_wildcard_keys(v)

    def _load_config_file(self, config_path):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return config_dict
    
    def dump(self, indent=2, sort_keys=False, default_flow_style=False, **kwargs):
        if self._dict is None:
            return 'Uninitialized'
        s = yaml.dump(self._safe_get_dict(), default_flow_style=default_flow_style, sort_keys=sort_keys, indent=indent, **kwargs)
        s = s.replace(': null\n', ':\n')
        return s
    
    def __str__(self):
        if self._dict is None:
            return 'ConfigProcessor: Uninitialized'
        return 'ConfigProcessor:' + self._dict.__str__()
    
    def __repr__(self):
        if self._dict is None:
            return '<ConfigProcessor>: Uninitialized'
        return '<ConfigProcessor>:' + self._dict.__str__()

    def to_dict(self):
        def _to_dict(cfg):
            if isinstance(cfg, ConfigProcessor):
                return _to_dict(cfg._dict)
            elif isinstance(cfg, dict):
                return {k: _to_dict(v) for k, v in cfg.items()}
            elif isinstance(cfg, list):
                return [_to_dict(v) for v in cfg]
            else:
                return cfg
        return _to_dict(self)

# Register ConfigProcessor with PyYAML
def config_processor_representer(dumper, data):
    return dumper.represent_dict(data._safe_get_dict())

yaml.add_representer(ConfigProcessor, config_processor_representer)

config = ConfigProcessor()

if __name__ == '__main__':
    config.init(default_config_path='../configs/default.yaml', 
                params='config=../configs/config_E300_one_batch.yaml model.tokenizer=new_model model.use_tsm name=SLT'
                )
    print(config)
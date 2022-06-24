import os
import yaml

class Config(dict):
    def __init__(self, init=None):
        super().__init__()
        if init is None:
            init = Config.get_defaults()
        self.update(init)
        if 'multi_slice_label' not in self.__dict__:
            self['multi_slice_label'] = False
        if 'res_units' not in self.__dict__:
            self['res_units'] = 2
        if 'restart_iter' not in self.__dict__:
            self['restart_iter'] = 100
        if 'dropout' not in self.__dict__:
            self['dropout'] = 0.2
        if 'tta_flip_dims' not in self.__dict__:
            self['tta_flip_dims'] = [[], [2], [3], [2, 3]]

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, val):
        self.__dict__[key] = val

    def __str__(self):
        return yaml.dump(self.__dict__)

    def as_dict(self):
        return self.__dict__

    def update(self, init):
        for key in init:
            self[key] = init[key]

    @staticmethod
    def get_defaults():
        conf_file = '../code/config.yaml'
        assert os.path.exists(conf_file)
        with open(conf_file) as file:
            return yaml.safe_load(file)

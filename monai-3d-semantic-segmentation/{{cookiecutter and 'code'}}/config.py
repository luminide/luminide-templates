import os
import yaml

class Config():
    def __init__(self, init=None):
        if init is None:
            init = Config.get_defaults()
        object.__setattr__(self, "_params", dict())
        self.update(init)
        if 'multi_slice_label' not in self._params:
            self['multi_slice_label'] = False
        if 'res_units' not in self._params:
            self['res_units'] = 2
        if 'restart_iter' not in self._params:
            self['restart_iter'] = 100
        if 'dropout' not in self._params:
            self['dropout'] = 0.2
        if 'tta_flip_dims' not in self._params:
            self['tta_flip_dims'] = [[], [2], [3], [2, 3]]

    def __getitem__(self, key):
        return self._params[key]

    def __setitem__(self, key, val):
        self._params[key] = val

    def __getattr__(self, key):
        return self._params[key]

    def __str__(self):
        return yaml.dump(self._params)

    def as_dict(self):
        return self._params

    def update(self, init):
        for key in init:
            self[key] = init[key]

    @staticmethod
    def get_defaults():
        conf_file = '../code/config.yaml'
        assert os.path.exists(conf_file)
        with open(conf_file) as file:
            return yaml.safe_load(file)

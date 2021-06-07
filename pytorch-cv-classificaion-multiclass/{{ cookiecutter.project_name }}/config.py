hp_dict = dict(
    # this can be any network from the timm library
    arch = 'resnet18',

    pretrained = True,

    crop_width = 384,
    crop_height = 384,
    # images will be resized to margin*4% larger than crop size
    margin = 3,

    aug_prob = 0.75,
    strong_aug = True,

    # optimizer settings
    lr = 0.0001,
    momentum = 0,
    nesterov = False,

    # scheduler settings
    gamma = 0.96,
)

class Config(object):
    def __init__(self, hp_dict):
        object.__setattr__(self, "_params", dict())
        for key in hp_dict:
            self[key] = hp_dict[key]

    def __getitem__(self, key):
        return self._params[key]

    def __setitem__(self, key, val):
        self._params[key] = val

    def __getattr__(self, key):
        return self._params[key]

    def get(self):
        return self._params

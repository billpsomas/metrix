from .cars import Cars
from .cub import CUBirds
from .SOP import SOP
from .Inshop import Inshop_Dataset
from .import utils
from .base import BaseDataset


_type = {
    'cars': Cars,
    'cub': CUBirds,
    'sop': SOP,
    'inshop': Inshop_Dataset
}

def load(name, root, mode, transform = None):
    return _type[name](root = root, mode = mode, transform = transform)
    

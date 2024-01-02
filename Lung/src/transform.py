import numpy as np
from scipy import ndimage

import torch
import torch.nn.functional as F

        
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, patch):
        if patch.dtype!=np.float32:
            patch = patch.astype('float32')
        return torch.from_numpy(patch)
        


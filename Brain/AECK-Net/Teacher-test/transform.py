# -*- coding: utf-8 -*-

import numpy as np
import torch

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, patch):
        if patch.dtype!=np.float32:
            patch = patch.astype('float32')
        return torch.from_numpy(patch)

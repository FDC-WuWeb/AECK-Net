import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Warper3d(nn.Module):
    def __init__(self, img_size):
        super(Warper3d, self).__init__()

        self.img_size = img_size

        H, W, D =img_size

        xx = torch.arange(0, H).view(-1, 1, 1).repeat(1, W, D)
        yy = torch.arange(0, W).view(1, -1, 1).repeat(H, 1, D)
        zz = torch.arange(0, D).view(1, 1, -1).repeat(H, W, 1)

        self.grid = np.zeros([1, H, W, D, 3])

        xx1 = 2.0 * xx / H - 1  # max(W-1,1)
        yy1 = 2.0 * yy / W - 1  # max(H-1,1)
        zz1 = 2.0 * zz / D - 1  # max(H-1,1)

        self.grid[0,:,:,:,0] = zz1
        self.grid[0, :, :, :, 1] = yy1
        self.grid[0, :, :, :, 2] = xx1

    def forward(self, img, flow):

        grid = self.grid
        grid = torch.tensor(grid)
        grid = grid.to(torch.float32)

        if img.is_cuda:
            grid = grid.cuda()

        flow = flow.permute(0, 2, 3, 4, 1)  # [bs, D, H, W, 3]
        vgrid = grid + flow
        output = F.grid_sample(img, vgrid, padding_mode='border', align_corners=True)
        return output

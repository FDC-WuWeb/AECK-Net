#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
from pathlib import Path
#import matplotlib.pyplot as plt
import torch

import os
import matplotlib.pylab as plt
import torchvision.transforms as transforms

from itertools import chain
import torch.optim as optim
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def Get_Jac(displacement):
    '''
    the expected input: displacement of shape(batch, H, W, D, channel),
    obtained in TensorFlow.
    '''
    D_y = (displacement[:, 1:, :-1, :-1, :] - displacement[:, :-1, :-1, :-1, :])
    D_x = (displacement[:, :-1, 1:, :-1, :] - displacement[:, :-1, :-1, :-1, :])
    D_z = (displacement[:, :-1, :-1, 1:, :] - displacement[:, :-1, :-1, :-1, :])

    D1 = (D_x[..., 0] + 1) * ((D_y[..., 1] + 1) * (D_z[..., 2] + 1) - D_z[..., 1] * D_y[..., 2])
    D2 = (D_x[..., 1]) * (D_y[..., 0] * (D_z[..., 2] + 1) - D_y[..., 2] * D_z[..., 0])
    D3 = (D_x[..., 2]) * (D_y[..., 0] * D_z[..., 1] - (D_y[..., 1] + 1) * D_z[..., 0])

    D = D1 - D2 + D3

    return D


img_size = [256, 256, 128]
TRE_ENH, TRE_ORG, TRE_CAD, TRE_final = [], [], [], [10]
timelist = []
mean_tot, std_tot = 0, 0
tre_all = {}
for case in range(1, 11):
    spacing_arr = [[0.97, 0.97, 2.5], [1.16, 1.16, 2.5], [1.15, 1.15, 2.5], [1.13, 1.13, 2.5], [1.10, 1.10, 2.5],
                   [0.97, 0.97, 2.5], [0.97, 0.97, 2.5], [0.97, 0.97, 2.5], [0.97, 0.97, 2.5], [0.97, 0.97, 2.5]]
    spacing = np.array(spacing_arr[case - 1])
    lmk_path = './mark/Case%gPack/ExtremePhases/' % case
    mov_lmk_fname = 'Case%g_300_T00_xyz.txt' % case
    ref_lmk_fname = 'Case%g_300_T50_xyz.txt' % case

    #Teacher
    flow_folder = '../flow/AECK-Net/Teacher/'
    flow_fname = 'TeacherCase%g.npy' % case

    # #Student
    # flow_folder = '../flow/AECK-Net/Student/'
    # flow_fname = 'StudentCase%g.npy' % case

    # #Single-level
    # flow_folder = '../flow/AECK-Net/Single-level/'
    # flow_fname = 'SingleCase%g.npy' % case

    # #VXM-diff
    # flow_folder = '../flow/VoxelMorph-diff/'
    # flow_fname = 'VXMdiffCase%g.npy' % case

    flow = np.load(flow_folder + flow_fname)

    #check jacobian
    jac = Get_Jac(np.transpose(flow, (0, 2, 3, 4, 1)))
    negative_elements_count = np.sum(jac < 0)

    flow = torch.Tensor(flow)
    H, W, D = img_size[0], img_size[1], img_size[2]
    xx = torch.arange(0, H).view(-1, 1, 1).repeat(1, W, D)
    yy = torch.arange(0, W).view(1, -1, 1).repeat(H, 1, D)
    zz = torch.arange(0, D).view(1, 1, -1).repeat(H, W, 1)
    xx = 2.0 * xx / H - 1
    yy = 2.0 * yy / W - 1
    zz = 2.0 * zz / D - 1
    xx_f = torch.zeros([H, W, D])
    yy_f = torch.zeros([H, W, D])
    zz_f = torch.zeros([H, W, D])
    xx_f[:, :, :] = flow[0, 2, :, :, :]
    yy_f[:, :, :] = flow[0, 1, :, :, :]
    zz_f[:, :, :] = flow[0, 0, :, :, :]
    xx_f = xx + xx_f
    yy_f = yy + yy_f
    zz_f = zz + zz_f
    xx_f = (xx_f + 1) * H / 2
    yy_f = (yy_f + 1) * W / 2
    zz_f = (zz_f + 1) * D / 2

    mov_lmk = np.loadtxt(lmk_path + mov_lmk_fname)
    ref_lmk = np.loadtxt(lmk_path + ref_lmk_fname)
    offset_arr = [[4, 40, 2], [1, 17, 3], [0, 37, 1], [0, 37, 2], [6, 47, 4], [112, 135, 13], [103, 127, 15],
                  [100, 70, 15], [126, 127, 2], [141, 113, 2]]
    resize_axis = [[257, 167, 92], [261, 195, 105], [260, 178, 99], [257, 176, 94], [246, 180, 97],
                   [327, 198, 104], [342, 208, 110], [307, 237, 113], [266, 202, 74], [249, 225, 101]]
    a = H / resize_axis[case - 1][0]
    b = W / resize_axis[case - 1][1]
    c = D / resize_axis[case - 1][2]
    resize_factor = np.array([a, b, c])
    mov_lmk0 = np.zeros([300, 3])
    ref_lmk0 = np.zeros([300, 3])

    mov_lmk0[:, 0] = (mov_lmk[:, 0] - offset_arr[case - 1][0] - 1) * resize_factor[0]
    mov_lmk0[:, 1] = (mov_lmk[:, 1] - offset_arr[case - 1][1] - 1) * resize_factor[1]
    mov_lmk0[:, 2] = ((mov_lmk[:, 2] - offset_arr[case - 1][2] - 1) * resize_factor[2])
    ref_lmk0[:, 0] = (ref_lmk[:, 0] - offset_arr[case - 1][0] - 1) * resize_factor[0]
    ref_lmk0[:, 1] = (ref_lmk[:, 1] - offset_arr[case - 1][1] - 1) * resize_factor[1]
    ref_lmk0[:, 2] = ((ref_lmk[:, 2] - offset_arr[case - 1][2] - 1) * resize_factor[2])
    ref_lmk_index = np.round(ref_lmk0).astype('int32')
    ref_lmk1 = ref_lmk0.copy()
    ref_lmk_index1 = np.zeros([300, 3])
    ref_lmk_index1 = ref_lmk_index
    for i in range(300):
        hi, wi, di = ref_lmk_index[i]
        h0 = xx_f[hi, wi, di]
        w0 = yy_f[hi, wi, di]
        d0 = zz_f[hi, wi, di]
        ref_lmk1[i] = [h0, w0, d0]
    spacing1 = spacing
    spacing1 = spacing / resize_factor
    factor1 = np.ones([300, 3])
    factor1 = ref_lmk_index1 / ref_lmk1
    ref_lmk1_xs = (ref_lmk0 - ref_lmk_index1) * factor1
    diff1 = (ref_lmk0 - mov_lmk0) * spacing1
    diff1 = torch.Tensor(diff1)
    tre1 = diff1.pow(2).sum(1).sqrt()
    mean1 = tre1.mean()
    std1 = tre1.std()
    diff1 = (ref_lmk1 - mov_lmk0 + ref_lmk1_xs) * spacing1
    diff1 = torch.Tensor(diff1)
    tre1 = diff1.pow(2).sum(1).sqrt()
    mean1 = tre1.mean()
    std1 = tre1.std()
    mean_tot += mean1
    std_tot += std1
    TRE_ENH.append(mean1)
    print('TRE case%g' % case, mean1, '    case%g' % case, std1)
    print(f'Num of elements in jac less than zero: {negative_elements_count}')
    print('Range of values for Jac: {:.2f}, {:.2f}\n'.format(round(jac.min(), 2), round(jac.max(), 2)))

mean_tot = mean_tot / 10
std_tot = std_tot / 10
print('TRE mean_tot', mean_tot, '    std_tot', std_tot)

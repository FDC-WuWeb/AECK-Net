#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import dataset
import model
import transform
import utils
"""
Mindboggle student test
"""

torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser(description='Testing codes')
parser.add_argument('-v', '--val', default=8, type=int,
                    help='the case index of validation')
parser.add_argument('-b', '--batch', default=1, type=int,
                    help='batch size')
parser.add_argument('-i', '--image', default=[256, 256, 128], type=int,
                    help='image size')
parser.add_argument('-p', '--pretrained_model', default=True, type=bool,
                    help='pretrained model')

args = parser.parse_args()

WEIGHTS_PATH = 'weights-adam/'
WEIGHTS_NAMECAD = 'StudentCADKD0.680.pth'
WEIGHTS_NAMEORG = 'StudentORGKD0.680.pth'
WEIGHTS_NAMEENH = 'StudentENHKD0.680.pth'

root = r'..\..\Data\Mindboggle/'
root2 = './dirlab_32x/'
Transform = transforms.Compose([
                                transform.ToTensor()])

test_dset = dataset.Volumes(root,root2, transform=Transform)
test_loader = data.DataLoader(test_dset, args.batch, shuffle=False)

img_size=args.image

modelCAD = model.snetCAD(img_size=args.image).cuda()
modelORG = model.snetORG(img_size=args.image).cuda()
modelENH = model.snetENH(img_size=args.image).cuda()

if args.pretrained_model:
    startEpochCAD = utils.load_weights(modelCAD, WEIGHTS_PATH + WEIGHTS_NAMECAD)
    startEpochORG = utils.load_weights(modelORG, WEIGHTS_PATH + WEIGHTS_NAMEORG)
    startEpochENH = utils.load_weights(modelENH, WEIGHTS_PATH + WEIGHTS_NAMEENH)

losses = []

### test ###
utils.test(modelCAD, modelORG, modelENH, test_loader)




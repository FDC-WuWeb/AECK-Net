#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import itertools as it
import numpy as np
from functools import reduce

import torch
import torch.utils.data as data

class Volumes(data.Dataset):

    def __init__(self, root, root2,transform=None):
        self.root = root
        self.root2 = root2
        self.transform = transform
        self.dset_list_name = 'test.pth'
        if os.path.exists(root2 + self.dset_list_name):
            image_list = torch.load(root2 + self.dset_list_name)
            self.image_list = image_list['test']
        else:
            self.image_list = self._make_dataset()

    def __getitem__(self, index):
        pairs = self.image_list[index]
        movCAD = np.load(pairs[0])
        refCAD = np.load(pairs[1])

        movORG = np.load(pairs[2])
        refORG = np.load(pairs[3])

        movENH = np.load(pairs[4])
        refENH = np.load(pairs[5])

        movCAD = np.expand_dims(movCAD, 0)  # shape(1, D, H, W)
        refCAD = np.expand_dims(refCAD, 0)

        movORG = np.expand_dims(movORG, 0)  # shape(1, D, H, W)
        refORG = np.expand_dims(refORG, 0)

        movENH = np.expand_dims(movENH, 0)  # shape(1, D, H, W)
        refENH = np.expand_dims(refENH, 0)

        if self.transform is not None:
            movCAD = self.transform(movCAD)
            refCAD = self.transform(refCAD)

            movORG = self.transform(movORG)
            refORG = self.transform(refORG)

            movENH = self.transform(movENH)
            refENH = self.transform(refENH)
        return movCAD, refCAD, movORG, refORG, movENH, refENH

    def __len__(self):
        return len(self.image_list)

    def _make_dataset(self):

        samples_test,samples_testCAD, samples_testORG,samples_testENH = [],[],[],[]

        for dataroot in ["test"]:

            if dataroot == "test":
                path_CAD = self.root + "/CAD/CADnpy/" + dataroot + '/'
                dcm_list = os.listdir(path_CAD)
                for pairs in it.permutations(dcm_list, 2):
                    mov_fname = path_CAD + pairs[0]
                    ref_fname = path_CAD + pairs[1]
                    sample = [mov_fname, ref_fname]
                    samples_testCAD.append(sample)

                path_ORG = self.root + "/seg/segnpy/" + dataroot + '/'
                dcm_list = os.listdir(path_ORG)
                for pairs in it.permutations(dcm_list, 2):
                    mov_fname = path_ORG + pairs[0]
                    ref_fname = path_ORG + pairs[1]
                    sample = [mov_fname, ref_fname]
                    samples_testORG.append(sample)

                path_ENH = self.root + "/HistAdap/HistAdapnpy/" + dataroot + '/'
                dcm_list = os.listdir(path_ENH)
                for pairs in it.permutations(dcm_list, 2):
                    mov_fname = path_ENH + pairs[0]
                    ref_fname = path_ENH + pairs[1]
                    sample = [mov_fname, ref_fname]
                    samples_testENH.append(sample)

                for i in range(1, 133):
                    fusion_train = []
                    fusion_train.append(samples_testCAD[i - 1])
                    fusion_train.append(samples_testORG[i - 1])
                    fusion_train.append(samples_testENH[i - 1])
                    result = reduce(lambda x, y: x.extend(y) or x, fusion_train)
                    samples_test.append(result)


        samples = {'test': samples_test}
        print("test", samples_test)
        torch.save(samples, self.root2 + self.dset_list_name)
        return samples_test

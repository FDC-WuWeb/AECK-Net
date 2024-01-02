import os
import numpy as np
import torch
import torch.utils.data as data

class Volumes(data.Dataset):
    """
    create a data set of 3D CT image saved in .npy format.
    :param:
        str root: the folder of image files;
        int case_id: the index of case for the validation; 1-10
        train: if True(default), load  images of 9 cases 
            by taking random timepoints per patient as fi
xed and moving images;
            else, load images of the remaining one case; 
        transform: transform the images 
    """

    def __init__(self, root, root2, roottest, case_id=1, train=True, transform=None):
        self.root = root
        self.root2 = root2
        self.roottest = roottest
        self.case_id = case_id
        self.train = train
        self.transform = transform
        self.dset_list_name = 'train_val_list_case%g.pth' % case_id  #
        if os.path.exists(root2 + self.dset_list_name):
            image_list = torch.load(root2 + self.dset_list_name)
            self.image_list = image_list['train' if self.train else 'val']
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
        samples_train, samples_val = [], []

        for dataroot in ["COPD", "Creates", "Dirlab", "Spare"]:
            if dataroot == "Dirlab":
                for index in [6,7,8]:
                    case = 'case%g' % index

                    path_valCAD = self.root + "/segCADnpy/" + dataroot + '/'
                    mov_fname = path_valCAD + case + '_T00.npy'
                    ref_fname = path_valCAD + case + '_T50.npy'
                    sampleCAD = [mov_fname, ref_fname]
                    sampleCAD2 = [ref_fname, mov_fname]

                    path_valORG = self.root + "/segnpy/" + dataroot + '/'
                    mov_fname = path_valORG + case + '_T00.npy'
                    ref_fname = path_valORG + case + '_T50.npy'
                    sampleORG = [mov_fname, ref_fname]
                    sampleORG2 = [ref_fname, mov_fname]

                    path_valENH = self.root + "/vessel77npy/" + dataroot + '/'
                    mov_fname = path_valENH + case + '_T00.npy'
                    ref_fname = path_valENH + case + '_T50.npy'
                    sampleENH = [mov_fname, ref_fname]
                    sampleENH2 = [ref_fname, mov_fname]

                    sample = sampleCAD + sampleORG + sampleENH
                    sample2 = sampleCAD2 + sampleORG2 + sampleENH2
                    samples_val.append(sample)
                    samples_val.append(sample2)

            elif dataroot == "COPD":
                for index in range(1, 11):
                    case = 'case%g' % index

                    pathCAD = self.root + "/segCADnpy/" + dataroot + '/'
                    mov_fname = pathCAD + case + '_T00.npy'
                    ref_fname = pathCAD + case + '_T50.npy'
                    sampleCAD = [mov_fname, ref_fname]
                    sampleCAD2 = [ref_fname, mov_fname]

                    pathORG = self.root + "/segnpy/" + dataroot + '/'
                    mov_fname = pathORG + case + '_T00.npy'
                    ref_fname = pathORG + case + '_T50.npy'
                    sampleORG = [mov_fname, ref_fname]
                    sampleORG2 = [ref_fname, mov_fname]

                    pathENH = self.root + "/vessel77npy/" + dataroot + '/'
                    mov_fname = pathENH + case + '_T00.npy'
                    ref_fname = pathENH + case + '_T50.npy'
                    sampleENH = [mov_fname, ref_fname]
                    sampleENH2 = [ref_fname, mov_fname]

                    sample = sampleCAD + sampleORG + sampleENH
                    sample2 = sampleCAD2 + sampleORG2 + sampleENH2

                    samples_train.append(sample)
                    samples_train.append(sample2)

            elif dataroot == "Creates":

                for index in range(1, 7):
                    case = 'case%g' % index
                    pathCAD = self.root + "/segCADnpy/" + dataroot + '/' + case + '/'

                    mov_fname = pathCAD + case + '_T00.npy'
                    ref_fname = pathCAD + case + '_T50.npy'
                    sampleCAD = [mov_fname, ref_fname]
                    sampleCAD2 = [ref_fname, mov_fname]

                    pathORG = self.root + "/segnpy/" + dataroot + '/' + case + '/'
                    mov_fname = pathORG + case + '_T00.npy'
                    ref_fname = pathORG + case + '_T50.npy'
                    sampleORG = [mov_fname, ref_fname]
                    sampleORG2 = [ref_fname, mov_fname]

                    pathENH = self.root + "/vessel77npy/" + dataroot + '/' + case + '/'
                    mov_fname = pathENH + case + '_T00.npy'
                    ref_fname = pathENH + case + '_T50.npy'
                    sampleENH = [mov_fname, ref_fname]
                    sampleENH2 = [ref_fname, mov_fname]

                    sample = sampleCAD + sampleORG + sampleENH
                    sample2 = sampleCAD2 + sampleORG2 + sampleENH2
                    samples_train.append(sample)
                    samples_train.append(sample2)

            elif dataroot == "Spare":

                for index in range(1, 23):
                    case = 'case%g' % index
                    pathCAD = self.root + "/segCADnpy/" + dataroot + '/' + case + '/'

                    mov_fname = pathCAD + case + '_T00.npy'
                    ref_fname = pathCAD + case + '_T50.npy'
                    sampleCAD = [mov_fname, ref_fname]
                    sampleCAD2 = [ref_fname, mov_fname]

                    pathORG = self.root + "/segnpy/" + dataroot + '/' + case + '/'
                    mov_fname = pathORG + case + '_T00.npy'
                    ref_fname = pathORG + case + '_T50.npy'
                    sampleORG = [mov_fname, ref_fname]
                    sampleORG2 = [ref_fname, mov_fname]

                    pathENH = self.root + "/vessel77npy/" + dataroot + '/' + case + '/'
                    mov_fname = pathENH + case + '_T00.npy'
                    ref_fname = pathENH + case + '_T50.npy'
                    sampleENH = [mov_fname, ref_fname]
                    sampleENH2 = [ref_fname, mov_fname]

                    sample = sampleCAD + sampleORG + sampleENH
                    sample2 = sampleCAD2 + sampleORG2 + sampleENH2

                    samples_train.append(sample)
                    samples_train.append(sample2)

        samples = {'train': samples_train,
                   'val': samples_val}
        torch.save(samples, self.root2 + self.dset_list_name)

        return samples_train if self.train else samples_val

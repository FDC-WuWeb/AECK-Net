import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import argparse
from pathlib import Path
import torch
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import dataset
import model_aspp
import transform
import utils
import loss
from itertools import chain

torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser(description='Training codes')
parser.add_argument('-v', '--val', default=8, type=int,
                    help='the case index of validation')
parser.add_argument('-b', '--batch', default=1, type=int,
                    help='batch size')
parser.add_argument('-l', '--lr', default=0.0001, type=float,
                    help='learning rate')
parser.add_argument('-e', '--epoch', default=500, type=int,
                    help='training epochs')
parser.add_argument('-d', '--lamb', default=0.2, type=float,
                    help='lambda, balance the losses.')
parser.add_argument('-w', '--win', default=[5,5,5], type=int,
                    help='window size, in the LCC loss')
parser.add_argument('-i', '--image', default=[256, 256, 128], type=int,
                    help='image size')
parser.add_argument('-p', '--pretrained_model', default=True, type=bool,
                    help='pretrained model')

args = parser.parse_args()
en = 0
att = 0
optim_label = 'adam'
loss_label = optim_label + '-val%g-bs%g-lr%.4f-lamb%g-win%g-epoch%g'%(
        args.val, args.batch, args.lr, args.lamb, args.win[0], args.epoch)

##pretrained mode
WEIGHTS_MODEL_PATH = './weights-adam/'

WEIGHTS_NAMECAD = 'weights-val8_plus-50-0.071-0.137CAD1.53.pth'
WEIGHTS_NAMEORG = 'weights-val8_plus-50-0.071-0.137ORG1.53.pth'
WEIGHTS_NAMEENH = 'weights-val8_plus-50-0.071-0.137ENH1.53.pth'

WEIGHTS_PATH = 'weights-adam/'
Path(WEIGHTS_PATH).mkdir(exist_ok=True)
LOSSES_PATH = 'losses/'
Path(LOSSES_PATH).mkdir(exist_ok=True)
RESULTS_PATH = 'results/creatis_flow/'
'''log file'''
f = open(WEIGHTS_PATH + 'README.txt', 'w')

root = r'C:\Data\merge\cropwhole/'
root2 = './dirlab_32x/'
roottest = r'C:\Data\merge\cropwhole/'
Transform = transforms.Compose([
                                transform.ToTensor()])
train_dset = dataset.Volumes(root,root2, roottest, args.val, train=True, transform=Transform)
val_dset = dataset.Volumes(root,root2, roottest, args.val, train=False, transform=Transform)
train_loader = data.DataLoader(train_dset, args.batch, shuffle=False)
val_loader = data.DataLoader(val_dset, args.batch, shuffle=False)

print("Train dset: %d" %len(train_loader))
print("Val dset: %d" %len(val_loader))

img_size=args.image

'''Train'''
modelCAD = model_aspp.snetCAD(img_size=args.image).cuda()
modelORG = model_aspp.snetORG(img_size=args.image).cuda()
modelENH = model_aspp.snetENH(img_size=args.image).cuda()
modelD = model_aspp.snetD(img_size=args.image).cuda()

optimizer = optim.Adam(params=chain(modelCAD.parameters(),modelORG.parameters(),modelENH.parameters()), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9,
                                                       patience=5, verbose=False, threshold=0.0001,
                                                       threshold_mode='rel', cooldown=0, min_lr=0.00001, eps=1e-08)

if args.pretrained_model:
    startEpochCAD = utils.load_weights(modelCAD, optimizer, WEIGHTS_PATH + WEIGHTS_NAMECAD)
    startEpochORG = utils.load_weights(modelORG, optimizer, WEIGHTS_PATH + WEIGHTS_NAMEORG)
    startEpochENH = utils.load_weights(modelENH, optimizer, WEIGHTS_PATH + WEIGHTS_NAMEENH)
val = args.val

'''lcc + grad'''
criterion = {'lcc': loss.LCC(args.win).cuda(),
             'ncc': loss.NCC(5).cuda(),
             'mse': torch.nn.MSELoss().cuda(),
             'lambda': args.lamb,
             'adver': torch.nn.BCEWithLogitsLoss().cuda(),}
losses = []
for epoch in range(1, args.epoch+1):

    since = time.time()
#    scheduler.step()# adjust lr
    ### Train ###
    trn_loss, trn_similarity, trn_regularization, trn_adversary, trn_mae = utils.train(
        modelCAD, modelORG, modelENH, modelD, train_loader, optimizer, criterion)
    print('Epoch {:d}\nTrain - Loss: {:.4f} | Simi: {:.4f} | Regular: {:.4f} | Adver: {:.4f} | MAE: {:.4f}'.format(
        epoch, trn_loss, trn_similarity, trn_regularization, trn_adversary, trn_mae))
    time_elapsed = time.time() - since
    print('Train Time {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Epoch {:d}\nTrain - Loss: {:.4f} | Simi: {:.4f} | Regular: {:.4f} | Adver: {:.4f} | MAE: {:.4f}'.format(
        epoch,  trn_loss, trn_similarity, trn_regularization, trn_adversary, trn_mae), file=f)
    print('Train Time {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60), file=f)

    ### Val ###
    val_loss, val_similarity, val_regularization, val_mae = utils.val(
            modelCAD, modelORG, modelENH, val_loader, criterion, epoch)
    print('Val - Loss: {:.4f} | Simi: {:.4f} | Regular: {:.4f} | MAE: {:.4f}'.format(
            val_loss, val_similarity, val_regularization, val_mae))
    time_elapsed = time.time() - since
    print('Total Time {:.0f}m {:.0f}s\n'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Val - Loss: {:.4f} |  Simi: {:.4f} | Regular: {:.4f} | MAE: {:.4f}'.format(
            val_loss, val_similarity, val_regularization, val_mae), file=f)
    print('Total Time {:.0f}m {:.0f}s\n'.format(
        time_elapsed // 60, time_elapsed % 60), file=f)
    scheduler.step(val_loss)

    print(optimizer.state_dict()['param_groups'][0]['lr'])
    ### Checkpoint ###
    utils.save_weightsCAD(en, att, modelCAD, optimizer, val, epoch, val_loss, val_similarity, WEIGHTS_PATH)
    utils.save_weightsORG(en, att, modelORG, optimizer, val, epoch, val_loss, val_similarity, WEIGHTS_PATH)
    utils.save_weightsENH(en, att, modelENH, optimizer, val, epoch, val_loss, val_similarity, WEIGHTS_PATH)
    ### Save/Plot loss ###
    loss_info = [epoch, trn_similarity,trn_regularization, trn_adversary, val_similarity,val_regularization]
    losses.append(loss_info)

utils.save_loss(losses, loss_label, LOSSES_PATH, RESULTS_PATH,8)
f.close()

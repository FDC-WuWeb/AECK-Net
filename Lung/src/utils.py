import os
import shutil
import matplotlib.pyplot as plt
import torch
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch.nn.functional as F

def save_weightsCAD(en,att,model,optimizer,val, epoch, loss, err, WEIGHTS_PATH):
    if en:
        weights_fname = 'weights-en-val%g-%d-%.3f-%.3fCAD.pth' % (val,epoch, loss, err)
    elif att:
        weights_fname = 'weights-val%g_plus-att-%d-%.3f-%.3fCAD.pth' % (val, epoch, loss, err)
    else:
        weights_fname = 'weights-val%g_plus-%d-%.3f-%.3fCAD.pth' % (val, epoch, loss, err)
    torch.save({
        'startEpoch': epoch,
        'loss': loss,
        'error': err,
        'state_dict': model.state_dict(),
        'optimizer':optimizer.state_dict()
    }, './weights-adam/' + weights_fname)
    shutil.copyfile('./weights-adam/'+ weights_fname, WEIGHTS_PATH + 'latestCAD.pth')

def save_weightsORG(en,att,model,optimizer,val, epoch, loss, err, WEIGHTS_PATH):
    if en:
        weights_fname = 'weights-en-val%g-%d-%.3f-%.3fdown1ORG.pth' % (val,epoch, loss, err)
    elif att:
        weights_fname = 'weights-val%g_plus-att-%d-%.3f-%.3fORG.pth' % (val, epoch, loss, err)
    else:
        weights_fname = 'weights-val%g_plus-%d-%.3f-%.3fORG.pth' % (val, epoch, loss, err)
    torch.save({
        'startEpoch': epoch,
        'loss': loss,
        'error': err,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, './weights-adam/' + weights_fname)
    shutil.copyfile('./weights-adam/'+ weights_fname, WEIGHTS_PATH + 'latestORG.pth')

def save_weightsENH(en,att,model,optimizer,val, epoch, loss, err, WEIGHTS_PATH):
    if en:
        weights_fname = 'weights-en-val%g-%d-%.3f-%.3fENH.pth' % (val,epoch, loss, err)
    elif att:
        weights_fname = 'weights-val%g_plus-att-%d-%.3f-%.3fENH.pth' % (val, epoch, loss, err)
    else:
        weights_fname = 'weights-val%g_plus-%d-%.3f-%.3fENH.pth' % (val, epoch, loss, err)
    torch.save({
        'startEpoch': epoch,
        'loss': loss,
        'error': err,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, './weights-adam/' + weights_fname)
    shutil.copyfile('./weights-adam/'+ weights_fname, WEIGHTS_PATH + 'latestENH.pth')

def load_weights(model,optimizer, fpath):
    weights = torch.load(fpath, map_location='cpu')
    startEpoch = weights['startEpoch']
    model.load_state_dict(weights['state_dict'],strict=False)
    optimizer.load_state_dict(weights['optimizer'])
    return startEpoch

def save_loss(losses, loss_label, LOSSES_PATH, RESULTS_PATH, case_id, plot=True):
    loss_fname = str(case_id) + 'losses-' + loss_label + '.pth'
    loss_fpath = os.path.join(LOSSES_PATH, loss_fname)
    torch.save(losses, loss_fpath)
    if plot:
        plot_loss_lcc(losses, RESULTS_PATH)
        
def load_loss(loss_fname, LOSSES_PATH, RESULTS_PATH, plot=True):
    loss_fpath = os.path.join(LOSSES_PATH, loss_fname)
    losses = torch.load(loss_fpath)
    if plot:
        plot_loss_lcc(losses, RESULTS_PATH)
        
def adjust_lr(lr, decay, optimizer, cur_epoch, n_epochs):
    """Sets the learning rate to the initially
        configured `lr` decayed by `decay` every `n_epochs`"""
    new_lr = lr * (decay ** (cur_epoch // n_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def plot_loss(losses):
    leng = len(losses)
    trn_loss, val_loss = [], []
    epochs = []
    for i in range(leng):
        epochs.append(losses[i][0])
        trn_loss.append(losses[i][1])
        val_loss.append(losses[i][3])
    #plot loss
    plt.figure('Loss/Error curves')
    plt.title('Loss curve')
    plt.plot(epochs, trn_loss, label='train')
    plt.plot(epochs, val_loss, label='val')
#    plt.ylim((0, 0.15))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()  
    plt.grid(True)
    plt.tight_layout()
    plt.show() 
    
def plot_loss_mae(losses):
    leng = len(losses)
    trn_loss, val_loss = [], []
    epochs = []
    trn_ed, val_ed = [], []
    for i in range(leng):
        epochs.append(losses[i][0])
        trn_loss.append(losses[i][1])
        trn_ed.append(losses[i][2])
        val_loss.append(losses[i][3])
        val_ed.append(losses[i][4])
    #plot loss
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('Loss & MAE curve')
    line1 = ax1.plot(epochs, trn_loss, label='train loss')
    line2 = ax1.plot(epochs, val_loss, label='val loss')
    ax2 = ax1.twinx()
    line3 = ax2.plot(epochs, trn_ed, '-r', label='train mae')
    line4 = ax2.plot(epochs, val_ed, '-b', label='val mae')
    #set legend in the fig
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    fig.legend(lines, labels, loc=1,
               bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epoch')
    ax2.set_ylabel('mae')
    ax1.grid(True)
    plt.tight_layout()
    plt.show()

def plot_loss_lcc(losses, RESULTS_PATH):
    leng = len(losses)
    trn_similarity, val_similarity = [], []
    epochs = []
    trn_regularization, val_regularization = [], []
    trn_adversary = []
    for i in range(leng):
        epochs.append(losses[i][0])
        trn_similarity.append(losses[i][1])
        trn_regularization.append(losses[i][2])
        trn_adversary.append(losses[i][3])
        val_similarity.append(losses[i][4])
        val_regularization.append(losses[i][5])
    #plot loss
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('Train Loss curves')
    ax1.plot(epochs, trn_similarity, '-b', label='similarity loss')
    ax1.plot(epochs, trn_regularization, '-r',label='regularization loss')
    ax1.plot(epochs, trn_adversary, '-g', label='adversarial loss')
    #set legend in the fig
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epoch')
    ax1.grid(False)
    plt.legend()

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.set_title('Val Loss curves')
    ax2.plot(epochs, val_similarity,'-b', label='similarity loss')
    ax2.plot(epochs, val_regularization, '-r',label='regularization loss')
    #set legend in the fig
    ax2.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    ax2.grid(False)
    plt.legend()
    # plt.tight_layout()
    plt.savefig(RESULTS_PATH + 'newlossfig.png')
    plt.show()

def antifoldloss(y_pred):
    dy = y_pred[:, :, :-1, :, :] - y_pred[:, :, 1:, :, :]-1
    dx = y_pred[:, :, :, :-1, :] - y_pred[:, :, :, 1:, :]-1
    dz = y_pred[:, :, :, :, :-1] - y_pred[:, :, :, :, 1:]-1

    dy = F.relu(dy) * torch.abs(dy*dy)
    dx = F.relu(dx) * torch.abs(dx*dx)
    dz = F.relu(dz) * torch.abs(dz*dz)
    return (torch.mean(dx)+torch.mean(dy)+torch.mean(dz))/3.0

def train(modelCAD, modelORG, modelENH, modelD, trn_loader, optimizer, criterion):
    modelCAD.train()
    modelORG.train()
    modelENH.train()
    modelD.train()
    a = 0.01
    b = 0.01
    c = 0.98
    m = 1
    trn_loss, trn_lcc, trn_mse, trn_d1, trn_d2, trn_diff, trn_mae, trn_adversary= 0, 0, 0, 0, 0, 0, 0, 0
    for i, [movCAD, refCAD, movORG, refORG, movENH, refENH] in enumerate(trn_loader):
        # print(i)
        movCAD = movCAD.cuda()
        refCAD = refCAD.cuda()

        movORG = movORG.cuda()
        refORG = refORG.cuda()

        movENH = movENH.cuda()
        refENH = refENH.cuda()

        optimizer.zero_grad()

        warpedCAD_ORG, warpedCAD, flowCAD = modelCAD(movCAD, refCAD, movORG)
        warpedORG_ENH, warpedORG, flowORG = modelORG(warpedCAD_ORG, refORG, flowCAD, movENH)
        warpedENH, flowENH = modelENH(warpedORG_ENH, refENH, flowORG, movENH)

        optimizer.zero_grad()
        loss1CAD = criterion['ncc'](warpedCAD, refCAD)
        loss1ORG = criterion['ncc'](warpedORG, refORG)
        loss1ENH = criterion['ncc'](warpedENH, refENH)
        loss2CAD = criterion['mse'](warpedCAD, refCAD)
        loss2ORG = criterion['mse'](warpedORG, refORG)
        loss2ENH = criterion['mse'](warpedENH, refENH)

        loss = a*(loss1CAD+ m*loss2CAD) + b*(loss1ORG+m*loss2ORG) + c*(loss1ENH+m*loss2ENH)

        loss.backward()
        optimizer.step()

        torch.cuda.empty_cache()
        
        trn_loss += loss.item()
        trn_lcc += (loss1CAD.item()*a + loss1ORG.item()*b + loss1ENH.item()*c)
        trn_mse += (loss2CAD.item()*a + loss2ORG.item()*b + loss1ENH.item()*c)

    trn_loss /= len(trn_loader)
    trn_lcc /= len(trn_loader)
    trn_mse /= len(trn_loader)
    trn_d1 /= len(trn_loader)
    trn_d2 /= len(trn_loader)
    trn_diff /= len(trn_loader)
    trn_mae /= len(trn_loader)
    trn_adversary /= len(trn_loader)
    trn_similarity = trn_lcc + trn_mse
    trn_regularization = trn_d1 + trn_d2 + trn_diff
    return trn_loss, trn_similarity, trn_regularization, trn_adversary, trn_mae

def val(modelCAD, modelORG, modelENH,val_loader, criterion, epoch):
    modelCAD.eval()
    modelORG.eval()
    modelENH.eval()
    a = 0.01
    b = 0.01
    c = 0.98
    d = 5
    m = 1
    val_loss, val_lcc, val_mse, val_d1, val_d2, val_diff, val_mae = 0, 0, 0, 0, 0, 0, 0
    for movCAD, refCAD, movORG, refORG, movENH, refENH in val_loader:
        movCAD = movCAD.cuda()
        refCAD = refCAD.cuda()

        movORG = movORG.cuda()
        refORG = refORG.cuda()

        movENH = movENH.cuda()
        refENH = refENH.cuda()

        with torch.no_grad():
            warpedCAD_ORG, warpedCAD, flowCAD = modelCAD(movCAD, refCAD, movORG)
            warpedORG_ENH, warpedORG, flowORG = modelORG(warpedCAD_ORG, refORG, flowCAD, movENH)
            warpedENH, flowENH = modelENH(warpedORG_ENH, refENH, flowORG, movENH)

            loss1CAD = criterion['ncc'](warpedCAD, refCAD)
            loss1ORG = criterion['ncc'](warpedORG, refORG)
            loss1ENH = criterion['ncc'](warpedENH, refENH)
            loss2CAD = criterion['mse'](warpedCAD, refCAD)
            loss2ORG = criterion['mse'](warpedORG, refORG)
            loss2ENH = criterion['mse'](warpedENH, refENH)

            loss = a*(loss1CAD+m*loss2CAD) + b*(loss1ORG+m*loss2ORG) + c*(loss1ENH+m*loss2ENH)

        torch.cuda.empty_cache()
        
        val_loss += loss.item()
        val_lcc += (loss1CAD.item()*a + loss1ORG.item()*b + loss1ENH.item()*c)
        val_mse += (loss2CAD.item()*a + loss2ORG.item()*b + loss1ENH.item()*c)

    val_loss /= len(val_loader)
    val_lcc /= len(val_loader)
    val_mse /= len(val_loader)
    val_d1 /= len(val_loader)
    val_d2 /= len(val_loader)
    val_diff /= len(val_loader)
    val_mae /= len(val_loader)
    val_similarity = val_lcc + val_mse
    val_regularization = val_d1 + val_d2 + val_diff
    return val_loss, val_similarity, val_regularization,val_mae



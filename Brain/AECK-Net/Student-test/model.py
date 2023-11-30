# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import warp
import ViP3d
from thop import profile
class conv_down(nn.Module):
    def __init__(self, inChan, outChan, down=True, pool_kernel=2):
        super(conv_down, self).__init__()
        self.down = down
        self.conv = nn.Sequential(
            nn.Conv3d(inChan, outChan, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(outChan),
            nn.LeakyReLU(0.2)
        )
        self.pool = nn.MaxPool3d(pool_kernel)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        if self.down:
            x = self.pool(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.LeakyReLU(0.2),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class gateconv(nn.Module):
    def __init__(self, inChan, outChan, down=True, pool_kernel=2):
        super(gateconv, self).__init__()
        self.down = down
        self.conv = nn.Sequential(
            nn.Conv3d(inChan, outChan, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        return x

class attention_gate(nn.Module):
    def __init__(self,channel):
        super(attention_gate, self).__init__()
        self.gateconv1 = gateconv(channel,3)
        self.gateconv2 = gateconv(3,3)
        self.gateconv3 = gateconv(3,3)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x_ra, x):
        x_flow = x
        x_ra = self.gateconv1(x_ra)
        x = self.gateconv2(x)
        x = self.relu(x + x_ra)
        x = self.gateconv3(x)
        x = self.sigmoid(x)
        x = x * x_flow
        return x

class NetCAD(nn.Module):
    def __init__(self,  nfea=[2,16,32,64,64,64,128,64,32,3]):  #num of channel
        super(NetCAD, self).__init__()
        """
        net architecture. 
        :param nfea: list of conv filters. right now it needs to be 1x8.
        """
        self.down1 = conv_down(nfea[0], nfea[1])
        self.down2 = conv_down(nfea[1], nfea[2])
        self.down3 = conv_down(nfea[2], nfea[3])

        self.same3 = conv_down(nfea[5], nfea[6], down=False)
        self.same4 = conv_down(nfea[6], nfea[7], down=False)

        self.channelAttention6 = SELayer(nfea[6])
        self.channelAttention7 = SELayer(nfea[7])

        self.ViP6 = ViP3d.WeightedPermuteMLP(128,8,8,4,seg_dim=4,res=True)
        self.ViP7 = ViP3d.WeightedPermuteMLP(64,8,8,4,seg_dim=4,res=True)
        self.attg1 = attention_gate(128)

        self.outconv = nn.Conv3d(
                64, nfea[9], kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):

        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        flow_CADdownKD = x

        x = self.same3(x)
        x = self.ViP6(x)
        x = self.channelAttention6(x)
        x_ra = x

        x = self.same4(x)
        x = self.ViP7(x)
        x = self.channelAttention7(x)

        x = self.outconv(x)
        x = self.attg1(x_ra, x)
        flow_CADoutKD = x
        x = F.interpolate(x,scale_factor=8,mode="trilinear", align_corners= True)
        return x, flow_CADdownKD,flow_CADoutKD

class NetORG(nn.Module):
    def __init__(self,  nfea=[2,16,32,64,64,64,128,64,32,3]):
        super(NetORG, self).__init__()

        self.down1 = conv_down(nfea[0], nfea[1])
        self.down2 = conv_down(nfea[1], nfea[2])
        self.down3 = conv_down(nfea[2], nfea[3])

        self.same3 = conv_down(nfea[5], nfea[6], down=False)
        self.same4 = conv_down(nfea[6], nfea[7], down=False)

        self.channelAttention6 = SELayer(nfea[6])
        self.channelAttention7 = SELayer(nfea[7])

        self.ViP6 = ViP3d.WeightedPermuteMLP(128,8,8,4,seg_dim=4,res=True)
        self.ViP7 = ViP3d.WeightedPermuteMLP(64,8,8,4,seg_dim=4,res=True)
        self.attg1 = attention_gate(128)

        self.outconv = nn.Conv3d(
                64, nfea[9], kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):

        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        flow_ORGdownKD = x
        x = self.same3(x)
        x = self.ViP6(x)
        x = self.channelAttention6(x)
        x_ra = x

        x = self.same4(x)
        x = self.ViP7(x)
        x = self.channelAttention7(x)

        x = self.outconv(x)
        x = self.attg1(x_ra, x)
        flow_ORGoutKD = x
        x = F.interpolate(x,scale_factor=8,mode="trilinear", align_corners= True)
        return x,flow_ORGdownKD,flow_ORGoutKD


class NetENH(nn.Module):
    def __init__(self,  nfea=[2,16,32,64,64,64,128,64,32,3]):  #num of channel
        super(NetENH, self).__init__()
        """
        net architecture. 
        :param nfea: list of conv filters. right now it needs to be 1x8.
        """
        self.down1 = conv_down(nfea[0], nfea[1])
        self.down2 = conv_down(nfea[1], nfea[2])
        self.down3 = conv_down(nfea[2], nfea[3])

        self.same3 = conv_down(nfea[5], nfea[6], down=False)
        self.same4 = conv_down(nfea[6], nfea[7], down=False)

        self.channelAttention6 = SELayer(nfea[6])
        self.channelAttention7 = SELayer(nfea[7])

        self.ViP6 = ViP3d.WeightedPermuteMLP(128,8,8,4,seg_dim=4,res=True)
        self.ViP7 = ViP3d.WeightedPermuteMLP(64,8,8,4,seg_dim=4,res=True)
        self.attg1 = attention_gate(128)

        self.outconv = nn.Conv3d(
                64, nfea[9], kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):

        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        flow_ENHdownKD = x
        x = self.same3(x)
        x = self.ViP6(x)
        x = self.channelAttention6(x)
        x_ra = x

        x = self.same4(x)
        x = self.ViP7(x)
        x = self.channelAttention7(x)

        x = self.outconv(x)
        x = self.attg1(x_ra, x)
        flow_ENHoutKD = x
        x = F.interpolate(x,scale_factor=8,mode="trilinear", align_corners= True)

        return x,flow_ENHdownKD,flow_ENHoutKD

class snetCAD(nn.Module):
    def __init__(self,  img_size=[256,256,128]):
        super(snetCAD, self).__init__()
        self.net = NetCAD()
        self.warper = warp.Warper3d(img_size)

    def forward(self, movCAD, refCAD, movORG):
        input0 = torch.cat((movCAD, refCAD), 1)
        flowCAD,flow_CADdownKD,flow_CADoutKD = self.net(input0)
        warpedCAD = self.warper(movCAD,flowCAD)
        warpedCAD_ORG = self.warper(movORG, flowCAD)
        return warpedCAD_ORG, warpedCAD, flowCAD,flow_CADdownKD,flow_CADoutKD

class snetORG(nn.Module):
    def __init__(self,  img_size=[256,256,96]):
        super(snetORG, self).__init__()
        self.net = NetORG()
        self.warper = warp.Warper3d(img_size)

    def forward(self, warpedCAD_ORG, refORG, flowCAD, movENH):
        input0 = torch.cat((warpedCAD_ORG, refORG), 1)
        flowres_ORG,flow_ORGdownKD,flow_ORGoutKD = self.net(input0)
        flowORG = flowCAD + flowres_ORG
        warpORG = self.warper(warpedCAD_ORG,flowORG)
        warpedORG_ENH = self.warper(movENH, flowORG)
        return warpedORG_ENH, warpORG, flowORG,flow_ORGdownKD,flow_ORGoutKD

class snetENH(nn.Module):
    def __init__(self,  img_size=[256,256,96]):
        super(snetENH, self).__init__()
        self.net = NetENH()
        self.warper = warp.Warper3d(img_size)

    def forward(self, warpedORG_ENH, refENH, flowORG, movENH):
        input0 = torch.cat((warpedORG_ENH, refENH), 1)
        flowres_ENH,flow_ENHdownKD,flow_ENHoutKD = self.net(input0)
        flowENH = flowORG + flowres_ENH
        warpedENH = self.warper(movENH, flowENH)
        return warpedENH, flowENH,flow_ENHdownKD,flow_ENHoutKD

class Discriminator(nn.Module):
    def __init__(self, nfea=[1, 8, 16, 32, 64, 64, 64, 128, 64, 32, 3]):
        super(Discriminator, self).__init__()
        self.down1 = conv_down(nfea[0], nfea[1])
        self.down2 = conv_down(nfea[1], nfea[2])
        self.down3 = conv_down(nfea[2], nfea[3])

        self.outconv = nn.Conv3d(
            nfea[3], 1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.outconv(x)
        return x

class snetD(nn.Module):
    def __init__(self,  img_size=[256,256,96]):
        super(snetD, self).__init__()
        self.net = Discriminator()

    def forward(self, warped):

        RealorFake = self.net(warped)
        return RealorFake

def print_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    print('Number of params: %.4fM' % (total / 1e6))

if __name__ == '__main__':
    device = torch.device("cpu")
    net1 = snetCAD().to(device)
    input = torch.randn(1, 1, 256, 256, 128).to(device)
    flops, params = profile(net1, (input, input, input))
    print('flops: %.2f G, params: %.2f M' % (flops / 1e9, params / 1e6))

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch.autograd import Variable
import math
import random
import numpy as np

stage_out_channel = [32] + [64] + [128] * 2 + [256] * 2 + [512] * 6 + [1024] * 2

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def binaryconv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return HardBinaryConv(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)


def binaryconv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return HardBinaryConv(in_planes, out_planes, kernel_size=1, stride=stride, padding=0)

class firstconv3x3(nn.Module):
    def __init__(self, inp, oup, stride):
        super(firstconv3x3, self).__init__()

        self.conv1 = nn.Conv2d(inp, oup, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(oup)

    def forward(self, x, empty=None):

        out = self.conv1(x)
        out = self.bn1(out)

        return out

class StdMean(nn.Module):
    def __init__(self, num_features, momentum=0.01):
        super(StdMean, self).__init__()
        # 对每个batch的mean和var进行追踪统计
        self._running_mean = torch.zeros(size=(num_features, )).cuda()
        # 更新self._running_xxx时的动量
        self._momentum = momentum

    def update(self, x):
        """
        BN向传播
        :param x: 数据
        :return: BN输出
        """
        self._running_mean = (1-self._momentum) * x.detach() + self._momentum * self._running_mean.detach()
        return self._running_mean

    def val(self):
        return self._running_mean

class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out

class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out

class HardBinaryConv(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1):
        super(HardBinaryConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        self.weights = nn.Parameter(torch.rand((self.number_of_weights,1)) * 0.001, requires_grad=True)

    def forward(self, x):
        real_weights = self.weights.view(self.shape)
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        #print(scaling_factor, flush=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        #print(binary_weights, flush=True)
        y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)

        return y

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        norm_layer = nn.BatchNorm2d

        self.move11 = LearnableBias(inplanes)
        self.binary_3x3= binaryconv3x3(inplanes, inplanes, stride=stride)
        self.bn1 = norm_layer(inplanes)

        self.move12 = LearnableBias(inplanes)
        self.prelu1 = nn.PReLU(inplanes)
        self.move13 = LearnableBias(inplanes)

        self.move21 = LearnableBias(inplanes)

        if inplanes == planes:
            self.binary_pw = binaryconv1x1(inplanes, planes)
            self.bn2 = norm_layer(planes)
        else:
            self.binary_pw_down1 = binaryconv1x1(inplanes, inplanes)
            self.binary_pw_down2 = binaryconv1x1(inplanes, inplanes)
            self.bn2_1 = norm_layer(inplanes)
            self.bn2_2 = norm_layer(inplanes)

        self.move22 = LearnableBias(planes)
        self.prelu2 = nn.PReLU(planes)
        self.move23 = LearnableBias(planes)

        self.binary_activation = BinaryActivation()
        self.stride = stride
        self.inplanes = inplanes
        self.planes = planes

        if self.inplanes != self.planes:
            self.pooling = nn.AvgPool2d(2,2)

    def forward(self, x, get_binary_feat=False):

        out1 = self.move11(x)

        out1 = self.binary_activation(out1)
        if get_binary_feat:
            outout = [out1]
        # print("basicblock_conv1_std: ", out1.var([2,3]).pow(2).mean().item())
        out1 = self.binary_3x3(out1)
        out1 = self.bn1(out1)

        if self.stride == 2:
            x = self.pooling(x)

        out1 = x + out1

        out1 = self.move12(out1)
        out1 = self.prelu1(out1)
        out1 = self.move13(out1)

        out2 = self.move21(out1)
        out2 = self.binary_activation(out2)
        if get_binary_feat:
            outout.append(out2)

        if self.inplanes == self.planes:
            out2 = self.binary_pw(out2)
            out2 = self.bn2(out2)
            out2 += out1

        else:
            assert self.planes == self.inplanes * 2

            out2_1 = self.binary_pw_down1(out2)
            out2_2 = self.binary_pw_down2(out2)
            out2_1 = self.bn2_1(out2_1)
            out2_2 = self.bn2_2(out2_2)
            out2_1 += out1
            out2_2 += out1
            out2 = torch.cat([out2_1, out2_2], dim=1)

        out2 = self.move22(out2)
        out2 = self.prelu2(out2)
        out2 = self.move23(out2)

        if get_binary_feat:
            return out2, outout
        return out2


class ReActNet(nn.Module):
    def __init__(self, num_classes=1000, cal_var_loss=False, var_type='last'):
        super(ReActNet, self).__init__()
        self.feature = nn.ModuleList()
        self.var_type = var_type
        self.cal_var_loss = cal_var_loss
        for i in range(len(stage_out_channel)):
            if i == 0:
                self.feature.append(firstconv3x3(3, stage_out_channel[i], 2))
            elif stage_out_channel[i-1] != stage_out_channel[i] and stage_out_channel[i] != 64:
                self.feature.append(BasicBlock(stage_out_channel[i-1], stage_out_channel[i], 2))
            else:
                self.feature.append(BasicBlock(stage_out_channel[i-1], stage_out_channel[i], 1))
        #     if i != 0:
        #         if cal_var_loss and var_type=='all':
        #             self.var_mean = []
        #             self.var_std = []
        #             self.var_mean.append(StdMean(stage_out_channel[i-1]))
        #             self.var_std.append(StdMean(stage_out_channel[i-1]))
        #             self.var_mean.append(StdMean(stage_out_channel[i]))
        #             self.var_std.append(StdMean(stage_out_channel[i]))
        # if var_type[0]=='*':
        #     self.var_mean = []
        #     self.var_std = []
        #     num_list = self.var_type[1::].split('_')
        #     for i in num_list:
        #         num = int(i)
        #         self.var_mean.append(StdMean(stage_out_channel[num-1]))
        #         self.var_std.append(StdMean(stage_out_channel[num-1]))
        #         self.var_mean.append(StdMean(stage_out_channel[num]))
        #         self.var_std.append(StdMean(stage_out_channel[num]))

        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

        # if cal_var_loss and var_type=='last':
        #     self.var_mean = [StdMean(stage_out_channel[-1])]
        #     self.var_std = [StdMean(stage_out_channel[-1])]

    def is_patch_based(self):
        return False

    def forward(self, x, gt=None, flag=None, epoch=None):
        cal_var_loss = self.cal_var_loss
        var_out = []
        for i, block in enumerate(self.feature):
            # print("block"+str(i), end='')
            if i !=0 and cal_var_loss:
                x, binfeat = block(x, cal_var_loss)
                var_out += binfeat
            else:
                x = block(x, cal_var_loss)

        x = self.pool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        out={'class_logit': x}
        if cal_var_loss:
            if self.var_type=='last':
                # self.var_mean[0].update(var_out[-1].mean([0,2,3]))
                tmp = var_out[-1].var([2,3]).pow(2).mean(0)
                # self.var_std[0].update(tmp)
                var_loss = tmp.mean()
                mean_loss = var_out[-1].abs().mean()
            elif self.var_type=='all':
                var_loss = 0
                mean_loss = 0
                # var_std = []
                for num, i in enumerate(var_out):
                    # self.var_mean[num].update(i.mean([0,2,3]))
                    tmp = i.var([2,3]).pow(2).mean(0)
                    # var_std.append(tmp)
                    var_loss += tmp.mean()
                    mean_loss += i.abs().mean()
            else:
                id = 0
                assert self.var_type[0]=='-'
                num_list = self.var_type[1::].split('_')
                var_loss = 0
                mean_loss = 0
                # var_std = []
                for num, i in enumerate(var_out):
                    if str((num)//2+1) in num_list:
                        # self.var_mean[id].update(i.mean([0,2,3]))
                        tmp = i.var([2,3]).pow(2).mean(0)
                        # var_std.append(tmp)
                        var_loss += tmp.mean()
                        mean_loss += i.abs().mean()
                        id += 1
            # else:
            #     raise("wrong var_type: " + self.var_type)
            out.update({"var_loss": var_loss.sqrt()})
            out.update({"mean_loss": mean_loss})
            # out.update({"var_mean": self.var_mean})
            # out.update({"var_std": self.var_std})
        return out

    def forward_get_binary_feat(self, x, feat=True):
        out = []
        for i, block in enumerate(self.feature):
            # print("block"+str(i), end='')
            if i !=0 and feat:
                x, binfeat = block(x, feat)
                out+=binfeat
            else:
                x = block(x, feat)
        x = self.pool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, out

def reactnet(classes, cal_var_loss, var_type, pretrained=True):
    """Constructs a BiRealNet-18 model. """
    model = ReActNet(num_classes=classes, cal_var_loss=cal_var_loss, var_type=var_type)
    if pretrained:
        a = torch.load('./models/model_best_reactnet.pth.tar')['state_dict']
        # from collections import OrderedDict
        model_state = model.state_dict()
        for k, v in a.items():
            name = k.replace("module.", "")
            if 'fc' in name:
                continue
            elif v is not None:
                model_state[name].copy_(v)
                print("load layer " + name)
            else:
                print(name + "is not loaded")
        model.load_state_dict(model_state)
    return model
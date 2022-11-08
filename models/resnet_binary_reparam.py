import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F


__all__ = ['birealnet18', 'birealnet34']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

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
        self._running_mean = (1-self._momentum)*x.detach() + self._momentum*self._running_mean.detach()
        return self._running_mean

    def val(self):
        return self._running_mean.detach()

class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        #out_e1 = (x^2 + 2*x)
        #out_e2 = (-x^2 + 2*x)
        # out_e_total = 0
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
        # self.weights = nn.Parameter(torch.rand((self.number_of_weights,1)) * 0.001, requires_grad=True)
        self.weights = nn.Parameter(torch.rand((out_chn, in_chn, kernel_size, kernel_size)) * 0.001, requires_grad=True)

    def forward(self, x, binary=True, distortion=False):
        real_weights = self.weights.view(self.shape)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(cliped_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        #print(scaling_factor, flush=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        #print(binary_weights, flush=True)

        if binary:
            y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)
            return y
        else:
            res_with_grad = cliped_weights - binary_weights_no_grad.detach()
            # distortion added here
            if distortion:
                d=((scaling_factor*0.5)**0.5)*torch.randn(self.shape).cuda()
            else:
                d=0
            y = F.conv2d(x, binary_weights+res_with_grad+d, stride=self.stride, padding=self.padding)
            return y, torch.mean(abs(res_with_grad))

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.planes=inplanes
        self.move0 = LearnableBias(inplanes)
        self.binary_activation = BinaryActivation()
        self.binary_conv = HardBinaryConv(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.move1 = LearnableBias(planes)
        self.prelu = nn.PReLU(planes)
        self.move2 = LearnableBias(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x, binary=True, get_binary_feat=False, distortion=False):
        residual = x
        out = self.move0(x)
        out = self.binary_activation(out)
        if get_binary_feat:
            outout = out
        if binary:
            out = self.binary_conv(out, binary)
        else:
            out, restmp = self.binary_conv(out, binary, distortion=distortion)
        out = self.bn1(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.move1(out)
        out = self.prelu(out)
        out = self.move2(out)
        finalout={'out': out}
        if get_binary_feat:
            finalout.update({'feat': outout})
        if not binary:
            finalout.update({'restmp': restmp})
        return finalout
        # if binary:
        #     return out
        # else:
        #     return out, restmp
        # if get_binary_feat:
        #     return out, outout
        # return out

class BiRealNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, cal_var_loss=False, var_type='last'):
        super(BiRealNet, self).__init__()
        self.inplanes = 64
        self.var_type = var_type
        self.cal_var_loss = cal_var_loss
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        if var_type=='last':
            self.var_mean = [StdMean(512)]
            self.var_std = [StdMean(512)]
        elif var_type=='all' or var_type[0]=='-':
            self.var_mean = []
            self.var_std = []
            for layers in [self.layer1, self.layer2, self.layer3, self.layer4]:
                for layer in layers:
                    self.var_mean.append(StdMean(layer.planes))
                    self.var_std.append(StdMean(layer.planes))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=stride),
                conv1x1(self.inplanes, planes * block.expansion),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def is_patch_based(self):
        return False

    def forward(self, x, gt=None, binary=True, flag=None, epoch=None, more_out=False, distortion=False):
        cal_var_loss = self.cal_var_loss
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        res = []
        feat = []
        if more_out:
            more = []
        for layers in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for layer in layers:
                out = layer(x, binary=binary, get_binary_feat=cal_var_loss, distortion=distortion)
                x = out['out']
                if more_out:
                    more.append(x)
                if cal_var_loss:
                    binfeat = out['feat']
                    feat.append(binfeat)
                if not binary:
                    restmp = out['restmp']
                    res.append(restmp)
        if cal_var_loss:
            if self.var_type=='last':
                self.var_mean[0].update(feat[-1].mean([0,2,3]))
                tmp = feat[-1].var(0).pow(2).mean([0,1,2])
                self.var_std[0].update(tmp)
                var_loss = tmp.mean()
                mean_loss = feat[-1].abs().mean()
            elif self.var_type=='all':
                var_loss = 0
                mean_loss = 0
                var_std = []
                for num, i in enumerate(feat):
                    self.var_mean[num].update(i.mean([0,2,3]))
                    tmp = i.var(0).pow(2).mean([0,1,2])
                    mean_loss += i.abs().mean()
                    var_std.append(tmp)
                    var_loss += tmp.mean()
            elif self.var_type[0]=='-':
                var_loss = 0
                mean_loss = 0
                var_std = []
                num_list = self.var_type[1::].split('_')
                for num, i in enumerate(feat):
                    if str(num) in num_list:
                        self.var_mean[num].update(i.mean([0,2,3]))
                        tmp = i.var(0).pow(2).mean([0,1,2])
                        var_std.append(tmp)
                        mean_loss += i.abs().mean()
                        var_loss += tmp.mean()
            else:
                raise("wrong var_type: " + self.var_type)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        out={'class_logit': x}
        if cal_var_loss:
            out.update({"var_loss": var_loss.sqrt()})
            out.update({"mean_loss": mean_loss})
            out.update({"var_mean": self.var_mean})
            out.update({"var_std": self.var_std})
        if not binary:
            out.update({'res': sum(res)})
        if more_out and flag=='debug':
            out.update({'more': more})
        return out

    def forward_get_binary_feat(self, x, feat=True):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        featout = []
        for layers in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for layer in layers:
                out = layer(x, binary=True, get_binary_feat=feat)
                x, binfeat = out['out'], out['feat']
                featout.append(binfeat)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, featout

def birealnet18(pretrained="/media/disk2/yjm/DG/logs/test/cartoon-photo-sketch_to_art_painting/SGD_eps100_bs64_lr0.01_class7_jigClass30_jigWeight0.7decay1e-06_dataset_PACS_resnet18_binary_clip_TAll_bias0.9_Vholidayend_pretrained0/best_checkpoint_test.pth", classes=7, **kwargs):
    """Constructs a BiRealNet-18 model. """
    model = BiRealNet(BasicBlock, [4, 4, 4, 4], num_classes=classes, **kwargs)
    # print("hi")
    if not pretrained=='':
        a = torch.load(pretrained)['state_dict']
        # from collections import OrderedDict
        model_state = model.state_dict()
        for k, v in a.items():
            name = k.replace("module.", "")
            if 'fc' in name and pretrained=='./models/model_best_resnet_oldv.pth.tar':
                continue
            if v is not None:
                # print("load layer " + name)
                model_state[name].copy_(v)
            else:
                print(name + "is not loaded")
        model.load_state_dict(model_state)
    return model


def birealnet34(pretrained=False, **kwargs):
    """Constructs a BiRealNet-34 model. """
    model = BiRealNet(BasicBlock, [6, 8, 12, 6], **kwargs)
    return model


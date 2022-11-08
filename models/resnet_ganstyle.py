from torch import nn
from torch.utils import model_zoo
from torchvision.models.resnet import model_urls, Bottleneck, BasicBlock
import torch
from torch import nn as nn
from torch.autograd import Variable
import numpy.random as npr
import numpy as np
import torch.nn.functional as F
import math
from models.permuteAdaIN import AdaptiveInstanceNorm2d, calc_mean_std
import random
import torch.optim as optim

class Generator(nn.Module):
    def __init__(self, outchannel):
        super(Generator, self).__init__()
        self.outchannel = outchannel
        self.mean_vector = nn.Sequential(
            nn.Linear(64, outchannel * 2, bias=False),
            nn.BatchNorm1d(outchannel * 2),
            nn.ReLU(True),
            nn.Linear(outchannel * 2, outchannel * 2, bias=False),
            nn.BatchNorm1d(outchannel * 2),
            nn.ReLU(True),
            nn.Linear(outchannel * 2, outchannel, bias=False),
        )

        self.std_vector = nn.Sequential(
            nn.Linear(64, outchannel * 2, bias=False),
            nn.BatchNorm1d(outchannel * 2),
            nn.ReLU(True),
            nn.Linear(outchannel * 2, outchannel * 2, bias=False),
            nn.BatchNorm1d(outchannel * 2),
            nn.ReLU(True),
            nn.Linear(outchannel * 2, outchannel, bias=False),
        )

    def forward(self, input):
        return self.mean_vector(input), self.std_vector(input)


class Discriminator_Vector(nn.Module):
    def __init__(self, outchannel):
        super(Discriminator_Vector, self).__init__()
        self.outchannel = outchannel
        self.main = nn.Sequential(
            nn.Linear(outchannel, outchannel * 2, bias=False),
            nn.BatchNorm1d(outchannel * 2),
            nn.ReLU(True),
            nn.Linear(outchannel * 2, outchannel * 2, bias=False),
            nn.BatchNorm1d(outchannel * 2),
            nn.ReLU(True),
            nn.Linear(outchannel * 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


class Discriminator_Image(nn.Module):
    def __init__(self):
        super(Discriminator_Image, self).__init__()
        nc = 3
        ndf = 64
        self.main = nn.Sequential(
            # 输入大小 (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 输入大小. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 输入大小. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 输入大小. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

class BasicBlock1(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, with_style_adain=False):
        super().__init__()

        self.with_style_adain = with_style_adain

        if with_style_adain:
            self.stylegen1 = Generator(out_channels)
            self.stylegen2 = Generator(out_channels * BasicBlock.expansion)
            self.styledis1mean = Discriminator_Vector(out_channels)
            self.styledis1std = Discriminator_Vector(out_channels)
            self.styledis2mean = Discriminator_Vector(out_channels * BasicBlock.expansion)
            self.styledis2std = Discriminator_Vector(out_channels * BasicBlock.expansion)
            self.conv1_style_adain = AdaptiveInstanceNorm2d(out_channels * BasicBlock.expansion)
            self.conv2_style_adain = AdaptiveInstanceNorm2d(out_channels * BasicBlock.expansion)

        #residual function
        self.residual_function_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),

        )

        self.residual_function_2 = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock1.expansion, kernel_size=3, padding=1, bias=False),
        )

        self.residual_function_3 = nn.Sequential(
            nn.BatchNorm2d(out_channels * BasicBlock1.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock1.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock1.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock1.expansion)
            )

    def forward(self, x, noise=None, ganstyle=False, get_loss=True, get_mean_std=False):
        out1 = self.residual_function_1(x)


        if ganstyle and noise is not None:
            mean1_fake, std1_fake = self.stylegen1(noise)
            mean1_real, std1_real = calc_mean_std(out1)
            out1 = self.conv1_style_adain(out1, mean1_fake, std1_fake)


        out2 = self.residual_function_2(out1)


        if ganstyle and noise is not None:
            mean2_fake, std2_fake = self.stylegen2(noise)
            mean2_real, std2_real = calc_mean_std(out2)
            out2 = self.conv2_style_adain(out2, mean2_fake, std2_fake)

        outf = self.residual_function_3(out2)
        outf = nn.ReLU(inplace=True)(outf + self.shortcut(x))
        if get_loss and ganstyle and noise is not None and self.training:
            assert get_mean_std==False
            # mean1_out_real = self.styledis1mean(mean1_real)
            # std1_out_real = self.styledis2std(std1_real)
            # mean2_out_real = self.styledis2mean(mean2_real)
            # std2_out_real = self.styledis2std(std2_real)
            # mean1_out_fake = self.styledis1mean(mean1_fake)
            # std1_out_fake = self.styledis2std(std1_fake)
            # mean2_out_fake = self.styledis2mean(mean2_fake)
            # std2_out_fake = self.styledis2std(std2_fake)
            return [outf, mean1_real, std1_real, mean2_real, std2_real,
                    mean1_fake, std1_fake, mean2_fake, std2_fake]
        # elif get_mean_std:
        #     assert get_loss == False
        #     assert self.training
        #     mean1, std1 = calc_mean_std(out1)
        #     mean2, std2 = calc_mean_std(out2)
        #     mean1_out = self.styledis1mean(mean1)
        #     std1_out = self.styledis2std(std1)
        #     mean2_out = self.styledis2mean(mean2)
        #     std2_out = self.styledis2std(std2)
        #     return [outf, mean1_out, std1_out, mean2_out, std2_out]
        else:
            return outf

class ResNet(nn.Module):
    def __init__(self, block, layers, with_style_adain=False, p_adain=None, jigsaw_classes=1000, classes=100):
        self.inplanes = 64
        super(ResNet, self).__init__()
        #  init for random gan adain
        self.p_adain = p_adain
        self.with_style_adain = with_style_adain
        if with_style_adain:
            self.styleconv1gen = Generator(64)
            self.styleconv1meandis = Discriminator_Vector(64)
            self.styleconv1stddis = Discriminator_Vector(64)
            # self.styledis_I = Discriminator_Image()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], with_style_adain=with_style_adain)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, with_style_adain=with_style_adain)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, with_style_adain=with_style_adain)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, with_style_adain=with_style_adain)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.class_classifier = nn.Linear(512 * block.expansion, classes)
        self.pecent = 1/3
        if with_style_adain:
            self.gen = [self.styleconv1gen,
                        self.layer1[0].stylegen1, self.layer1[0].stylegen2,
                        self.layer1[1].stylegen1, self.layer1[1].stylegen2,
                        self.layer2[0].stylegen1, self.layer2[0].stylegen2,
                        self.layer2[1].stylegen1, self.layer2[1].stylegen2,
                        self.layer3[0].stylegen1, self.layer3[0].stylegen2,
                        self.layer3[1].stylegen1, self.layer3[1].stylegen2,
                        self.layer4[0].stylegen1, self.layer4[0].stylegen2,
                        self.layer4[1].stylegen1, self.layer4[1].stylegen2,]
            self.mean_dis = [self.styleconv1meandis,
                             self.layer1[0].styledis1mean, self.layer1[0].styledis2mean,
                             self.layer1[1].styledis1mean, self.layer1[1].styledis2mean,
                             self.layer2[0].styledis1mean, self.layer2[0].styledis2mean,
                             self.layer2[1].styledis1mean, self.layer2[1].styledis2mean,
                             self.layer3[0].styledis1mean, self.layer3[0].styledis2mean,
                             self.layer3[1].styledis1mean, self.layer3[1].styledis2mean,
                             self.layer4[0].styledis1mean, self.layer4[0].styledis2mean,
                             self.layer4[1].styledis1mean, self.layer4[1].styledis2mean,]
            self.std_dis = [self.styleconv1stddis,
                            self.layer1[0].styledis1std, self.layer1[0].styledis2std,
                            self.layer1[1].styledis1std, self.layer1[1].styledis2std,
                            self.layer2[0].styledis1std, self.layer2[0].styledis2std,
                            self.layer2[1].styledis1std, self.layer2[1].styledis2std,
                            self.layer3[0].styledis1std, self.layer3[0].styledis2std,
                            self.layer3[1].styledis1std, self.layer3[1].styledis2std,
                            self.layer4[0].styledis1std, self.layer4[0].styledis2std,
                            self.layer4[1].styledis1std, self.layer4[1].styledis2std,]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, blocks, stride=1, with_style_adain=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))#, with_style_adain=with_style_adain
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))#, with_style_adain=with_style_adain

        return nn.Sequential(*layers)

    def is_patch_based(self):
        return False

    def dis_loss(self, x):
        assert self.training
        mean_out = []
        std_out = []
        x = self.conv1(x)

        mean, std = calc_mean_std(x)
        mean_out.append(mean)
        std_out.append(std)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            x = layer(x, None, False, get_loss=False, get_mean_std=True)
            mean_out += [x[1], x[3]]
            std_out += [x[2], x[4]]
            x = x[0]
        return x, mean_out, std_out

    def forward(self, x, noise=None, gt=None, flag=None, epoch=None, randyes=False):
        # if self.training:
        #     assert noise is not None
        ganstyle = randyes and self.with_style_adain
        if self.training and ganstyle:
            mean_out_real = []
            mean_out_fake = []
            std_out_real = []
            std_out_fake = []
        x = self.conv1(x)
        if ganstyle and self.training:
            assert noise is not None
            mean_real, std_real = calc_mean_std(x)
            mean_out_real.append(mean_real)
            std_out_real.append(std_real)
            mean_fake, std_fake = self.styleconv1gen(noise)
            mean_out_fake.append(mean_fake)
            std_out_fake.append(std_fake)
            x = AdaptiveInstanceNorm2d()(x, mean_fake, std_fake)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        for layers in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for layer in layers:
                x = layer(x)#, noise, ganstyle, get_loss=ganstyle and self.training
                if ganstyle and self.training:
                    mean_out_real += [x[1],x[3]]
                    mean_out_fake += [x[5],x[7]]
                    std_out_real += [x[2],x[4]]
                    std_out_fake += [x[6],x[8]]
                    x = x[0]
            # x = self.layer2(x, noise, ganstyle, get_loss=ganstyle and self.training)
            # x = self.layer3(x, noise, ganstyle, get_loss=ganstyle and self.training)
            # x = self.layer4(x, noise, ganstyle, get_loss=ganstyle and self.training)

        if flag:
            interval = 10
            if epoch % interval == 0:
                self.pecent = 3.0 / 10 + (epoch / interval) * 2.0 / 10

            self.eval()
            x_new = x.clone().detach()
            x_new = Variable(x_new.data, requires_grad=True)
            x_new_view = self.avgpool(x_new)
            x_new_view = x_new_view.view(x_new_view.size(0), -1)
            output = self.class_classifier(x_new_view)
            class_num = output.shape[1]
            index = gt
            num_rois = x_new.shape[0]
            num_channel = x_new.shape[1]
            H = x_new.shape[2]
            HW = x_new.shape[2] * x_new.shape[3]
            # one_hot = torch.zeros((1), dtype=torch.float32).cuda()
            # one_hot = Variable(one_hot, requires_grad=False)
            sp_i = torch.ones([2, num_rois]).long()
            sp_i[0, :] = torch.arange(num_rois)
            sp_i[1, :] = index
            sp_v = torch.ones([num_rois])
            one_hot_sparse = torch.sparse.FloatTensor(sp_i, sp_v, torch.Size([num_rois, class_num])).to_dense().cuda()
            one_hot_sparse = Variable(one_hot_sparse, requires_grad=False)
            one_hot = torch.sum(output * one_hot_sparse)
            self.zero_grad()
            one_hot.backward()
            grads_val = x_new.grad.clone().detach()
            grad_channel_mean = torch.mean(grads_val.view(num_rois, num_channel, -1), dim=2)
            channel_mean = grad_channel_mean
            grad_channel_mean = grad_channel_mean.view(num_rois, num_channel, 1, 1)
            spatial_mean = torch.sum(x_new * grad_channel_mean, 1)
            spatial_mean = spatial_mean.view(num_rois, HW)
            self.zero_grad()

            choose_one = random.randint(0, 9)
            if choose_one <= 4:
                # ---------------------------- spatial -----------------------
                spatial_drop_num = math.ceil(HW * 1 / 3.0)
                th18_mask_value = torch.sort(spatial_mean, dim=1, descending=True)[0][:, spatial_drop_num]
                th18_mask_value = th18_mask_value.view(num_rois, 1).expand(num_rois, 49)
                mask_all_cuda = torch.where(spatial_mean > th18_mask_value, torch.zeros(spatial_mean.shape).cuda(),
                                            torch.ones(spatial_mean.shape).cuda())
                mask_all = mask_all_cuda.reshape(num_rois, H, H).view(num_rois, 1, H, H)
            else:
                # -------------------------- channel ----------------------------
                vector_thresh_percent = math.ceil(num_channel * 1 / 3.2)
                vector_thresh_value = torch.sort(channel_mean, dim=1, descending=True)[0][:, vector_thresh_percent]
                vector_thresh_value = vector_thresh_value.view(num_rois, 1).expand(num_rois, num_channel)
                vector = torch.where(channel_mean > vector_thresh_value,
                                     torch.zeros(channel_mean.shape).cuda(),
                                     torch.ones(channel_mean.shape).cuda())
                mask_all = vector.view(num_rois, num_channel, 1, 1)

            # ----------------------------------- batch ----------------------------------------
            cls_prob_before = F.softmax(output, dim=1)
            x_new_view_after = x_new * mask_all
            x_new_view_after = self.avgpool(x_new_view_after)
            x_new_view_after = x_new_view_after.view(x_new_view_after.size(0), -1)
            x_new_view_after = self.class_classifier(x_new_view_after)
            cls_prob_after = F.softmax(x_new_view_after, dim=1)

            sp_i = torch.ones([2, num_rois]).long()
            sp_i[0, :] = torch.arange(num_rois)
            sp_i[1, :] = index
            sp_v = torch.ones([num_rois])
            one_hot_sparse = torch.sparse.FloatTensor(sp_i, sp_v, torch.Size([num_rois, class_num])).to_dense().cuda()
            before_vector = torch.sum(one_hot_sparse * cls_prob_before, dim=1)
            after_vector = torch.sum(one_hot_sparse * cls_prob_after, dim=1)
            change_vector = before_vector - after_vector - 0.0001
            change_vector = torch.where(change_vector > 0, change_vector, torch.zeros(change_vector.shape).cuda())
            th_fg_value = torch.sort(change_vector, dim=0, descending=True)[0][int(round(float(num_rois) * self.pecent))]
            drop_index_fg = change_vector.gt(th_fg_value).long()
            ignore_index_fg = 1 - drop_index_fg
            not_01_ignore_index_fg = ignore_index_fg.nonzero()[:, 0]
            mask_all[not_01_ignore_index_fg.long(), :] = 1

            self.train()
            mask_all = Variable(mask_all, requires_grad=True)
            x = x * mask_all

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.class_classifier(x)
        if ganstyle and self.training:
            return x, mean_out_real, std_out_real, mean_out_fake, std_out_fake
        return x


def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model

def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model

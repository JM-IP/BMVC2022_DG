# coding=UTF-8
from torch import nn
from torch.utils import model_zoo
from torchvision.models.resnet import model_urls, Bottleneck
import torch
from torch import nn as nn
from torch.autograd import Variable
import numpy.random as npr
import numpy as np
import torch.nn.functional as F
import random
import math
from models.permuteAdaIN import AdaptiveInstanceNorm2d, calc_mean_std

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class BasicBlock_gananstyle(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, with_style_adain=False):
        super(BasicBlock_gananstyle, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.with_style_adain = with_style_adain
        if with_style_adain:
            self.stylegen1 = Generator(planes)
            self.styledis1mean = Discriminator_Vector(planes)
            self.styledis1std = Discriminator_Vector(planes)
            self.stylegen2 = Generator(planes * BasicBlock.expansion)
            self.styledis2mean = Discriminator_Vector(planes * BasicBlock.expansion)
            self.styledis2std = Discriminator_Vector(planes * BasicBlock.expansion)
            self.conv1_style_adain = AdaptiveInstanceNorm2d(planes * BasicBlock.expansion)
            self.conv2_style_adain = AdaptiveInstanceNorm2d(planes * BasicBlock.expansion)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, noise=None, ganstyle=False, get_loss=True, get_mean_std=False):
        identity = x

        out = self.conv1(x)
        if ganstyle and noise is not None:
            mean1_fake, std1_fake = self.stylegen1(noise)
            mean1_real, std1_real = calc_mean_std(out)
            out = self.conv1_style_adain(out, mean1_fake, std1_fake)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if ganstyle and noise is not None:
            mean2_fake, std2_fake = self.stylegen2(noise)
            mean2_real, std2_real = calc_mean_std(out)
            out = self.conv2_style_adain(out, mean2_fake, std2_fake)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        if get_loss and ganstyle and noise is not None and self.training:
            assert get_mean_std==False
            return [out, mean1_real, std1_real, mean2_real, std2_real,
                    mean1_fake, std1_fake, mean2_fake, std2_fake]
        else:
            return out
# ref to https://github.com/ShuvozitGhose/Shadow-Detection-RESNET/blob/master/ShadowNet.py

class TransBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
        super(TransBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, inplanes)
        # self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        if upsample is not None and stride != 1:
            self.conv2 = nn.ConvTranspose2d(inplanes, planes,
                                            kernel_size=3, stride=stride, padding=1,
                                            output_padding=1, bias=False)
        else:
            self.conv2 = conv3x3(inplanes, planes, stride)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, jigsaw_classes=1000, classes=100, with_style_adain=False, p_adain=None, type='image+feat'):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.p_adain = p_adain
        self.type = type
        self.with_style_adain = with_style_adain
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.jigsaw_classifier = nn.Linear(512 * block.expansion, jigsaw_classes)
        self.class_classifier = nn.Linear(512 * block.expansion, classes)
        #self.domain_classifier = nn.Linear(512 * block.expansion, domains)
        self.pecent = 1/3

        transblock = TransBasicBlock # 2 convolution + 1 deconvolutoin decoder block
        self.conv5 = nn.Conv2d(512, 512, kernel_size=1, bias=False)

        #Decoder part
        self.inplanes = 512
        self.deconv1 = self._make_transpose(transblock, 256, 2, stride=2)
        self.deconv2 = self._make_transpose(transblock, 128, 2, stride=2)
        self.deconv3 = self._make_transpose(transblock, 64, 2, stride=2)
        self.deconv4 = self._make_transpose(transblock, 64, 2, stride=2)

        # final block
        self.inplanes = 64
        self.final_conv = self._make_transpose(transblock, 64, 3)

        self.upsam = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2,
                                        padding=0, bias=False)

        self.conv6 = conv3x3(32, 16)
        self.conv7 = conv3x3(16, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if with_style_adain:
            self.styleconv1gen = Generator(64)
            self.styleconv1meandis = Discriminator_Vector(64)
            self.styleconv1stddis = Discriminator_Vector(64)
            self.stylegen1 = Generator(64)
            self.stylegen2 = Generator(128)
            self.stylegen3 = Generator(256)
            self.stylegen4 = Generator(512)
            self.styledis1mean = Discriminator_Vector(64)
            self.styledis2mean = Discriminator_Vector(128)
            self.styledis3mean = Discriminator_Vector(256)
            self.styledis4mean = Discriminator_Vector(512)
            self.styledis1std = Discriminator_Vector(64)
            self.styledis2std = Discriminator_Vector(128)
            self.styledis3std = Discriminator_Vector(256)
            self.styledis4std = Discriminator_Vector(512)
            self.gen = [self.styleconv1gen,
                        self.stylegen1, self.stylegen2,
                        self.stylegen3, self.stylegen4]
            self.mean_dis = [self.styleconv1meandis,
                             self.styledis1mean, self.styledis2mean,
                             self.styledis3mean, self.styledis4mean,]
            self.std_dis = [self.styleconv1stddis,
                            self.styledis1std, self.styledis2std,
                            self.styledis3std, self.styledis4std,]
            if 'image' in self.type:
                self.mse_loss = nn.MSELoss()
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_transpose(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes,
                                   kernel_size=2, stride=stride,
                                   padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []

        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)

    def is_patch_based(self):
        return False

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        # assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        # assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def forward(self, x, gt=None, flag=None, epoch=None, noise=None, randyes=False, type='feat+image'):
        # randyes is base on random func. before calling forward
        ganstyle = randyes and self.with_style_adain
        if self.training and ganstyle:
            mean_out_real = []
            mean_out_fake = []
            std_out_real = []
            std_out_fake = []
        x = self.conv1(x)
        if ganstyle and self.training and 'feat' in type:
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
        if 'image' in self.type:
            outx = [x]
        for i in range(4):
            # [self.layer1, self.layer2, self.layer3, self.layer4]:
            layerfunc = getattr(self, 'layer{:d}'.format(i + 1))
            x = layerfunc(x)
            if ganstyle and self.training:
                mean_real, std_real = calc_mean_std(x)
                mean_out_real.append(mean_real)
                std_out_real.append(std_real)
                genfunc = getattr(self, 'stylegen{:d}'.format(i + 1))
                mean_fake, std_fake = genfunc(noise)
                mean_out_fake.append(mean_fake)
                std_out_fake.append(std_fake)
                x = AdaptiveInstanceNorm2d()(x, mean_fake, std_fake)
                if 'image' in self.type:
                    outx.append(x)
        if ganstyle and self.training and 'image' in type:

            decode_input = x
            # print(decode_input.size())
            decode_input = self.conv5(decode_input)
            # print(decode_input.size())

            # Decoder forward
            decode_input = self.deconv1(decode_input)
            # print(decode_input.size())
            decode_input = self.deconv2(decode_input)
            # print(decode_input.size())
            decode_input = self.deconv3(decode_input)
            # print(decode_input.size())
            decode_input = self.deconv4(decode_input)
            # print(decode_input.size())

            # final convolution
            decode_input = self.final_conv(decode_input)
            # print(decode_input.size())
            decode_input = self.upsam(decode_input)
            # print(decode_input.size())
            decode_input = self.conv6(decode_input)
            # print(decode_input.size())
            image_out_fake = self.conv7(decode_input)
            # print(decode_input.size())
            # image_out_fake
            image_out_fake_feat = self.conv1(image_out_fake)
            image_out_fake_feat = self.bn1(image_out_fake_feat)
            image_out_fake_feat = self.relu(image_out_fake_feat)
            image_out_fake_feat = self.maxpool(image_out_fake_feat)
            image_out_fake_feat = [image_out_fake_feat]
            for i in range(4):
                # [self.layer1, self.layer2, self.layer3, self.layer4]:
                layerfunc = getattr(self, 'layer{:d}'.format(i + 1))
                image_out_fake_feat.append(layerfunc(image_out_fake_feat[-1]))
            # for layers in [self.layer1, self.layer2, self.layer3, self.layer4]:
            #     for layer in layers:
            #         image_out_fake_feat.append(layer(image_out_fake_feat[-1]))
            loss_c = self.calc_content_loss(image_out_fake_feat[-1], x)
            loss_s = self.calc_style_loss(image_out_fake_feat[0], outx[0])
            for i in range(1, len(image_out_fake_feat)-1):
                loss_s += self.calc_style_loss(image_out_fake_feat[i], outx[i])
        #print(x.size())

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
            one_hot = torch.zeros((1), dtype=torch.float32).cuda()
            one_hot = Variable(one_hot, requires_grad=False)
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
        out ={"class_logit": x}
        if ganstyle and self.training:
            if 'feat' in type:
                out.update({"mean_out_real": mean_out_real,
                    "std_out_real": std_out_real,
                    "mean_out_fake": mean_out_fake,
                    "std_out_fake": std_out_fake})
            if 'image' in type:
                out.update({"loss_c": loss_c, "loss_s": loss_s, "image_out_fake": image_out_fake})
            return out
        return x

    def get_allinfo(self, x, noise=None, randyes=True, type='feat+image'):
        # randyes is base on random func. before calling forward
        ganstyle = randyes and self.with_style_adain
        if ganstyle:
            mean_out_real = []
            mean_out_fake = []
            std_out_real = []
            std_out_fake = []
        x = self.conv1(x)
        if ganstyle and 'feat' in type:
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
        if 'image' in self.type:
            outx = [x]
        for i in range(4):
            # [self.layer1, self.layer2, self.layer3, self.layer4]:
            layerfunc = getattr(self, 'layer{:d}'.format(i + 1))
            x = layerfunc(x)
            if ganstyle:
                mean_real, std_real = calc_mean_std(x)
                mean_out_real.append(mean_real)
                std_out_real.append(std_real)
                genfunc = getattr(self, 'stylegen{:d}'.format(i + 1))
                mean_fake, std_fake = genfunc(noise)
                mean_out_fake.append(mean_fake)
                std_out_fake.append(std_fake)
                x = AdaptiveInstanceNorm2d()(x, mean_fake, std_fake)
                if 'image' in self.type:
                    outx.append(x)
        if ganstyle and 'image' in type:

            decode_input = x
            # print(decode_input.size())
            decode_input = self.conv5(decode_input)
            # print(decode_input.size())

            # Decoder forward
            decode_input = self.deconv1(decode_input)
            # print(decode_input.size())
            decode_input = self.deconv2(decode_input)
            # print(decode_input.size())
            decode_input = self.deconv3(decode_input)
            # print(decode_input.size())
            decode_input = self.deconv4(decode_input)
            # print(decode_input.size())

            # final convolution
            decode_input = self.final_conv(decode_input)
            # print(decode_input.size())
            decode_input = self.upsam(decode_input)
            # print(decode_input.size())
            decode_input = self.conv6(decode_input)
            # print(decode_input.size())
            image_out_fake = self.conv7(decode_input)
            # print(decode_input.size())

            image_out_fake_feat = self.conv1(image_out_fake)
            image_out_fake_feat = self.bn1(image_out_fake_feat)
            image_out_fake_feat = self.relu(image_out_fake_feat)
            image_out_fake_feat = self.maxpool(image_out_fake_feat)
            image_out_fake_feat = [image_out_fake_feat]
            for i in range(4):
                # [self.layer1, self.layer2, self.layer3, self.layer4]:
                layerfunc = getattr(self, 'layer{:d}'.format(i + 1))
                image_out_fake_feat.append(layerfunc(image_out_fake_feat[-1]))
            # for layers in [self.layer1, self.layer2, self.layer3, self.layer4]:
            #     for layer in layers:
            #         image_out_fake_feat.append(layer(image_out_fake_feat[-1]))
            loss_c = self.calc_content_loss(image_out_fake_feat[-1], x)
            loss_s = self.calc_style_loss(image_out_fake_feat[0], outx[0])
            for i in range(1, len(image_out_fake_feat)-1):
                # print(len(image_out_fake_feat), len(outx), i)
                loss_s += self.calc_style_loss(image_out_fake_feat[i], outx[i])
        #print(x.size())
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.class_classifier(x)
        out ={"class_logit": x}
        if ganstyle:
            if 'feat' in type:
                out.update({"mean_out_real": mean_out_real,
                            "std_out_real": std_out_real,
                            "mean_out_fake": mean_out_fake,
                            "std_out_fake": std_out_fake})
            if 'image' in type:
                out.update({"loss_c": loss_c, "loss_s": loss_s, "image_out_fake": image_out_fake})
            return out
        return x


# def resnet18_gananstyle(pretrained=True, with_style_adain=True, **kwargs):
#     """Constructs a ResNet-18 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(BasicBlock_gananstyle, [2, 2, 2, 2], with_style_adain=with_style_adain, **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
#     return model

def resnet18_gananstyleV2(pretrained=True, with_style_adain=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], with_style_adain=with_style_adain, **kwargs)
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
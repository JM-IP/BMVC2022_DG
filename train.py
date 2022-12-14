#coding=UTF-8
import argparse

import torch
#from IPython.core.debugger import set_trace
from torch import nn
#from torch.nn import functional as F
from data import data_helper
## from IPython.core.debugger import set_trace
from data.data_helper import available_datasets
from optimizer.optimizer_helper import get_optim_and_scheduler
from utils.Logger import Logger
import numpy as np
from models.resnet import resnet18, resnet50
from models.resnet_binary import birealnet18
from models.ReActNet_binary import reactnet
from models.ReActNet_binary_reparam import reactnet as reactnet_reparam
from models.resnet_gananstyle import resnet18_gananstyle as resnet18_gananstyleV1
from models.resnet_gananstyleV2 import resnet18_gananstyleV2 as resnet18_gananstyleV2
from models.resnet_binary_clip import birealnet18 as birealnet18_clip
from torch import optim
import random
import os

def get_args():
    parser = argparse.ArgumentParser(description="Script to launch jigsaw training",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source", help="Source", nargs='+')
    parser.add_argument("--target", help="Target")
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    # data aug stuff
    parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
    parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    parser.add_argument("--random_horiz_flip", default=0.5, type=float, help="Chance of random horizontal flip")
    parser.add_argument("--jitter", default=0.4, type=float, help="Color jitter amount")
    parser.add_argument("--tile_random_grayscale", default=0.1, type=float, help="Chance of randomly greyscaling a tile")
    #
    parser.add_argument("--limit_source", default=None, type=int,
                        help="If set, it will limit the number of training samples")
    parser.add_argument("--limit_target", default=None, type=int,
                        help="If set, it will limit the number of testing samples")
    parser.add_argument("--learning_rate", "-l", type=float, default=.01, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=20, help="Number of epochs")
    parser.add_argument("--warming_epoch_total", type=int, default=0, help="Number of warming epochs")
    parser.add_argument("--n_classes", "-c", type=int, default=7, help="Number of classes")
    parser.add_argument("--network", help="Which network to use", default="resnet18")
    parser.add_argument("--tf_logger", type=bool, default=True, help="If true will save tensorboard compatible logs")
    parser.add_argument("--val_size", type=float, default="0.1", help="Validation size (between 0 and 1)")
    parser.add_argument("--folder_name", default='test', help="Used by the logger to save logs")
    parser.add_argument("--bias_whole_image", default=0.9, type=float, help="If set, will bias the training procedure to show more often the whole image")
    parser.add_argument("--TTA", type=bool, default=False, help="Activate test time data augmentation")
    parser.add_argument("--classify_only_sane", default=False, type=bool, help="If true, the network will only try to classify the non scrambled images")
    parser.add_argument("--train_all", default=True, type=bool, help="If true, all network weights will be trained")
    parser.add_argument("--suffix", default="", help="Suffix for the logger")
    parser.add_argument("--nesterov", default=False, type=bool, help="Use nesterov")
    parser.add_argument("--flag", default=False, action='store_true', help="Use RSC")
    parser.add_argument("--seed", type=int, default=0, help="seed value")
    parser.add_argument("--with_style_adain", default=False, action='store_true', help="Use GananStyle")
    parser.add_argument("--p_adain", default=0.01, type=float, help="")
    parser.add_argument("--beta", default=0.01, type=float, help="")
    parser.add_argument("--warming", action='store_true', default=False, help="Use RSC")
    parser.add_argument("--log_path", default='/shuju/yjm/DG/', type=str, help="log_path")
    parser.add_argument("--dataset", choices=['PACS', 'office', 'office_home', 'VLCS', 'DomainNet'], help="Which dataset to use", default="PACS")
    parser.add_argument("--data_path", default='/shuju/yjm/data/', type=str, help="data_path")
    parser.add_argument("--type", default='feat+image', type=str, help="log_path")
    parser.add_argument("--var_type", default='last', type=str, help="")
    parser.add_argument("--cal_var_loss", default=False, action='store_true', help="Use var loss")
    parser.add_argument("--pretrained", default=False, action='store_true', help="Use autoaug")
    parser.add_argument("--var_loss_weights", default=0.01, type=float, help="")
    parser.add_argument("--mean_loss_weights", default=0, type=float, help="")
    parser.add_argument("--autoaug", default=False, action='store_true', help="Use autoaug")
    parser.add_argument("--optimizer_type", default='Adam', type=str, help="")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="")
    return parser.parse_args()

class Trainer:
    def __init__(self, args, device):
        self.logger = Logger(args, update_frequency=30)
        self.log_path = self.logger.log_path
        self.args = args
        self.flag = args.flag
        self.device = device
        self.with_style_adain = args.with_style_adain
        if args.network == 'resnet18':
            model = resnet18(pretrained=True, classes=args.n_classes)
        elif args.network == 'resnet50':
            model = resnet50(pretrained=True, classes=args.n_classes)
        elif args.network == 'resnet18_binary':
            model = birealnet18(pretrained=args.pretrained, classes=args.n_classes, cal_var_loss=args.cal_var_loss, var_type=args.var_type)
        elif args.network == 'resnet18_binary_clip':
            model = birealnet18_clip(pretrained=args.pretrained, classes=args.n_classes)
        elif args.network == 'reactnet':
            model = reactnet(pretrained=True, classes=args.n_classes, cal_var_loss=args.cal_var_loss, var_type=args.var_type)
        elif args.network == 'reactnet_reparam':
            model = reactnet_reparam(pretrained=True, classes=args.n_classes, cal_var_loss=args.cal_var_loss, var_type=args.var_type)
        elif args.network == 'resnet18_gananstyleV1':
            model = resnet18_gananstyleV1(pretrained=True, classes=args.n_classes, with_style_adain=args.with_style_adain, p_adain=args.p_adain)
        elif args.network == 'resnet18_gananstyleV2':
            model = resnet18_gananstyleV2(pretrained=True, classes=args.n_classes, with_style_adain=args.with_style_adain, p_adain=args.p_adain, type=args.type)
        else:
            raise("model type is not defined!")
        self.model = model.to(device)
        print(self.model)
        if args.dataset in ['PACS', 'DomainNet', 'VLCS']:# val set exists
            self.source_loader, self.val_loader = data_helper.get_train_dataloader(args, patches=model.is_patch_based())
            self.target_loader = data_helper.get_val_dataloader(args, patches=model.is_patch_based())
            self.test_loaders = {"val": self.val_loader, "test": self.target_loader}

            self.len_dataloader = len(self.source_loader)
            print("Dataset size: train %d, val %d, test %d" % (
            len(self.source_loader.dataset), len(self.val_loader.dataset), len(self.target_loader.dataset)))
        else:# no val set
            self.source_loader, _ = data_helper.get_train_dataloader(args, patches=model.is_patch_based())
            self.target_loader = data_helper.get_val_dataloader(args, patches=model.is_patch_based())
            self.test_loaders = {"test": self.target_loader}
            self.len_dataloader = len(self.source_loader)
            print("Dataset size: train %d, test %d" % (len(self.source_loader.dataset), len(self.target_loader.dataset)))
        # self.optimizer, self.scheduler = get_optim_and_scheduler(model, args.epochs, args.learning_rate, args.train_all,
        #                                                          nesterov=args.nesterov)
        args.beta1 = 0.5
        args.ganlr = 0.0002
        if args.with_style_adain:
            self.optimizerD = optim.Adam([{'params': i.parameters()} for i in self.model.mean_dis] +
                                         [{'params': i.parameters()} for i in self.model.std_dis],
                                         lr=args.ganlr, betas=(args.beta1, 0.999))
            self.optimizerG =optim.Adam([{'params': i.parameters()} for i in self.model.gen], lr=args.ganlr, betas=(args.beta1, 0.999))
        params_list=[]
        # a = [] + \
        #     [i.named_parameters() for i in self.model.std_dis] + \
        #     [i.named_parameters() for i in self.model.gen]
        param_dict = dict(self.model.named_parameters())
        for key, value in param_dict.items():
            if "style" not in key:
                print("updating " + key)
                params_list.append({'params':  value})
        if args.optimizer_type=='SGD':
            self.optimizer = optim.SGD(params_list, weight_decay=.0005, momentum=.9, nesterov=args.nesterov, lr=args.learning_rate)
        elif args.optimizer_type=='Adam':
            self.optimizer = optim.Adam(params_list, weight_decay=args.weight_decay, lr=args.learning_rate)

        step_size = int(args.epochs * .8)
        if self.with_style_adain:
            step_size = int(args.epochs * .8+ args.warming_epoch_total)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size)
        print("Step size: %d" % step_size)
        self.n_classes = args.n_classes
        if args.target in args.source:
            self.target_id = args.source.index(args.target)
            print("Target in source: %d" % self.target_id)
            print(args.source)
        else:
            self.target_id = None
        self.start_epoch=0
        checkpoint_tar=os.path.join(self.log_path, "checkpoint_current.pth")
        if os.path.exists(checkpoint_tar):
            print('loading checkpoint {} ..........'.format(checkpoint_tar))
            checkpoint = torch.load(checkpoint_tar)
            self.start_epoch = checkpoint['epoch'] + 1
            self.logger.current_epoch = self.start_epoch
            best_top1_acc = checkpoint['test_acc']

            self.model.load_state_dict(checkpoint['state_dict'], strict=True)
            print("loaded checkpoint {} epoch = {}" .format(checkpoint_tar, checkpoint['epoch']))
            print("best acc {}" .format(best_top1_acc))
            with torch.no_grad():
                for phase, loader in self.test_loaders.items():
                    final_out = self.do_test(loader)
                    total = len(loader.dataset)
                    class_correct = final_out["class_correct"]
                    class_acc = float(class_correct) / total
                    log_loss = {"class_fp": class_acc}
                    print(phase, log_loss)
        else:
            checkpoint_tar=os.path.join(self.log_path, "best_checkpoint_test.pth")
            if os.path.exists(checkpoint_tar):
                print('loading checkpoint {} ..........'.format(checkpoint_tar))
                checkpoint = torch.load(checkpoint_tar)
                self.start_epoch = checkpoint['epoch'] + 1
                self.logger.current_epoch = self.start_epoch

                self.model.load_state_dict(checkpoint['state_dict'], strict=True)
                print("loaded checkpoint {} epoch = {}" .format(checkpoint_tar, checkpoint['epoch']))



    def _do_warming_epoch(self, epoch=None):
        criterion_BCE = nn.BCELoss()
        for it, ((data, jig_l, class_l), d_idx) in enumerate(self.source_loader):
            data, jig_l, class_l, d_idx = data.to(self.device), jig_l.to(self.device), \
                                          class_l.to(self.device), d_idx.to(self.device)

            data_flip = torch.flip(data, (3,)).detach().clone()
            data = torch.cat((data, data_flip))
            class_l = torch.cat((class_l, class_l))

            noise = torch.randn(data.size()[0], 64).cuda()
            out = self.model(data, gt=class_l, flag=self.flag, epoch=epoch, noise=noise, randyes=True)
            mean_out_real = out['mean_out_real']
            std_out_real = out['std_out_real']
            mean_out_fake = out['mean_out_fake']
            std_out_fake = out['std_out_fake']
            fake_label = torch.full((data.size()[0],), 0.0).cuda()
            real_label = torch.full((data.size()[0],), 1.0).cuda()
            for dis in self.model.mean_dis:
                dis.zero_grad()
            for dis in self.model.std_dis:
                dis.zero_grad()
            D_G_z1_mean = []
            errD_fake_mean = 0
            for i, mean in enumerate(mean_out_fake):
                output = self.model.mean_dis[i](mean.detach()).view(-1)
                # ???????????????D????????????????????????????????????
                errD_fake_mean += criterion_BCE(output, fake_label)
                D_G_z1_mean.append(output.mean().item())
            D_G_z1_std = []
            errD_fake_std = 0
            for i, std in enumerate(std_out_fake):
                output = self.model.std_dis[i](std.detach()).view(-1)
                # ???????????????D????????????????????????????????????
                errD_fake_std += criterion_BCE(output, fake_label)
                D_G_z1_std.append(output.mean().item())
            D_x_mean = []
            errD_real_mean = 0
            for i, mean in enumerate(mean_out_real):
                output = self.model.mean_dis[i](mean.view(mean.shape[0],mean.shape[1]).detach()).view(-1)
                # ???????????????D????????????????????????????????????
                if (output<0).any() or (output>1).any():
                    print(output)
                errD_real_mean += criterion_BCE(output, real_label)
                D_x_mean.append(output.mean().item())
            D_x_std = []
            errD_real_std = 0
            for i, std in enumerate(std_out_real):
                output = self.model.std_dis[i](std.view(std.shape[0],std.shape[1]).detach()).view(-1)
                # ???????????????D????????????????????????????????????
                errD_real_std += criterion_BCE(output, real_label)
                D_x_std.append(output.mean().item())
            errD_fake_mean.backward()
            errD_fake_std.backward()
            errD_real_mean.backward()
            errD_real_std.backward()
            self.optimizerD.step()

            ############################
            # (2) ?????? G ??????: ????????? log(D(G(z)))
            ###########################
            for i in self.model.gen:
                i.zero_grad()
            # ????????????????????????D?????????D??????????????????????????????????????????
            errG_mean = 0
            D_G_z2_mean = []
            for i, mean in enumerate(mean_out_fake):
                output = self.model.mean_dis[i](mean.detach()).view(-1)
                D_G_z2_mean.append(output.mean().item())
                errG_mean  += criterion_BCE(output, real_label)

            errG_std = 0
            D_G_z2_std = []
            for i, std in enumerate(std_out_fake):
                output = self.model.std_dis[i](std.detach()).view(-1)
                errG_std += criterion_BCE(output, real_label)
                D_G_z2_std.append(output.mean().item())
            errG_mean.backward()
            errG_std.backward()
            self.optimizerG.step()

            if 'image' in self.args.type:
                # TODO:
                loss_c = out['loss_c']
                loss_s = out['loss_s']
                ((loss_s + loss_c) * self.args.beta).backward()
                self.optimizer.step()

            log_loss = {}
            for num, errloss in enumerate(D_G_z1_mean):
                log_loss.update({'D_G_z1_mean'+str(num): errloss})
            for num, errloss in enumerate(D_G_z1_std):
                log_loss.update({'D_G_z1_std'+str(num): errloss})
            for num, errloss in enumerate(D_x_mean):
                log_loss.update({'D_x_mean'+str(num): errloss})
            for num, errloss in enumerate(D_x_std):
                log_loss.update({'D_x_std'+str(num): errloss})
            for num, errloss in enumerate(D_G_z2_mean):
                log_loss.update({'D_G_z2_mean'+str(num): errloss})
            for num, errloss in enumerate(D_G_z2_std):
                log_loss.update({'D_G_z2_std'+str(num): errloss})
            if 'image' in self.args.type:
                log_loss.update({'loss_c': loss_c.item()})
                log_loss.update({'loss_s': loss_s.item()})
            self.logger.log(it, len(self.source_loader),
                            log_loss,
                            {"class":0, }, data.shape[0])
            del log_loss


    def _do_epoch(self, epoch=None):
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        if self.with_style_adain:
            criterion_BCE = nn.BCELoss()
        for it, ((data, jig_l, class_l), d_idx) in enumerate(self.source_loader):
            data, jig_l, class_l, d_idx = data.to(self.device), jig_l.to(self.device), class_l.to(self.device), d_idx.to(self.device)
            self.optimizer.zero_grad()

            randyes = self.args.with_style_adain and random.random() < self.model.p_adain
            data_flip = torch.flip(data, (3,)).detach().clone()
            data = torch.cat((data, data_flip))
            class_l = torch.cat((class_l, class_l))
            loss = 0
            if randyes:
                noise = torch.randn(data.size()[0], 64).cuda()
                out = self.model(data, gt=class_l, flag=self.flag, epoch=epoch, noise=noise, randyes=randyes)
                class_logit = out['class_logit']
                if 'feat' in self.args.type:
                    mean_out_real = out['mean_out_real']
                    std_out_real = out['std_out_real']
                    mean_out_fake = out['mean_out_fake']
                    std_out_fake = out['std_out_fake']
                    fake_label = torch.full((data.size()[0],), 0).cuda()
                    real_label = torch.full((data.size()[0],), 1).cuda()
                    for dis in self.model.mean_dis:
                        dis.zero_grad()
                    for dis in self.model.std_dis:
                        dis.zero_grad()
                    D_G_z1_mean = []
                    errD_fake_mean = 0
                    for i, mean in enumerate(mean_out_fake):
                        output = self.model.mean_dis[i](mean.detach()).view(-1)
                        # ???????????????D????????????????????????????????????
                        errD_fake_mean += criterion_BCE(output, fake_label)
                        D_G_z1_mean.append(output.mean().item())
                    D_G_z1_std = []
                    errD_fake_std = 0
                    for i, std in enumerate(std_out_fake):
                        output = self.model.std_dis[i](std.detach()).view(-1)
                        # ???????????????D????????????????????????????????????
                        errD_fake_std += criterion_BCE(output, fake_label)
                        D_G_z1_std.append(output.mean().item())
                    D_x_mean = []
                    errD_real_mean = 0
                    for i, mean in enumerate(mean_out_real):
                        output = self.model.mean_dis[i](mean.view(mean.shape[0],mean.shape[1]).detach()).view(-1)
                        # ???????????????D????????????????????????????????????
                        errD_real_mean += criterion_BCE(output, real_label)
                        D_x_mean.append(output.mean().item())
                    D_x_std = []
                    errD_real_std = 0
                    for i, std in enumerate(std_out_real):
                        output = self.model.std_dis[i](std.view(std.shape[0],std.shape[1]).detach()).view(-1)
                        # ???????????????D????????????????????????????????????
                        errD_real_std += criterion_BCE(output, real_label)
                        D_x_std.append(output.mean().item())
                    errD_fake_mean.backward()
                    errD_fake_std.backward()
                    errD_real_mean.backward()
                    errD_real_std.backward()
                    self.optimizerD.step()

                    ############################
                    # (2) ?????? G ??????: ????????? log(D(G(z)))
                    ###########################
                    for i in self.model.gen:
                        i.zero_grad()
                    # ????????????????????????D?????????D??????????????????????????????????????????
                    errG_mean = 0
                    D_G_z2_mean = []
                    for i, mean in enumerate(mean_out_fake):
                        output = self.model.mean_dis[i](mean.detach()).view(-1)
                        D_G_z2_mean.append(output.mean().item())
                        errG_mean  += criterion_BCE(output, real_label)

                    errG_std = 0
                    D_G_z2_std = []
                    for i, std in enumerate(std_out_fake):
                        output = self.model.std_dis[i](std.detach()).view(-1)
                        errG_std += criterion_BCE(output, real_label)
                        D_G_z2_std.append(output.mean().item())
                    errG_mean.backward()
                    errG_std.backward()
                    self.optimizerG.step()
                if 'image' in self.args.type:
                    # TODO:
                    loss_c = out['loss_c']
                    loss_s = out['loss_s']
                    loss += (loss_s + loss_c) * self.args.beta
                    # self.optimizer.step()
            else:
                out = self.model(data, class_l, self.flag, epoch)
                class_logit = out['class_logit']
            if self.args.cal_var_loss:
                loss += -out['var_loss'] * self.args.var_loss_weights
                loss += out['mean_loss'] * self.args.mean_loss_weights
            class_loss = criterion(class_logit, class_l)
            _, cls_pred = class_logit.max(dim=1)
            loss += class_loss

            loss.backward()
            self.optimizer.step()

            log_loss = {"class": class_loss.item()}
            if randyes:
                if 'feat' in self.args.type:
                    for num, errloss in enumerate(D_G_z1_mean):
                        log_loss.update({'D_G_z1_mean'+str(num): errloss})
                    for num, errloss in enumerate(D_G_z1_std):
                        log_loss.update({'D_G_z1_std'+str(num): errloss})
                    for num, errloss in enumerate(D_x_mean):
                        log_loss.update({'D_x_mean'+str(num): errloss})
                    for num, errloss in enumerate(D_x_std):
                        log_loss.update({'D_x_std'+str(num): errloss})
                    for num, errloss in enumerate(D_G_z2_mean):
                        log_loss.update({'D_G_z2_mean'+str(num): errloss})
                    for num, errloss in enumerate(D_G_z2_std):
                        log_loss.update({'D_G_z2_std'+str(num): errloss})
                if 'image' in self.args.type:
                    log_loss.update({'loss_c': loss_c.item()})
                    log_loss.update({'loss_s': loss_s.item()})
            if self.args.cal_var_loss:
                log_loss.update({'var_loss(bigger better)': out['var_loss'].item()})
            self.logger.log(it, len(self.source_loader), log_loss,
                            {"class": torch.sum(cls_pred == class_l.data).item(), }, data.shape[0])
            # del loss, class_loss, log_loss

        self.model.eval()
        with torch.no_grad():
            for phase, loader in self.test_loaders.items():
                total = len(loader.dataset)

                final_out = self.do_test(loader)
                class_correct = final_out["class_correct"]
                class_acc = float(class_correct) / total
                log_loss = {"class": class_acc}
                if self.args.cal_var_loss:
                    log_loss.update({'test_var_loss(bigger better)': final_out["var_loss"]/ total})
                self.results[phase][self.current_epoch] = class_acc
                if self.args.cal_var_loss:
                    self.results_var_loss[phase][self.current_epoch] = final_out["var_loss"]/ total
                self.results_class_loss[phase][self.current_epoch] = final_out["class_loss"]/total
                self.logger.log_test(phase, log_loss)
            save_dict = {"state_dict": self.model.state_dict(), "epoch": epoch,
                         "train_class_loss": class_loss.item(),
                         "train_acc": torch.sum(cls_pred == class_l.data).item()/data.shape[0],
                         "test_class_loss": self.results_class_loss['test'][self.current_epoch],
                         "test_acc": self.results['test'][self.current_epoch],
                         "val_class_loss": self.results_class_loss['val'][self.current_epoch],
                         "val_acc": self.results['val'][self.current_epoch],
                         }
            if self.args.cal_var_loss:
                save_dict.update({"train_var_loss": out['var_loss'].item(),
                "test_var_loss": self.results_var_loss['test'][self.current_epoch],
                "val_var_loss": self.results_var_loss['val'][self.current_epoch],}
                )
            # if epoch>70:
            #     torch.save(save_dict, self.log_path+'/checkpoint_'+str(epoch)+'.pth')
            torch.save(save_dict, self.log_path+'/checkpoint_current.pth')
            for phase, loader in self.test_loaders.items():
                save_dict = {"state_dict": self.model.state_dict(), "epoch": epoch}
                checkpoint_path = self.log_path+'/best_checkpoint_'+phase+'.pth'
                if self.results[phase][self.current_epoch] >= self.results[phase].max():
                    for phase_in, loader_in in self.test_loaders.items():
                        save_dict.update({phase_in: self.results[phase_in][self.current_epoch]})
                    torch.save(save_dict, checkpoint_path)



    def do_test(self, loader):
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        class_correct = 0
        if self.args.cal_var_loss:
            var_loss=0
        for it, ((data, nouse, class_l), _) in enumerate(loader):
            data, nouse, class_l = data.to(self.device), nouse.to(self.device), class_l.to(self.device)

            out = self.model(data, class_l, False)
            class_logit = out['class_logit']
            _, cls_pred = class_logit.max(dim=1)

            class_correct += torch.sum(cls_pred == class_l.data)
            if self.args.cal_var_loss:
                var_loss += out['var_loss'].item()
            class_loss = criterion(class_logit, class_l)
        final_out = {"class_correct": class_correct, 'class_loss': class_loss}
        if self.args.cal_var_loss:
            final_out.update({'var_loss': var_loss})
        return final_out


    def do_training(self):

        self.results = {"val": torch.zeros(self.args.epochs+ + self.args.warming_epoch_total),
                        "test": torch.zeros(self.args.epochs+ + self.args.warming_epoch_total)}
        self.results_var_loss = {"val": torch.zeros(self.args.epochs+ + self.args.warming_epoch_total),
                        "test": torch.zeros(self.args.epochs+ + self.args.warming_epoch_total)}
        self.results_class_loss = {"val": torch.zeros(self.args.epochs+ + self.args.warming_epoch_total),
                        "test": torch.zeros(self.args.epochs+ + self.args.warming_epoch_total)}
        if self.with_style_adain:

            checkpoint_path = self.log_path+'/../warming_up_checkpoint_e'+str(self.args.warming_epoch_total)+'.pth'
            if os.path.exists(checkpoint_path):
                a = torch.load(checkpoint_path)
                # model_state = self.model.state_dict()
                # for k, v in a.items():
                #     name = k.replace("module.","")
                #     if 'fc' in name:
                #         continue
                #     if isinstance(v, nn.Parameter):
                #         model_state[name].copy_(v)
                #         print("load layer " + name)
                self.model.load_state_dict(a)
                print("load pre-trained gan")
            else:
                for self.warming_epoch in range(self.args.warming_epoch_total):
                    self.logger.new_epoch(self.scheduler.get_lr())
                    self._do_warming_epoch(self.warming_epoch)

                torch.save(self.model.state_dict(), checkpoint_path)
            self.current_epoch = self.args.warming_epoch_total
        else:
            self.args.warming_epoch_total = 0
        for self.current_epoch in range(0, self.start_epoch):
            self.scheduler.step()
        # insert loading
        for self.current_epoch in range(self.start_epoch, self.args.epochs+ self.args.warming_epoch_total):
            self.scheduler.step()
            if self.args.with_style_adain:
                for param_group in self.optimizerD.param_groups:
                    param_group['lr'] = self.scheduler.get_lr()[0]*0.1
                for param_group in self.optimizerG.param_groups:
                    param_group['lr'] = self.scheduler.get_lr()[0]*0.1
            self.logger.new_epoch(self.scheduler.get_lr())
            self._do_epoch(self.current_epoch)
        val_res = self.results["val"]
        test_res = self.results["test"]
        idx_best = val_res.argmax()
        print("Best val %g, corresponding test %g - best test: %g, best epoch: %g" % (
        val_res.max(), test_res[idx_best], test_res.max(), idx_best))
        self.logger.save_best(test_res[idx_best], test_res.max())
        return self.logger, self.model

def main():
    args = get_args()
    if args.var_type=='-':
        args.cal_var_loss=False
    print(args)
    if args.dataset=='PACS':
        domains = ['art_painting', 'cartoon', 'sketch', 'photo']
    elif args.dataset=='VLCS':
        domains = ["CALTECH", "LABELME", "PASCAL", "SUN"]
    elif args.dataset=='office':
        domains = ["amazon", "dslr", "webcam"]
    elif args.dataset=='office_home':
        domains = ['Art','Clipart','Product','Real_World']
    elif args.dataset=='DomainNet':
        domains = ['clipart','infograph','painting','real','quickdraw','sketch']
    args.source = domains
    args.source.remove(args.target)

    print("Target domain: {}".format(args.target))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(args, device)
    trainer.do_training()

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()

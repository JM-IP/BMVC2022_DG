import argparse

import torch
#from IPython.core.debugger import set_trace
from torch import nn
#from torch.nn import functional as F
from data import data_helper
from utils.Logger import Logger
from utils.mix_up import shuffle_minibatch
import numpy as np
from models.resnet import resnet18, resnet50
from models.resnet_binary import birealnet18
from models.resnet_binary_reparam import birealnet18 as birealnet18_reparam
from models.resnet_binary_clip import birealnet18 as birealnet18_clip
from models.ReActNet_binary import reactnet
from models.resnet_gananstyle import resnet18_gananstyle as resnet18_gananstyleV1
from models.resnet_gananstyleV2 import resnet18_gananstyleV2 as resnet18_gananstyleV2
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
    parser.add_argument("--pretrained", default='', type=str, help="Use pretrained")
    parser.add_argument("--var_loss_weights", default=0.01, type=float, help="")
    parser.add_argument("--mean_loss_weights", default=0, type=float, help="")
    parser.add_argument("--autoaug", default=False, action='store_true', help="Use autoaug")
    parser.add_argument("--optimizer_type", default='SGD', type=str, help="")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="")
    parser.add_argument("--model_path", type=str, help="model_path", required=True)
    parser.add_argument("--updating_values", type=str, help="model_path", required=True)
    # cartoon-photo-sketch_to_art_painting/eps100_bs32_lr0.01_class7_jigClass30_jigWeight0.7_dataset_PACS_resnet18_binary_TAll_bias0.9Var_last0_V2pretrained_useV2var0Vlast_365
    # art_painting-photo-sketch_to_cartoon/eps100_bs32_lr0.01_class7_jigClass30_jigWeight0.7_dataset_PACS_resnet18_binary_TAll_bias0.9Var_last0_V2pretrained_useV2var0Vlast_931
    # art_painting-cartoon-sketch_to_photo/eps100_bs32_lr0.01_class7_jigClass30_jigWeight0.7_dataset_PACS_resnet18_binary_TAll_bias0.9Var_last0_V2pretrained_useV2var0Vlast_608
    # art_painting-cartoon-photo_to_sketch/eps100_bs32_lr0.01_class7_jigClass30_jigWeight0.7_dataset_PACS_resnet18_binary_TAll_bias0.9Var_last0_V2pretrained_useV2var0Vlast_408

    parser.add_argument('--two_step', action='store_true', default=False, help='evaluate the model')
    parser.add_argument('--mix_up', action='store_true', default=False, help='evaluate the model')
    parser.add_argument('--loss_res_weight', type=float, default=0.001, help='loss_res_weight')
    parser.add_argument('--loss_fp_weight', type=float, default=0.01, help='loss_res_weight')
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
        elif args.network == 'resnet18_binary_reparam':
            model = birealnet18_reparam(pretrained=args.pretrained, classes=args.n_classes, cal_var_loss=args.cal_var_loss, var_type=args.var_type)
        elif args.network == 'resnet18_binary_clip':
            model = birealnet18_clip(pretrained=args.pretrained, classes=args.n_classes)
        elif args.network == 'reactnet':
            model = reactnet(pretrained=True, classes=args.n_classes, cal_var_loss=args.cal_var_loss, var_type=args.var_type)
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
        args.beta1 = 0.5
        args.ganlr = 0.0002
        if args.with_style_adain:
            self.optimizerD = optim.Adam([{'params': i.parameters()} for i in self.model.mean_dis] +
                                         [{'params': i.parameters()} for i in self.model.std_dis],
                                         lr=args.ganlr, betas=(args.beta1, 0.999))
            self.optimizerG =optim.Adam([{'params': i.parameters()} for i in self.model.gen], lr=args.ganlr, betas=(args.beta1, 0.999))
        self.model.load_state_dict(torch.load(args.model_path)['state_dict'])
        params_list=[]
        # a = [] + \
        #     [i.named_parameters() for i in self.model.std_dis] + \
        #     [i.named_parameters() for i in self.model.gen]
        param_dict = dict(self.model.named_parameters())
        updating_values = args.updating_values
        for key, value in param_dict.items():
            #print("questioning " + key)
            if "conv1.weight"==key:
                continue
            if "bn1.weight"==key:
                continue
            if "bn1.bias"==key:
                continue
            if "fc" in key and "fc" not in updating_values:
                continue
            if "binary_conv" in key and "binary_conv" not in updating_values:
                continue
            if "downsample.1" in key and "binary_conv" not in updating_values:
                continue
            if ("move" in key and "move" not in updating_values) or ("prelu" in key and "prelu" not in updating_values):
                continue
            if "bn" in key and "bn" not in updating_values:
                continue
            if "downsample.2" in key and "bn" not in updating_values:
                continue
            print("updating " + key)
            params_list.append({'params':  value})
        if args.optimizer_type=='SGD':
            self.optimizer = optim.SGD(params_list, weight_decay=args.weight_decay, momentum=.9, nesterov=args.nesterov, lr=args.learning_rate)
        elif args.optimizer_type=='Adam':
            self.optimizer = optim.Adam(params_list, weight_decay=args.weight_decay, lr=args.learning_rate)
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
        else:
            checkpoint_tar=os.path.join(self.log_path, "best_checkpoint_test.pth")
            if os.path.exists(checkpoint_tar):
                print('loading checkpoint {} ..........'.format(checkpoint_tar))
                checkpoint = torch.load(checkpoint_tar)
                self.start_epoch = checkpoint['epoch'] + 1
                self.logger.current_epoch = self.start_epoch

                self.model.load_state_dict(checkpoint['state_dict'], strict=True)
                print("loaded checkpoint {} epoch = {}" .format(checkpoint_tar, checkpoint['epoch']))

    def _do_epoch(self, epoch=None):
        criterion = nn.CrossEntropyLoss()
        self.model.train()

        for it, ((data, jig_l, class_l), d_idx) in enumerate(self.source_loader):
            self.model.train()
            self.optimizer.zero_grad()

            data_flip = torch.flip(data, (3,)).detach().clone()
            data = torch.cat((data, data_flip))
            class_l = torch.cat((class_l, class_l))

            # data_shuffle, class_l_shuffle = shuffle_minibatch(
            #     data, class_l, self.args.mix_up)
            data_shuffle, class_l_shuffle = data, class_l
            data, jig_l, class_l, d_idx = data.to(self.device), jig_l.to(self.device), class_l.to(self.device), d_idx.to(self.device)
            data_shuffle, class_l_shuffle = data_shuffle.to(self.device), class_l_shuffle.to(self.device)

            loss = 0
            out = self.model(x=data_shuffle, gt=class_l_shuffle, binary=True, flag=self.flag, epoch=epoch)
            # (self, x, binary=True, gt=None, flag=None, epoch=None)
            class_logit = out['class_logit']
            if self.args.cal_var_loss:
                loss += -out['var_loss'] * self.args.var_loss_weights
                loss += out['mean_loss'] * self.args.mean_loss_weights
            if self.args.mix_up:
                m = nn.LogSoftmax(dim=1)
                class_loss = -m(class_logit) * class_l_shuffle
                class_loss = torch.sum(class_loss) / self.args.batch_size
            else:
                class_loss = criterion(class_logit, class_l)
            _, cls_pred = class_logit.max(dim=1)
            loss += class_loss

            loss.backward()
            self.optimizer.step()

            log_loss = {"class": class_loss.item()}

            if self.args.cal_var_loss:
                log_loss.update({'var_loss(bigger better)': out['var_loss'].item()})
            self.logger.log(it, len(self.source_loader), log_loss,
                            {"class": torch.sum(cls_pred == class_l.data).item(), }, data.shape[0])

            if self.args.two_step:
                self.model.eval()
                out = self.model.forward(x=data_shuffle, gt=class_l_shuffle, binary=False, flag=self.flag, epoch=epoch)
                class_logit_fp, loss_res = out['class_logit'], out['res']
                loss_fp = criterion(class_logit_fp, class_l)
                loss = loss_fp * self.args.loss_fp_weight + loss_res * self.args.loss_res_weight
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                _, cls_pred_fp = class_logit_fp.max(dim=1)
                log_loss = {"class_fp": loss_fp.item()}

                self.logger.log(it, len(self.source_loader), log_loss,
                                {"class": torch.sum(cls_pred_fp == class_l.data).item(), }, data.shape[0])
            # del loss, class_loss, log_loss

        self.model.eval()
        with torch.no_grad():
            for phase, loader in self.test_loaders.items():
                total = len(loader.dataset)

                final_out = self.do_test(loader)
                class_correct = final_out["class_correct"]
                class_acc = float(class_correct) / total
                class_correct_fp = final_out["class_correct_fp"]
                class_acc_fp = float(class_correct_fp) / total
                log_loss = {"class": class_acc, "class_fp": class_acc_fp}
                if self.args.cal_var_loss:
                    log_loss.update({'test_var_loss(bigger better)': final_out["var_loss"]/ total})
                self.results[phase][self.current_epoch] = class_acc
                self.results_fp[phase][self.current_epoch] = class_acc_fp
                if self.args.cal_var_loss:
                    self.results_var_loss[phase][self.current_epoch] = final_out["var_loss"]/ total
                self.results_class_loss[phase][self.current_epoch] = final_out["class_loss"]/total
                self.results_class_loss_fp[phase][self.current_epoch] = final_out["class_loss_fp"]/total
                self.logger.log_test(phase, log_loss)
            save_dict = {"state_dict": self.model.state_dict(), "epoch": epoch,
                         "test_class_loss": self.results_class_loss['test'][self.current_epoch],
                         "test_acc": self.results['test'][self.current_epoch],
                         "val_class_loss": self.results_class_loss['val'][self.current_epoch],
                         "val_acc": self.results['val'][self.current_epoch],
                         "test_class_loss_fp": self.results_class_loss_fp['test'][self.current_epoch],
                         "test_acc_fp": self.results_fp['test'][self.current_epoch],
                         "val_class_loss_fp": self.results_class_loss_fp['val'][self.current_epoch],
                         "val_acc_fp": self.results_fp['val'][self.current_epoch],
                         }
            if self.args.cal_var_loss:
                save_dict.update({"train_var_loss": out['var_loss'].item(),
                "test_var_loss": self.results_var_loss['test'][self.current_epoch],
                "val_var_loss": self.results_var_loss['val'][self.current_epoch],}
                )
            torch.save(save_dict, self.log_path+'/checkpoint_'+str(epoch)+'.pth')
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
        class_correct_fp = 0
        if self.args.cal_var_loss:
            var_loss=0
        class_loss = 0
        class_loss_fp = 0
        for it, ((data, nouse, class_l), _) in enumerate(loader):
            data, nouse, class_l = data.to(self.device), nouse.to(self.device), class_l.to(self.device)

            out = self.model(x=data, gt=class_l, binary=True, flag=None)
            class_logit = out['class_logit']
            _, cls_pred = class_logit.max(dim=1)

            class_correct += torch.sum(cls_pred == class_l.data)
            if self.args.cal_var_loss:
                var_loss += out['var_loss'].item()
            class_loss += criterion(class_logit, class_l).data.item()

            out_fp = self.model(x=data, gt=class_l, binary=False, flag=None)
            class_logit_fp = out_fp['class_logit']
            class_loss_fp += criterion(class_logit_fp, class_l).data.item()
            _, cls_pred_fp = class_logit_fp.max(dim=1)
            class_correct_fp += torch.sum(cls_pred_fp == class_l.data)
        final_out = {"class_correct": class_correct, "class_correct_fp": class_correct_fp, 'class_loss': class_loss, 'class_loss_fp': class_loss_fp}
        if self.args.cal_var_loss:
            final_out.update({'var_loss': var_loss})

        return final_out


    def do_training(self):
        with torch.no_grad():
            for phase, loader in self.test_loaders.items():
                final_out = self.do_test(loader)
                total = len(loader.dataset)
                class_correct = final_out["class_correct"]
                class_acc = float(class_correct) / total
                class_correct_fp = final_out["class_correct_fp"]
                class_acc_fp = float(class_correct_fp) / total
                log_loss = {"class": class_acc, "class_fp": class_acc_fp}
                print(log_loss)
        self.results = {"val": torch.zeros(self.args.epochs+ + self.args.warming_epoch_total),
                        "test": torch.zeros(self.args.epochs+ + self.args.warming_epoch_total)}
        self.results_fp = {"val": torch.zeros(self.args.epochs+ + self.args.warming_epoch_total),
                        "test": torch.zeros(self.args.epochs+ + self.args.warming_epoch_total)}
        self.results_var_loss = {"val": torch.zeros(self.args.epochs+ + self.args.warming_epoch_total),
                        "test": torch.zeros(self.args.epochs+ + self.args.warming_epoch_total)}
        self.results_class_loss = {"val": torch.zeros(self.args.epochs+ + self.args.warming_epoch_total),
                        "test": torch.zeros(self.args.epochs+ + self.args.warming_epoch_total)}
        self.results_class_loss_fp = {"val": torch.zeros(self.args.epochs+ + self.args.warming_epoch_total),
                                   "test": torch.zeros(self.args.epochs+ + self.args.warming_epoch_total)}
        # insert loading
        for self.current_epoch in range(self.start_epoch, self.args.epochs+ self.args.warming_epoch_total):
            cur_lr=[]
            for param_group in self.optimizer.param_groups:
                cur_lr.append(param_group['lr'])
            self.logger.new_epoch(cur_lr)
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

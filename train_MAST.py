import argparse
import os
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import lib.model as model
import lib.transform as tran
from lib.MAST import (ExplicitInterClassGraphLoss, bins_deltas_to_ts_batch,
                      grid_xyz, t_to_bin_delta_batch, xentropy)
from lib.read_data import ImageList_r as ImageList


torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if not os.path.exists('checkpoints'):
    os.mkdir('checkpoints')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
ctr = grid_xyz()


def make_dataset(args):
    data_transforms = {
        'train': tran.rr_train(resize_size=224),
        'val': tran.rr_train(resize_size=224),
        'test': tran.rr_eval(resize_size=224),
    }
    # set dataset
    batch_size = {"train": args.train_bs, "val": args.train_bs, "test": args.test_bs}
    c="data/image_list/color_train.txt"
    n="data/image_list/noisy_train.txt"
    s="data/image_list/scream_train.txt"

    c_t="data/image_list/color_test.txt"
    n_t="data/image_list/noisy_test.txt"
    s_t="data/image_list/scream_test.txt"

    if args.src =='c':
        source_path = c
    elif args.src =='n':
        source_path = n
    elif args.src =='s':
        source_path = s

    if args.tgt =='c':
        target_path = c
    elif args.tgt =='n':
        target_path = n
    elif args.tgt =='s':
        target_path = s

    if args.tgt =='c':
        target_path_t = c_t
    elif args.tgt =='n':
        target_path_t = n_t
    elif args.tgt =='s':
        target_path_t = s_t

    dsets = {"train": ImageList(open(source_path).readlines(), transform=data_transforms["train"]),
            "val": ImageList(open(target_path).readlines(),transform=data_transforms["val"]),
            "test": ImageList(open(target_path_t).readlines(),transform=data_transforms["test"])}
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size[x],
                                                shuffle=True, num_workers=16)
                    for x in ['train', 'val']}
    dset_loaders["test"] = torch.utils.data.DataLoader(dsets["test"], batch_size=batch_size["test"],
                                                    shuffle=False, num_workers=16)
    return batch_size, dset_loaders


def Regression_test(args, loader, model, optimizer=None, save=False, iter_num=None):
    model.eval()
    MSE = [0, 0, 0, 0]
    MAE = [0, 0, 0, 0]
    number = 0
    with torch.no_grad():
        for (imgs, labels) in loader:
            imgs = imgs.to(device)
            labels_source = labels.to(device)
            labels1 = labels_source[:, 2]
            labels3 = labels_source[:, 4]
            labels4 = labels_source[:, 5]
            labels1 = labels1.unsqueeze(1)
            labels3 = labels3.unsqueeze(1)
            labels4 = labels4.unsqueeze(1)
            # labels_source = torch.cat((labels1, labels3, labels4), dim=1)
            # labels = labels_source.float()
            labels = labels1.float()
            bins, deltas, f = model(imgs)
            pred = bins_deltas_to_ts_batch(bins, deltas, ctr)
            # MSE[0] += torch.nn.MSELoss(reduction='sum')(pred[:, 0], labels[:, 0])
            # MAE[0] += torch.nn.L1Loss(reduction='sum')(pred[:, 0], labels[:, 0])
            # MSE[1] += torch.nn.MSELoss(reduction='sum')(pred[:, 1], labels[:, 1])
            # MAE[1] += torch.nn.L1Loss(reduction='sum')(pred[:, 1], labels[:, 1])
            # MSE[2] += torch.nn.MSELoss(reduction='sum')(pred[:, 2], labels[:, 2])
            # MAE[2] += torch.nn.L1Loss(reduction='sum')(pred[:, 2], labels[:, 2])
            # MSE[3] += torch.nn.MSELoss(reduction='sum')(pred, labels)
            MAE[3] += torch.nn.L1Loss(reduction='sum')(pred, labels)
            number += imgs.size(0)
    for j in range(4):
        MSE[j] = MSE[j] / number
        MAE[j] = MAE[j] / number
    # print(f"\tMSE: {MSE[0]},{MSE[1]},{MSE[2]}")
    # print(f"\tMAE: {MAE[0]},{MAE[1]},{MAE[2]}")
    # print(f"\tMSEall : {MSE[3]}")
    print(f"\tMAEall : {MAE[3]}")
    if save:
        torch.save({'model':model.state_dict(), 'optim': optimizer.state_dict()},
         f'checkpoints/{args.src}->{args.tgt}-it_{iter_num}-MAE_{MAE[3]:.3f}.pth')
        print(f'checkpoints/{args.src}->{args.tgt}-it_{iter_num}-MAE_{MAE[3]:.3f}.pth')


def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma, power, init_lr=0.001, weight_decay=0.0005):
    lr = init_lr * (1 + gamma * iter_num) ** (-power)
    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        param_group['weight_decay'] = weight_decay * 2
        i += 1
    return optimizer


class Model_Regression(nn.Module):
    def __init__(self):
        super(Model_Regression,self).__init__()
        self.model_fc = model.Resnet18Fc()
        self.cls_layer = nn.Linear(512, 6)
        self.reg_layer = nn.Linear(512, 6)
        self.sigmoid = nn.Sigmoid()


    def forward(self,x):
        feature = self.model_fc(x)
        cls= self.cls_layer(feature)
        reg = self.reg_layer(feature)
        reg = self.sigmoid(reg)
        return cls, reg, feature


def pretrain_on_src(args, model):
    criterion = {"cls": xentropy, "reg": nn.MSELoss(), "icg": ExplicitInterClassGraphLoss()}
    optimizer_dict = [{"params": filter(lambda p: p.requires_grad, model.model_fc.parameters()), "lr": 0.1},
                    {"params": filter(lambda p: p.requires_grad, model.cls_layer.parameters()), "lr": 0.01},
                    {"params": filter(lambda p: p.requires_grad, model.reg_layer.parameters()), "lr": 0.01}]
    optimizer = optim.SGD(optimizer_dict, lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)

    train_cross_loss = train_mse_loss = train_icg_loss = train_total_loss = 0.0

    len_source = len(dset_loaders["train"]) - 1
    iter_source = iter(dset_loaders["train"])

    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])

    test_interval = 10
    num_iter = 20002
    print(args)
    for iter_num in range(1, num_iter + 1):
        model.train()
        optimizer = inv_lr_scheduler(param_lr, optimizer, iter_num, init_lr=args.lr, gamma=args.gamma, power=0.75,
                                    weight_decay=0.0005)
        optimizer.zero_grad()

        # initialize a new one after enumerated the whole datasets
        if iter_num % len_source == 0:
            iter_source = iter(dset_loaders["train"])
        data_source = iter_source.next()
        inputs_source, labels_source = data_source
        
        # label of dSprites: ('none', 'shape', 'scale', 'orientation', 'position x', 'position y')
        labels1 = labels_source[:, 2]
        # labels3 = labels_source[:, 4]
        # labels4 = labels_source[:, 5]

        labels1 = labels1.unsqueeze(1)
        # labels3 = labels3.unsqueeze(1)
        # labels4 = labels4.unsqueeze(1)

        # labels_source = torch.cat((labels1,labels3,labels4),dim=1)
        labels_source = labels1
        labels_source = labels_source.float().to(device)

        inputs = inputs_source
        inputs = inputs.to(device)

        bins, deltas = t_to_bin_delta_batch(labels_source, ctr)
        inputs_s = inputs.narrow(0, 0, batch_size["train"])
        # inputs_t = inputs.narrow(0, batch_size["train"], batch_size["train"])
        cls, reg, f = model(inputs_s)
        
        classifier_loss = criterion["cls"](cls, bins) * args.cls_weight
        regressor_loss = criterion['reg'](reg, deltas) * args.reg_weight
        icg_loss = criterion['icg'](torch.max(bins, dim=-1)[1], f) * args.icg_weight
        total_loss = classifier_loss + regressor_loss + icg_loss
        total_loss.backward()
        optimizer.step()
        train_cross_loss += classifier_loss.item()
        train_mse_loss += regressor_loss.item()
        train_icg_loss += icg_loss.item()
        train_total_loss += total_loss.item()
        if iter_num % test_interval == 0:
            print((f"Iter {iter_num:05d}, "
                  f"Average Cross Entropy Loss: {train_cross_loss / float(test_interval):.4f}; "
                  f"Average MSE Loss: {train_mse_loss / float(test_interval):.4f}; "
                  f"Average ICG Loss: {train_icg_loss / float(test_interval):.4f}; "
                  f"Average Training Loss: {train_total_loss / float(test_interval):.4f}"))
            train_cross_loss = train_mse_loss = train_icg_loss = train_total_loss = 0.0
        if (iter_num % test_interval) == 0:
            Regression_test(args, dset_loaders['test'], model, optimizer=optimizer, save=True, iter_num=iter_num)


def collect_samples_with_pseudo_label(model, threshold, sample_iter):
    '''Pseudo label generation and selection on target dataset'''
    global iter_target
    model.eval()
    img_lbl = deque()
    with torch.no_grad():
        for i in range(sample_iter):
            print(f'iter {i}: collected {len(img_lbl)} samples', end='\r', flush=True)
            try:
                imgs, labels = iter_target.next()
            except StopIteration:
                iter_target = iter(dset_loaders["val"])
                imgs, labels = iter_target.next()
            labels_source = labels.to(device)
            labels1 = labels_source[:, 2]
            labels3 = labels_source[:, 4]
            labels4 = labels_source[:, 5]

            labels1 = labels1.unsqueeze(1)
            labels3 = labels3.unsqueeze(1)
            labels4 = labels4.unsqueeze(1)

            # labels_source = torch.cat((labels1, labels3, labels4), dim=1)
            labels_source = labels1
            labels = labels_source.float()

            imgs = imgs.to(device)
            cls, reg, f = model(imgs)
            sel = F.softmax(cls, dim=-1).max(-1)[0] > threshold
            if sum(sel) == 0: continue
            preds = bins_deltas_to_ts_batch(cls, reg, ctr)
            img_lbl.extend(list(zip(imgs[sel].cpu(), preds[sel].cpu())))
    print(f'---------finished, collected {len(img_lbl)} samples with {i+1} iters---------')
    return img_lbl


def selftrain_t(args, model, sample_selected):
    global iter_source
    criterion = {"cls": xentropy, "reg": nn.MSELoss(), "icg": ExplicitInterClassGraphLoss()}
    optimizer_dict = [{"params": filter(lambda p: p.requires_grad, model.model_fc.parameters()), "lr": 0.03},
                    {"params": filter(lambda p: p.requires_grad, model.cls_layer.parameters()), "lr": 0.01},
                    {"params": filter(lambda p: p.requires_grad, model.reg_layer.parameters()), "lr": 0.01}]
    optimizer = optim.SGD(optimizer_dict, lr=0.03, momentum=0.9, weight_decay=0.0005, nesterov=True)
    train_cross_loss = train_mse_loss = train_icg_loss = train_total_loss = 0.0
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    test_interval = 10

    for iter_num, (img, labelt) in enumerate(sample_selected):
        model.train()
        optimizer = inv_lr_scheduler(param_lr, optimizer, iter_num, init_lr=args.lr, gamma=args.gamma, power=0.75,
                                    weight_decay=0.0005)
        optimizer.zero_grad()
        # pass src data
        try:
            data_source = iter_source.next()
        except StopIteration:
            iter_source = iter(dset_loaders["train"])
            data_source = iter_source.next()
        inputs_source, labels_source = data_source
        labels_source = labels_source[:, 2].float()
        cls_s, reg_s, fs = model(inputs_source.cuda())
        bins_s, deltas_s = t_to_bin_delta_batch(labels_source.cuda(), ctr)
        classifier_loss_s = criterion["cls"](cls_s, bins_s) * args.cls_weight
        regressor_loss_s = criterion['reg'](reg_s, deltas_s) * args.reg_weight
        icg_loss_s = criterion['icg'](torch.max(bins_s, dim=-1)[1], fs) * args.icg_weight
        total_loss_s = classifier_loss_s + regressor_loss_s + icg_loss_s
        total_loss_s *= args.src_loss_weight
        total_loss_s.backward()

        # pass tgt data
        cls, reg, f = model(img.cuda())
        bins, deltas = t_to_bin_delta_batch(labelt.cuda(), ctr)
        classifier_loss = criterion["cls"](cls, bins) * args.cls_weight
        regressor_loss = criterion['reg'](reg, deltas) * args.reg_weight
        icg_loss = criterion['icg'](torch.max(bins, dim=-1)[1], f) * args.icg_weight
        total_loss = classifier_loss + regressor_loss + icg_loss
        total_loss.backward()
        optimizer.step()
        train_cross_loss += classifier_loss.item()
        train_mse_loss += regressor_loss.item()
        train_icg_loss += icg_loss.item()
        train_total_loss += total_loss.item()
        if iter_num % test_interval == 0:
            print((f"Iter {iter_num:05d}, "
                  f"Average Cross Entropy Loss: {train_cross_loss / float(test_interval):.4f}; "
                  f"Average MSE Loss: {train_mse_loss / float(test_interval):.4f}; "
                  f"Average ICG Loss: {train_icg_loss / float(test_interval):.4f}; "
                  f"Average Training Loss: {train_total_loss / float(test_interval):.4f}"))
            train_cross_loss = train_mse_loss = train_icg_loss = train_total_loss = 0.0
            Regression_test(args, dset_loaders['test'], model, optimizer, save=True)


def ema(teacher_ema_model, student_model, global_step=1e5, a=0.96):
    a = min(1 - 1 / (global_step + 1), a)
    for ema_param, param in zip(teacher_ema_model.parameters(), student_model.parameters()):
        ema_param.data.mul_(a).add_(param.data, alpha=1 - a)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch DAregre experiment')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--src', type=str, default='c', metavar='S', help='source dataset')
    parser.add_argument('--tgt', type=str, default='n', metavar='S', help='target dataset')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.train_bs = 36
    args.test_bs = 128
    args.threshold = 0.95  # 0.8、0.47
    args.threshold_decay = 0.01
    args.src_loss_weight = 0.01
    args.cls_weight = 1.0
    args.reg_weight = 1.0
    args.icg_weight = 1.0
    args.lr = 0.03  # init learning rate for fine-tune
    args.gamma = 0.0001  # learning rate decay
    args.sample_iter = 500
    batch_size, dset_loaders = make_dataset(args)
    Model_R = Model_Regression().to(device)
    Model_da = Model_Regression().to(device)



    # pretrain_on_src(args, Model_R)
    Model_R.load_state_dict(torch.load('checkpoints/c->s-it_None-MAE_0.293.pth')['model'])
    iter_source = iter(dset_loaders["train"])
    iter_target = iter(dset_loaders["val"])
    for _ in range(15):
        args.threshold -= args.threshold_decay
        img_selected = collect_samples_with_pseudo_label(Model_R, args.threshold, sample_iter=args.sample_iter)
        img_selected = DataLoader(img_selected, batch_size=args.train_bs, shuffle=True, num_workers=16)
        selftrain_t(args, Model_da, img_selected)
        ema(Model_R, Model_da)

import numpy as np
import torch
import torch.nn as nn
import math
import random
import argparse
import shutil
import struct as st
from branch_util import *


parser = argparse.ArgumentParser(description='Pytorch Branch Finetuning')
parser.add_argument('-ilf', '--image-list-file', default='../../data/stitching/sample_img_list.txt',
        type=str, metavar='N', help='image list file')
parser.add_argument('-ff', '--feat-file', default='../../data/stitching/ext_feat.bin',
        type=str, metavar='N', help='extracted feature')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-f', '--feat-len', default=256, type=int, metavar='F', help='feature length (default: 256)')
parser.add_argument('--iters', default=10000, type=int, metavar='N', help='number of total iters to run')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')

def main():
    args = parser.parse_args()
    # Load data
    img_list_file = args.image_list_file
    feat_file = args.feat_file

    train_map = get_dict(img_list_file)


    data = load_feat(feat_file)
    data = np.vstack(data)
    model = Branch(feat_dim=256)
    model.cuda()
    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    model.train()
    losses = AverageMeter()
    for iter in range(args.iters):
        batch_train_feat, batch_target_feat, batch_norm_angle = gen_batch(train_map, data, args.batch_size, args.feat_len)
        batch_train_feat = torch.autograd.Variable(torch.from_numpy(batch_train_feat.astype(np.float32))).cuda()
        batch_norm_angle = torch.autograd.Variable(torch.from_numpy(batch_norm_angle.astype(np.float32))).cuda()
        batch_target_feat = torch.autograd.Variable(torch.from_numpy(batch_target_feat.astype(np.float32))).cuda()

        output = model(batch_train_feat, batch_norm_angle)
        loss = criterion(output, batch_target_feat)
        losses.update(loss.data[0], loss.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter % args.print_freq == 0:
            print('Iter: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   iter, args.iters, loss=losses))

    torch.save({'state_dict': model.state_dict()}, 'checkpoint.pth')


def get_dict(file_name):
    res_dict = dict()
    with open(file_name) as f:
        for idx, line in enumerate(f):
            record = line.strip().split()
            img_name, yaw, img_id = record[0],float(record[1]), idx
            celeb_name = img_name.split('/')[1]
            if celeb_name not in res_dict:
                res_dict[celeb_name] = (list(), list())
            yaw = abs(yaw)
            if yaw <= 20:
                res_dict[celeb_name][0].append((img_name, yaw, img_id))
            elif yaw >= 50:
                res_dict[celeb_name][1].append((img_name, yaw, img_id))
    pop_list = []
    for key in res_dict:
        if len(res_dict[key][0]) == 0 or len(res_dict[key][1]) == 0:
            pop_list.append(key)
    for key in pop_list:
        res_dict.pop(key)
    return res_dict

def gen_batch(train_map, data, batch_size, feat_len):
    batch_train_feat = np.zeros([batch_size, feat_len])
    batch_target_feat = np.zeros([batch_size, feat_len])
    batch_yaw = np.zeros([batch_size, 1])
    keys = train_map.keys()
    for i in range(batch_size):
        this_key = random.sample(keys, 1)[0]
        frontal_set = train_map[this_key][0]
        profile_set = train_map[this_key][1]
        frontals_index = [a[2] for a in frontal_set]
        frontals_feat = data[frontals_index, ...]
        profile_selec = random.sample(profile_set, 1)
        batch_train_feat[i, ...] = data[profile_selec[0][2], ...]
        batch_target_feat[i, ...] =  np.mean(frontals_feat, axis = 0)
        batch_yaw[i, ...] = norm_angle(profile_selec[0][1])
    return batch_train_feat, batch_target_feat, batch_yaw


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def load_feat(feat_file):
    feats = list()
    with open(feat_file, 'rb') as in_f:
        feat_num, feat_dim = st.unpack('ii', in_f.read(8))
        for i in range(feat_num):
            feat = np.array(st.unpack('f'*feat_dim, in_f.read(4*feat_dim)))
            feats.append(feat)
    return feats

if __name__ == '__main__':
    main()

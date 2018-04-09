import numpy as np
import struct as st
import torch
import torch.nn as nn
import math
from branch_util import *

def load_feat(feat_file):
    feats = list()
    with open(feat_file, 'rb') as in_f:
        feat_num, feat_dim = st.unpack('ii', in_f.read(8))
        for i in range(feat_num):
            feat = np.array(st.unpack('f'*feat_dim, in_f.read(4*feat_dim)))
            feats.append(feat)
    return feats

if __name__ == '__main__':
    frontal_feat_file = '/home/sensetime/Downloads/ext_feat/data/frontal_feat_naive.bin'
    profile_feat_file = '/home/sensetime/Downloads/ext_feat/data/profile_feat_naive.bin'

    frontal_feats = np.vstack(load_feat(frontal_feat_file))
    profile_feats = np.vstack(load_feat(profile_feat_file))

    frontal_angles = []
    profile_angles = []
    with open('../../data/pose_output.txt', 'r') as fin:
        tmp_count = 0
        for line in fin:
            line_split = line.strip().split()
            if tmp_count % 14 < 10:
                frontal_angles.append(float(line_split[2]))
            else:
                profile_angles.append(float(line_split[2]))
            tmp_count = tmp_count + 1
    frontal_angles = np.vstack(frontal_angles)
    profile_angles = np.vstack(profile_angles)

    model = Branch(feat_dim=256)
    model.cuda()
    checkpoint = torch.load('checkpoint.pth')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    frontal_feat_file = './frontal_feat.bin'
    feat_dim = 256
    data_num = frontal_feats.shape[0]
    with open(frontal_feat_file, 'wb') as bin_f:
        bin_f.write(st.pack('ii', data_num, feat_dim))
        for i in range(data_num):
            feat = frontal_feats[i, :].reshape([1, -1])
            yaw = np.zeros([1, 1])
            yaw[0,0] = norm_angle(frontal_angles[i, :])
            feat = torch.autograd.Variable(torch.from_numpy(feat.astype(np.float32)), volatile=True).cuda()
            yaw = torch.autograd.Variable(torch.from_numpy(yaw.astype(np.float32)), volatile=True).cuda()
            output = model(feat, yaw)
            output_data = output.cpu().data.numpy()
            feat_num  = output.size(0)
            for j in range(feat_num):
                bin_f.write(st.pack('f'*feat_dim, *tuple(output_data[j,:])))

    data_num = profile_feats.shape[0]
    profile_feat_file = './profile_feat.bin'
    with open(profile_feat_file, 'wb') as bin_f:
        bin_f.write(st.pack('ii', data_num, feat_dim))
        for i in range(data_num):
            feat = profile_feats[i, :].reshape([1, -1])
            yaw = np.zeros([1, 1])
            yaw[0,0] = norm_angle(profile_angles[i, :])
            feat = torch.autograd.Variable(torch.from_numpy(feat.astype(np.float32)), volatile=True).cuda()
            yaw = torch.autograd.Variable(torch.from_numpy(yaw.astype(np.float32)), volatile=True).cuda()
            output = model(feat, yaw)
            output_data = output.cpu().data.numpy()
            feat_num  = output.size(0)
            for j in range(feat_num):
                bin_f.write(st.pack('f'*feat_dim, *tuple(output_data[j,:])))

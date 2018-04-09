import os, sys, shutil
import struct as st
import numpy as np
import bisect
import pickle
from scipy import spatial
from sklearn import metrics 
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from sklearn.metrics.pairwise import cosine_similarity

def load_meta_data(meta_file):
    meta_data = dict()
    with open(meta_file, 'r') as in_f:
        in_f.readline() # the first line is not data
        for line in in_f:
            record = line.strip().split(',')
            template, class_id, img_path = int(record[0]), int(record[1]), record[2]
            if template not in meta_data:
                meta_data[template] = ( class_id, [img_path,] )
            else:
                meta_data[template][1].append(img_path)
    return meta_data

def load_feat(list_file, bin_file):
    mid_feats = dict()
    with open(list_file, 'r') as list_f, open(bin_file, 'rb') as bin_f:
        (data_num, feat_dim) = st.unpack('ii', bin_f.read(8))
        for line in list_f:
            record = line.strip().split('/')
            img_name = '/'.join(record[-2:])
            feat = np.array(st.unpack('f'*feat_dim, bin_f.read(4*feat_dim)))
            mid_feats[img_name] = feat
    return mid_feats, feat_dim

def update_meta_data(meta_data, feats, feat_dim):
    new_meta_data = dict()
    for template in meta_data.keys():
        class_id, img_names = meta_data[template]
        feat = np.zeros(feat_dim)
        feat_num = 0
        for img_name in img_names:
            if img_name in feats:
                feat += feats[img_name]
                feat_num += 1
        if feat_num > 0: feat /= feat_num
        if feat_num > 0: new_meta_data[template] = (class_id, feat)
    return new_meta_data

# calc_tar2{{{
def calc_tar2(fpr, tpr, method=0):
    if method == 0:
        min_dis, tar2 = 100.0, 1.0
        for i in range(fpr.size):
            if(fpr[i]+tpr[i] > 1.0):
                break
            mid_res = abs(fpr[i]+tpr[i]-1.0)
            if(mid_res < min_dis):
                min_dis = mid_res
                tar2 = fpr[i]
        return tar2
    else:
        f = lambda x: np.interp(x, fpr, tpr)+x-1
        return fsolve(f, 0.0)
#}}}

def eval_roc(pair_file, meta_data, split):
    labels, scores = [], []
    with open(pair_file, 'r') as in_f:
        for line in in_f:
            record = line.strip().split(',')
            temp1, temp2 = int(record[0]),int(record[1])
            if not(temp1 in meta_data and temp2 in meta_data):
                continue
            info1, info2 = meta_data[temp1], meta_data[temp2]
            label = int(info1[0] == info2[0])
            score = 1-spatial.distance.cosine(info1[1],info2[1])
            labels.append(label)
            scores.append(score)
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
    idx1 = bisect.bisect_left(fpr,0.01)
    tar1 = tpr[ bisect.bisect_left(fpr,0.001) ]
    tar2 = tpr[ bisect.bisect_left(fpr,0.01)  ]
    return tar1, tar2


def test_verify(model_type):
    
    #model_type = 'resnet18'
    protocol_dir = '../../data/IJBA/protocol_11'
    align_img_dir = '../../data/IJBA/align_image_11'
    tar1s, tar2s = [],[]
    split_num = 10
    for split in range(1, split_num+1):

        # load meta data first
        split_protocol_dir = os.path.join(protocol_dir, 'split'+str(split))
        meta_file = os.path.join(split_protocol_dir, 'verify_metadata_{}.csv'.format(split))
        meta_data = load_meta_data(meta_file)

        # load extract feat
        feats = dict()
        feat_dim = 0
        split_img_dir = os.path.join(align_img_dir, 'ijb_a_11_align_split{}'.format(split))
        for img_type in ['frame', 'img']:
            list_file = os.path.join(split_img_dir, '{}_list.txt'.format(img_type))
            bin_file = os.path.join(split_img_dir, '{}_{}_feat.bin'.format(model_type,img_type))
            mid_feats, feat_dim = load_feat(list_file, bin_file)
            feats.update(mid_feats)

        # update meta data
        meta_data = update_meta_data(meta_data, feats, feat_dim)

        # eval roc
        pair_file = os.path.join(split_protocol_dir, 'verify_comparisons_{}.csv'.format(split))
        tar1, tar2 = eval_roc(pair_file, meta_data, split)
        tar1s.append(tar1)
        tar2s.append(tar2)
        print('split {}, tar1: {}, tar2: {}'.format(split,tar1,tar2))
    print('tar1: {} +/- {}'.format(np.mean(tar1s), np.std(tar1s)))
    print('tar2: {} +/- {}'.format(np.mean(tar2s), np.std(tar2s)))
    
    return np.mean(tar1s), np.std(tar1s), np.mean(tar2s), np.std(tar2s)

if __name__ == '__main__':
    test_verify(model_type='resnet18')

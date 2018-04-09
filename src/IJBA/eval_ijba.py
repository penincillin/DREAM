import argparse
import os,sys,shutil
import time
import struct as st

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
#import transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

from selfDefine import CFPDataset, CaffeCrop

from ResNet import resnet18, resnet50, resnet101

from test_recog import test_recog
from test_verify import test_verify


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch CelebA Training')
parser.add_argument('--img_dir', metavar='DIR', default='', help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18', choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: alexnet)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='./data/resnet18.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--model_dir','-m', default='./model', type=str)



def extract_feat(arch, model_path, yaw_type):
    global args, best_prec1
    args = parser.parse_args()

    if arch.find('end2end')>=0:
        end2end=True
    else:
        end2end=False

    arch = arch.split('_')[0]
  
    class_num = 87020
    #class_num = 13386
    
    model = None
    assert(arch in ['resnet18','resnet50','resnet101'])
    if arch == 'resnet18':
        model = resnet18(pretrained=False, num_classes=class_num, \
                extract_feature=True, end2end=end2end)
    if arch == 'resnet50':
        model = resnet50(pretrained=False, num_classes=class_num, \
                extract_feature=True, end2end=end2end)
    if arch == 'resnet101':
        model = resnet101(pretrained=False, num_classes=class_num, \
                extract_feature=True, end2end=end2end)

    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    
    assert(os.path.isfile(model_path))
    checkpoint = torch.load(model_path)
    pretrained_state_dict = checkpoint['state_dict']
    model_state_dict = model.state_dict()
    for key in pretrained_state_dict:
        if key in model_state_dict:
            model_state_dict[key] = pretrained_state_dict[key]
    model.load_state_dict(model_state_dict)

    print('load trained model complete')

    caffe_crop = CaffeCrop('test')

    infos = [ ('../../data/IJBA/align_image_11', 'ijb_a_11_align_split', 'frame'),
               ('../../data/IJBA/align_image_11', 'ijb_a_11_align_split', 'img'), 
               ('../../data/IJBA/align_image_1N', 'split', 'gallery'),
               ('../../data/IJBA/align_image_1N', 'split', 'probe') ] 

    
    for root_dir, sub_dir, img_type in infos:

        for split in range(1,11):
            split_dir = os.path.join(root_dir, sub_dir+str(split))
            img_dir = os.path.join(split_dir, img_type)
            img_list_file = os.path.join(split_dir, '{}_list_{}.txt'.format(img_type, yaw_type))

            img_dataset =  CFPDataset(args.img_dir, img_list_file,
                    transforms.Compose([caffe_crop,transforms.ToTensor()]))
            img_loader = torch.utils.data.DataLoader(
                    img_dataset,
                    batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True)
       
        
            data_num = len(img_dataset)
            img_feat_file = os.path.join(split_dir, '{}_{}_feat.bin'.format(arch, img_type))
            feat_dim = 256
            with open(img_feat_file, 'wb') as bin_f:
                bin_f.write(st.pack('ii', data_num, feat_dim))
                for i, (input, yaw) in enumerate(img_loader):
                    yaw = yaw.float().cuda(async=True)
                    yaw_var = torch.autograd.Variable(yaw)
                    input_var = torch.autograd.Variable(input, volatile=True)
                    output = model(input_var, yaw_var)
                    output_data = output.cpu().data.numpy()
                    feat_num  = output.size(0)
                    
                    for j in range(feat_num):
                        bin_f.write(st.pack('f'*feat_dim, *tuple(output_data[j,:])))
        
            print('we have complete {} {}'.format(img_type, split))


if __name__ == '__main__':

    infos = [ ['resnet18_naive', '../../data/model/ijba_res18_naive.pth.tar', 'nonli'], 
              ['resnet18_end2end', '../../data/model/ijba_res18_end2end.pth.tar', 'nonli'], ]


    for arch, model_path, yaw_type in infos:
        print(model_path.split('/')[-1])
        extract_feat(arch, model_path, yaw_type)
        arch = arch.split('_')[0]
        test_recog(arch)
        test_verify(arch)

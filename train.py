#!/usr/bin/env python
import argparse
from face_swap import *


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()


parser.add_argument("cmd", help=argparse.SUPPRESS, nargs="*")
parser.add_argument('--dataset_a', nargs='?', default='faceA', type=str)
parser.add_argument('--dataset_b', nargs='?', default='faceB', type=str)
parser.add_argument('--enc_att', nargs='?', default=True, type=bool)
parser.add_argument('--dec_att', nargs='?', default=True, type=bool)
parser.add_argument('--disc_att', nargs='?', default=False, type=bool)
parser.add_argument('--batch_size', nargs='?', default=10, type=int)
parser.add_argument('--workers', nargs='?', default=16, type=int)
parser.add_argument('--res', nargs='?', default=128, type=int)
parser.add_argument('--res_layers_p', type=int, nargs='+', default=[2,6,9,14,17,20,23,26])
parser.add_argument('--res_layers_p_weight', type=int, nargs='+', default=[1,1,1,1,1,1,1,1])
parser.add_argument('--recon_weight', nargs='?', default=3., type=float)
parser.add_argument('--edge_weight', nargs='?', default=.1, type=float)
parser.add_argument('--beta1', nargs='?', default=.5, type=float)
parser.add_argument('--beta2', nargs='?', default=.999, type=float)
parser.add_argument('--lr', nargs='?', default=2e-4, type=float)
parser.add_argument('--lr_drop_every', nargs='?', default=40, type=int)
parser.add_argument('--lr_drop_start', nargs='?', default=0, type=int)
parser.add_argument('--save_every', nargs='?', default=10, type=int)
parser.add_argument('--save_img_every', nargs='?', default=1, type=int)
parser.add_argument('--train_epoch', nargs='?', default=200, type=int)
parser.add_argument('--save_root', nargs='?', default='faceswap', type=str)
parser.add_argument('--load_state', nargs='?', type=str)


params = vars(parser.parse_args())
# if load_state arg is not used, then train model from scratch
if __name__ == '__main__':
    sr = FaceSwap(params)
    if params['load_state']:
            sr.load_state(params['load_state'])
    else:
        print('Starting From Scratch')

    sr.train()

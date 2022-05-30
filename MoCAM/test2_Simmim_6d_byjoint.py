import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_proc.emotionmocap_dataset2 import EmotionDataset
from data_proc.utils import increment_path
# from model.model import TCN, PURE3D
import torch.nn.functional as F
from  pytorch3d import transforms

from model.MoT import MoT_6d
from model.SImMIM import SimMIM

def test(opt, device):

    save_dir = Path(os.path.join('runs', 'train', opt.exp_name))
    wdir = save_dir / 'weights'
    weights = os.listdir(wdir)

    if opt.weight == 'latest':
        weights_paths = [wdir / weight for weight in weights]
        weight_path = max(weights_paths , key = os.path.getctime)
    else:
        weight_path = wdir / ('train-' + opt.weight + '.pt')
    ckpt = torch.load(weight_path, map_location=device)
    print(f"Loaded weight: {weight_path}")


    # Load LAFAN Dataset
    Path(opt.processed_data_dir).mkdir(parents=True, exist_ok=True)
    emotion_dataset = EmotionDataset(data_dir=opt.data_path, processed_data_dir=opt.processed_data_dir, train=False, device=device, window=opt.window)
    emotion_data_loader = DataLoader(emotion_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)

    n_classes = 7
    seq_len=opt.window
    num_joints=35
    mot_dim = 1024
    mot_depth = 6
    mot_heads = 8 
    mot_mlp_dim = 2048
    mot_pool = 'cls'
    mot_channels =1, 
    mot_dim_head = 64
    n_hid = 70
    n_level = 4
    channel_sizes = [n_hid] * n_level
    kernel_size = 5
    
    model = MoT_6d(seq_len = seq_len, num_joints=num_joints, num_classes=n_classes,  dim=mot_dim, depth=mot_depth, heads=mot_heads, mlp_dim = mot_mlp_dim)
    model.load_state_dict(ckpt['MoT'])
    model.to(device)
    model.eval()
    correct = 0
    pbar = tqdm(emotion_data_loader, position=1, desc="Batch")
    selected_joint = [2,3,4,5,7,8,9,11,12,13,14,16,17,18,20,21,22,23,25,26,27,28,29,30,31,33,34,35,36,37,38,39,41,42,43]

    confusion_matrix = torch.zeros(7, 7)
    with torch.no_grad():
        for batch in pbar:
            data_6d = batch["data_6d"].to(device)
            labels = batch["labels"].to(device)            
            # local_R = local_R[:,:,:]
            # batch_size, seq_len, num_j, _  = local_R.shape
            # R_mat = local_R.reshape(-1,3,3)
            # data = transforms.matrix_to_rotation_6d(R_mat)
            # data = data.reshape(batch_size, seq_len, num_j, 6)
            print(data.shape)
            logits = model(data.to(device))
            logits = logits
            prediction = logits.data.max(1)[1]
            print(prediction)
            correct += prediction.eq(labels).sum()
            for t, p in zip(labels.view(-1), prediction.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
        print('Test set: Accuracy: {:.2f}%'.format(100. * correct / len(emotion_data_loader.dataset)))
        print(confusion_matrix)
        print(confusion_matrix.diag()/confusion_matrix.sum(1))

    # print(f"ID {test_idx[i]}: test completed.")

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='runs/train', help='project/name')
    parser.add_argument('--weight', default='500')
    parser.add_argument('--exp_name', default='SIM_MIM_MOT_6D_cls', help='experiment name')
    parser.add_argument('--data_path', type=str, default='/home/taehyun/workspace/childtoy/MotionReasoning/dataset/mocap_emotion_rig', help='BVH dataset path')
    parser.add_argument('--window', type=int, default=80, help='window')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--processed_data_dir', type=str, default='processed_data_mocam_80_All_Class_norm/', help='path to save pickled processed data')
    parser.add_argument('--save_path', type=str, default='runs/test', help='path to save model')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    device = torch.device("cpu")
    test(opt, device)

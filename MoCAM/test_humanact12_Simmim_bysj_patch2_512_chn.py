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

from data_proc.utils import increment_path
# from model.model import TCN, PURE3D
import torch.nn.functional as F
from motion.dataset.human36m import Human36mDataset, human36m_label_map
from motion.dataset.humanact12 import HumanAct12Dataset2, humanact12_label_map
from sklearn.preprocessing import LabelEncoder

from data_proc.utils import increment_path
import torch
from model.MoT import MoT_patch2_seg_chn_sj
from model.SImMIM import SimMIM_patch2_seg_chn_sj

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
    test_dataset = HumanAct12Dataset2(data_path="../dataset/experiment/HumanAct12Poses/humanact12poses.pkl", motion_length=150, dataset="test", transform=None)
    full_proc_label_list = list(humanact12_label_map.values())
    label_map = humanact12_label_map
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=128,
        num_workers=0,
        shuffle=False,
        pin_memory=True)
    le = LabelEncoder()
    le.fit(list(label_map.values()))
    # transform_labels = le.fit_transform(train_labels)
    # transform_labels = torch.Tensor(transform_labels).to(device)
    # tensor_dataset = TensorDataset(train_rotation_6d, train_valid_length_list, transform_labels)
    # train_data_loader = DataLoader(tensor_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)

    n_classes = 12
    seq_len=opt.window
    num_joints=22
    mot_dim = 256
    mot_depth = 6
    mot_heads = 12
    mot_mlp_dim = 512
    mot_pool = 'cls'
    mot_channels =1, 
    mot_dim_head = 64
    n_hid = 70
    n_level = 4
    channel_sizes = [n_hid] * n_level
    kernel_size = 5
    
    mot = MoT_patch2_seg_chn_sj(seq_len = seq_len, num_joints=num_joints, sub_seq=30, num_classes=n_classes,  dim=mot_dim, depth=mot_depth, heads=mot_heads, mlp_dim = mot_mlp_dim)
    # mim = SimMIM_patch2_seg( encoder = mot, masking_ratio = 0.5)
    mot.load_state_dict(ckpt['MoT'])
    mot.to(device)
    mot.eval()
    correct = 0

    pbar = tqdm(test_data_loader, position=1, desc="Batch")
    confusion_matrix = torch.zeros(12, 12)
    with torch.no_grad():
        for batch in pbar:
            motions = batch["rotation_6d_pose_list"].to(device)
            valid_length = batch['valid_length_list'].to(device)
            labels = batch["labels"]
            print(motions.shape)
            transform_labels = le.fit_transform(labels)
            transform_labels = torch.Tensor(transform_labels)
            transform_labels = transform_labels.long().to(device)
            B,T,J,C = motions.shape
            # motions = motions.view(B,T,J*C)
            # motions = motions.unsqueeze(1)
            motions = motions.permute(0,3,1,2)
            logits = mot(motions.to(device), valid_length.to(device))
            prediction = logits.data.max(1)[1]
            print(prediction)
            correct += prediction.eq(transform_labels).sum()
            for t, p in zip(transform_labels.view(-1), prediction.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
        print('Test set: Accuracy: {:.2f}%'.format(100. * correct / len(test_data_loader.dataset)))
        print(confusion_matrix)
        print(confusion_matrix.diag()/confusion_matrix.sum(1))

    # print(f"ID {test_idx[i]}: test completed.")

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='runs/train', help='project/name')
    parser.add_argument('--weight', default='150')
    parser.add_argument('--exp_name', default='humanact12_SIM_MIM_MOT_patch2_sj_noval_512_ssl_augthree_cls_noaug', help='experiment name')
    parser.add_argument('--data_path', type=str, default='/home/taehyun/workspace/childtoy/MotionReasoning/dataset/mocap_emotion_rig', help='BVH dataset path')
    parser.add_argument('--window', type=int, default=150, help='window')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--processed_data_dir', type=str, default='processed_data_mocam_80_All_Class_norm2/', help='path to save pickled processed data')
    parser.add_argument('--save_path', type=str, default='runs/test', help='path to save model')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    device = torch.device("cpu")
    test(opt, device)

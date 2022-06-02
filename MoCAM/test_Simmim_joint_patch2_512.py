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

from data_proc.emotionmocap_dataset import EmotionDataset
from data_proc.utils import increment_path
# from model.model import TCN, PURE3D
import torch.nn.functional as F
from model.MoT import MoT_patch2
from model.SImMIM import SimMIM_patch2

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
    mot_dim = 256
    mot_depth = 6
    mot_heads = 8 
    mot_mlp_dim = 512
    mot_pool = 'cls'
    mot_channels =1, 
    mot_dim_head = 64
    n_hid = 70
    n_level = 4
    channel_sizes = [n_hid] * n_level
    kernel_size = 5
    
    model = MoT_patch2(seq_len = seq_len, num_joints=num_joints, sub_seq=20, num_classes=n_classes,  dim=mot_dim, depth=mot_depth, heads=mot_heads, mlp_dim = mot_mlp_dim)
    # mim = SimMIM_patch( encoder = mot, masking_ratio = 0.5)
    model.load_state_dict(ckpt['MoT'])
    model.to(device)
    model.eval()
    correct = 0
    pbar = tqdm(emotion_data_loader, position=1, desc="Batch")
    confusion_matrix = torch.zeros(7, 7)
    with torch.no_grad():
        for batch in pbar:
            local_q = batch["local_q"].to(device)
            labels = batch["labels"].to(device)
            batch_size = local_q.shape[0]
            data = local_q
            data = data.unsqueeze(1)
            # data = torch.zeros(data.shape)
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
    parser.add_argument('--weight', default='50')
    parser.add_argument('--exp_name', default='SIM_MIM_MOT_patch2_512_cls8', help='experiment name')
    parser.add_argument('--data_path', type=str, default='/home/taehyun/workspace/childtoy/MotionReasoning/dataset/mocap_emotion_rig', help='BVH dataset path')
    parser.add_argument('--window', type=int, default=80, help='window')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--processed_data_dir', type=str, default='processed_data_mocam_80_All_Class_norm2/', help='path to save pickled processed data')
    parser.add_argument('--save_path', type=str, default='runs/test', help='path to save model')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    device = torch.device("cpu")
    test(opt, device)

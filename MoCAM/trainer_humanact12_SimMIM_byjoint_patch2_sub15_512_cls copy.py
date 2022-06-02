import argparse
import os
from pathlib import Path
import json
import pandas as pd

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from motion.dataset.human36m import Human36mDataset, human36m_label_map
from motion.dataset.humanact12 import HumanAct12Dataset, humanact12_label_map
from sklearn.preprocessing import LabelEncoder

from data_proc.utils import increment_path
import torch
from model.MoT import MoT_patch2_seg
from model.SImMIM import SimMIM_patch2_seg
def train(opt, device):
    # Prepare Directories
    save_dir = Path(opt.save_dir)
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)
    # Save run settings
    # with open(save_dir / 'opt.yaml', 'w') as f:
    #     yaml.safe_dump(vars(opt), f, sort_keys=True)
    epochs = opt.epochs
    save_interval = opt.save_interval
    ckpt = torch.load(opt.weight_path, map_location=device)
    # Loggers
    #init wandb
    wandb.init(config=opt, project=opt.wandb_pj_name, entity=opt.entity, name=opt.exp_name, dir=opt.save_dir)
    # Load EmotionMoCap Dataset
    train_dataset = HumanAct12Dataset(data_path="../dataset/experiment/HumanAct12Poses/humanact12poses.pkl", motion_length=150, dataset="train")
    full_proc_label_list = list(humanact12_label_map.values())
    label_map = humanact12_label_map
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=opt.batch_size,
        num_workers=0,
        shuffle=True,
        pin_memory=True)
    # train_rotation_6d = train_dataset.data['rotation_6d_pose_list']
    # train_valid_length_list = train_dataset.data['valid_length_list']
    # train_labels = train_dataset.data['labels']
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
    mot_heads = 8
    mot_mlp_dim = 512
    mot_pool = 'cls'
    mot_channels =1, 
    mot_dim_head = 64
    epochs = opt.epochs
    n_hid = 70
    n_level = 4
    channel_sizes = [n_hid] * n_level
    kernel_size = 5
    
    mot = MoT_patch2_seg(seq_len = seq_len, num_joints=num_joints, sub_seq=15, num_classes=n_classes,  dim=mot_dim, depth=mot_depth, heads=mot_heads, mlp_dim = mot_mlp_dim)
    mim = SimMIM_patch2_seg( encoder = mot, masking_ratio = 0.5)
    mot.load_state_dict(ckpt['MoT'])
    mot.to(device)
    # mim.to(device)
    mot.train()

    optim = AdamW(params=mot.parameters(), lr=opt.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=100, gamma=0.9)
    loss_fct = nn.CrossEntropyLoss()
    

    for epoch in range(1, epochs + 1):

        pbar = tqdm(train_data_loader, position=1, desc="Batch")
        for batch in pbar:
            motions = batch["rotation_6d_pose_list"].to(device)
            valid_length = batch['valid_length_list'].to(device)
            labels = batch["labels"]
            transform_labels = le.fit_transform(labels)
            transform_labels = torch.Tensor(transform_labels)
            transform_labels = transform_labels.long().to(device)
            B,T,J,C = motions.shape
            motions = motions.view(B,T,J*C)
            motions = motions.unsqueeze(1)
            # motions = motions.permute(0,1,3,2)
            optim.zero_grad()      
            logits = mot(motions.to(device), valid_length.to(device))
            loss = loss_fct(logits.view(-1, n_classes), transform_labels.view(-1))     
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(mot.parameters(), 2.0)

            optim.step()

        # Log
        log_dict = {
            "Train/Loss": loss, 
        }
        wandb.log(log_dict)

        # Save model
        if (epoch % save_interval) == 0:
            ckpt = {'epoch': epoch,
                    # 'SimMIM': mim.state_dict(),
                    'MoT' : mot.state_dict(), 
                    'seq_len': seq_len,
                    'n_joints': num_joints,
                    'n_heads' : mot_heads,
                    'mlp_dim' : mot_mlp_dim,
                    'dim': mot_dim,
                    'n_classes': n_classes,
                    'depth': mot_depth,
                    'optimizer_state_dict': optim.state_dict(),
                    'loss': loss}
            torch.save(ckpt, os.path.join(wdir, f'train-{epoch}.pt'))
            print(f"[MODEL SAVED at {epoch} Epoch]")

    wandb.run.finish()
    torch.cuda.empty_cache()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='runs/train', help='project/name')
    parser.add_argument('--data_path', type=str, default='/home/taehyun/workspace/childtoy/MotionReasoning/dataset/mocap_emotion_rig', help='Mat dataset path')
    parser.add_argument('--processed_data_dir', type=str, default='processed_data_mocam_80_All_Class_norm3/', help='path to save pickled processed data')
    parser.add_argument('--window', type=int, default=150, help='window')
    parser.add_argument('--weight_path', type=str, default='runs/train/humanact12_SIM_MIM_MOT_patch2_sub15_5122/weights/train-100.pt', help='pretrained weight path')    
    parser.add_argument('--wandb_pj_name', type=str, default='MoCAM', help='project name')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--device', default='3', help='cuda device')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--exp_name', default='humanact12_SIM_MIM_MOT_patch2_sub15_512_cls', help='save to project/name')
    parser.add_argument('--save_interval', type=int, default=100, help='Log model after every "save_period" epoch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='generator_learning_rate')

    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    opt.save_dir = str(increment_path(Path(opt.project) / opt.exp_name))
    opt.exp_name = opt.save_dir.split('/')[-1]
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")
    train(opt, device)

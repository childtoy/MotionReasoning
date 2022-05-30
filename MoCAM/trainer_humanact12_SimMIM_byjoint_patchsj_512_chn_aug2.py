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
from motion.dataset.humanact12 import HumanAct12Dataset2, humanact12_label_map

from data_proc.utils import increment_path
import torch
from model.MoT import MoT_patch2_seg_chn_sj
from model.SImMIM import SimMIM_patch2_seg_chn_sj
import math
def lerp_input_repr(input, valid_len, seq_len):

    amp_const = np.random.randint(8,12,[1])/10
    select_const = np.random.randint(1,3,[1])
    dataset = input.copy()
    data = dataset.copy()
    output = data.copy()
    mask_start_frame = 0

    t = np.arange(0, valid_len, 1)
    fs = valid_len
    dt = 1/fs
    x1 = np.arange(0, 1, dt)
    # nfft = 샘플 개수
    nfft = len(x1)
    # df = 주파수 증가량
    df = fs/nfft
    k = np.arange(nfft)
    # f = 0부터~최대주파수까지의 범위
    f = k*df 
    # 스펙트럼은 중앙을 기준으로 대칭이 되기 때문에 절반만 구함
    if valid_len % 2 :
        nfft_half = math.trunc(nfft/2)
    else : 
        nfft_half = math.trunc(nfft/2)+1
#     nfft_half = torch.trunc(nfft/2).to(device)
    f0 = f[range(0,nfft_half)] 
    # 증폭값을 두 배로 계산(위에서 1/2 계산으로 인해 에너지가 반으로 줄었기 때문) 
    for jt in range(22):
        for ax in range(6):
                # joint rw1 : 8 , x축
            y1 = data[:valid_len,jt,ax] - np.mean(data[:valid_len,jt,ax])
            fft_y = np.fft.fft(y1)/nfft * 2 
            fft_y0 = fft_y[(range(0,nfft_half))]
            # 벡터(복소수)의 norm 측정(신호 강도)
            amp = np.abs(fft_y0)
            idxy = np.argsort(-amp)   
            y_low5 = np.zeros(nfft)
            for i in range(len(idxy)): 
                freq = f0[idxy[i]] 
                yx = fft_y[idxy[i]] 
                coec = yx.real 
                coes = yx.imag * -1 
                if i < select_const :     
                    y_low5 += amp_const*(coec * np.cos(2 * np.pi * freq * x1) + coes * np.sin(2 * np.pi * freq * x1))                
                else : 
                    y_low5 += (coec * np.cos(2 * np.pi * freq * x1) + coes * np.sin(2 * np.pi * freq * x1))                                        
    
            y_low5 = y_low5 + np.mean(data[:valid_len,jt,ax])
                # print(torch.sum(input[:valid_len,jt,ax]-y_low5))
            data_low5 = y_low5

            if valid_len % 2:
                output[:valid_len, jt:jt+1,ax] = np.expand_dims(data_low5[:valid_len], axis=1)
            else : 
                output[:valid_len, jt:jt+1,ax] = np.expand_dims(data_low5[:valid_len], axis=1)
    dataset = output    
    return dataset

class DataAugmentation(object):
    def __init__(self, seq_len):
       self.seq_len = seq_len
    def __call__(self, query):
        motions, valid_len, labels, proc_labels = query['rotation_6d_pose_list'], query['valid_length_list'], query['labels'], query['proc_label_list'] 
        # augment_data = flip(motions)
        if np.random.random(1) <0.5 : 
            motions = lerp_input_repr(motions, valid_len, self.seq_len)

        return {'rotation_6d_pose_list': motions, 'valid_length_list': valid_len,'labels': labels, 'proc_label_list':proc_labels }



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
                          
    # Loggers
    #init wandb
    wandb.init(config=opt, project=opt.wandb_pj_name, entity=opt.entity, name=opt.exp_name, dir=opt.save_dir)

    # Load EmotionMoCap Dataset
    transform = DataAugmentation(seq_len=opt.window)
    train_dataset = HumanAct12Dataset2(data_path="../dataset/experiment/HumanAct12Poses/humanact12poses.pkl", motion_length=150, dataset="train", transform=transform)
    full_proc_label_list = list(humanact12_label_map.values())
    label_map = humanact12_label_map
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=128,
        num_workers=0,
        shuffle=True,
        pin_memory=True)

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
    epochs = opt.epochs
    n_hid = 70
    n_level = 4
    channel_sizes = [n_hid] * n_level
    kernel_size = 5
    
    mot = MoT_patch2_seg_chn_sj(seq_len = seq_len, num_joints=num_joints, sub_seq=30, num_classes=n_classes,  dim=mot_dim, depth=mot_depth, heads=mot_heads, mlp_dim = mot_mlp_dim)
    mim = SimMIM_patch2_seg_chn_sj( encoder = mot, masking_ratio = 0.5)
    mim.to(device)
    mim.train()

    optim = AdamW(params=mim.parameters(), lr=opt.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=100, gamma=0.9)
    loss_fct = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):

        pbar = tqdm(train_data_loader, position=1, desc="Batch")
        for batch in pbar:
            motions = batch["rotation_6d_pose_list"].to(device)
            valid_length = batch['valid_length_list'].to(device)

            B,T,J,C = motions.shape
            motions = motions.permute(0,3,1,2)
            # motions = motions.unsqueeze(1)
            # motions = motions.permute(0,1,3,2)
            optim.zero_grad()      
            loss = mim(motions.to(device), valid_length.to(device))
            loss.backward()
            optim.step()

        # Log
        log_dict = {
            "Train/Loss": loss, 
        }
        wandb.log(log_dict)

        # Save model
        if (epoch % save_interval) == 0:
            ckpt = {'epoch': epoch,
                    'SimMIM': mim.state_dict(),
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
    parser.add_argument('--wandb_pj_name', type=str, default='MoCAM', help='project name')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', default='2', help='cuda device')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--exp_name', default='humanact12_SIM_MIM_MOTsj_patch2_512_augtwo', help='save to project/name')
    parser.add_argument('--save_interval', type=int, default=50, help='Log model after every "save_period" epoch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='generator_learning_rate')

    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    opt.save_dir = str(increment_path(Path(opt.project) / opt.exp_name))
    opt.exp_name = opt.save_dir.split('/')[-1]
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")
    train(opt, device)

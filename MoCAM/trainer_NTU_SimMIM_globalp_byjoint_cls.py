import argparse
import os
from pathlib import Path
from matplotlib import transforms

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader,TensorDataset
from tqdm import tqdm

from data_proc.nturgb_data import Feeder
from data_proc.utils import increment_path
import torch
from  pytorch3d import transforms
from model.MoT import MoT_pos
from model.SImMIM import SimMIM_pos
import sys
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
    data_path ='/home/taehyun/workspace/childtoy/MotionReasoning/dataset/data/NTU-RGB-D/xview/train_data.npy'
    label_path ='/home/taehyun/workspace/childtoy/MotionReasoning/dataset/data/NTU-RGB-D/xview/train_label.pkl'
    Path(opt.processed_data_dir).mkdir(parents=True, exist_ok=True)
    ntu_dataset = Feeder(data_path,label_path)

    data_mean_p = np.mean(ntu_dataset.data, axis=(0,2,3,4))
    data_std_p = np.std(ntu_dataset.data, axis=(0,2,3,4))
    norm_data = (ntu_dataset.data.transpose(0,2,3,4,1) - data_mean_p) / data_std_p
    norm_label = ntu_dataset.label
    tensor_dataset = TensorDataset(torch.FloatTensor(norm_data), torch.FloatTensor(norm_label))
    ntu_data_loader = DataLoader(tensor_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)

    n_classes = 60
    seq_len=opt.window
    num_joints=25
    mot_dim = 1024
    mot_depth = 6
    mot_heads = 8 
    mot_mlp_dim = 2048
    mot_pool = 'cls'
    mot_channels =1, 
    mot_dim_head = 64
    epochs = opt.epochs
    n_hid = 70
    n_level = 4
    channel_sizes = [n_hid] * n_level
    kernel_size = 5
    
    mot = MoT_pos(seq_len = seq_len, num_joints=num_joints, num_classes=n_classes,  dim=mot_dim, depth=mot_depth, heads=mot_heads, mlp_dim = mot_mlp_dim)
    mim = SimMIM_pos( encoder = mot, masking_ratio = 0.5)
    mim.load_state_dict(ckpt['SimMIM'])
    mim.to(device)


    optim = AdamW(params=mot.parameters(), lr=opt.learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=100, gamma=0.9)
    selected_joint = [2,3,4,5,7,8,9,11,12,13,14,16,17,18,20,21,22,23,25,26,27,28,29,30,31,33,34,35,36,37,38,39,41,42,43]
    loss_fct = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):

        pbar = tqdm(ntu_data_loader, position=1, desc="Batch")
        for data, label in pbar:
            x = data
            N, T, V, M, C = x.size()
            labels = torch.cat([label.unsqueeze(1), label.unsqueeze(1)], axis=1)
            labels = labels.reshape(N*2)
            labels = labels.long().to(device)
            x1 = x[:,:,:,0,:]
            x2 = x[:,:,:,1,:]
            x = torch.cat([x1,x2],axis=0)
            # x = x.permute(0,1,2,3).contiguous()
            # x = x.view(N * M, T, V, C)

            # batch_size, seq_len, num_j, _  = global_p.shape
            # data = data.reshape(batch_size, seq_len, num_j, 6)
            optim.zero_grad()      
            logits = mot(x.to(device))
            logits = logits
            loss = loss_fct(logits.view(-1, n_classes), labels.view(-1))     
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
    parser.add_argument('--processed_data_dir', type=str, default='processed_data_mocam_80_All_Class_norm2/', help='path to save pickled processed data')
    parser.add_argument('--weight_path', type=str, default='runs/train/SIM_MIM_MOT_NTU_norm_globalp3/weights/train-1000.pt', help='pretrained weight path')
    parser.add_argument('--window', type=int, default=300, help='window')
    parser.add_argument('--wandb_pj_name', type=str, default='MoCAM', help='project name')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--device', default='0', help='cuda device')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--exp_name', default='SIM_MIM_MOT_NTU_norm_globalp_cls', help='save to project/name')
    parser.add_argument('--save_interval', type=int, default=50, help='Log model after every "save_period" epoch')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='generator_learning_rate')

    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    opt.save_dir = str(increment_path(Path(opt.project) / opt.exp_name))
    opt.exp_name = opt.save_dir.split('/')[-1]
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")
    train(opt, device)

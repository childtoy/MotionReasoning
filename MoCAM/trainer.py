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
from model.model import TCN
import torch.nn.functional as F

def train(opt, device):
    # Prepare Directories
    save_dir = Path(opt.save_dir)
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)

    # Save run settings
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=True)

    epochs = opt.epochs
    save_interval = opt.save_interval
                          
    # Loggers
    #init wandb
    # wandb.init(config=opt, project=opt.wandb_pj_name, entity=opt.entity, name=opt.exp_name, dir=opt.save_dir)

    # Load EmotionMoCap Dataset
    Path(opt.processed_data_dir).mkdir(parents=True, exist_ok=True)
    emotion_dataset = EmotionDataset(data_dir=opt.data_path, processed_data_dir=opt.processed_data_dir, train=True, device=device, window=opt.window)
    emotion_data_loader = DataLoader(emotion_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)

    n_classes = 7
    input_channels = 105
    seq_length = opt.window
    epochs = opt.epochs
    n_hid = 256
    n_level = 4
    channel_sizes = [n_hid] * n_level
    model = TCN(input_channels, n_classes, channel_sizes, kernel_size=5, dropout=0.05)
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    optim = AdamW(params=model.parameters(), lr=opt.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=100, gamma=0.9)

    for epoch in range(1, epochs + 1):

        pbar = tqdm(emotion_data_loader, position=1, desc="Batch")
        for batch in pbar:
            
            local_q = batch["local_q"].to(device)
            q_vec = batch["q_vel"].to(device) 
            q_acc = batch["q_acc"].to(device) 
            labels = batch["labels"].to(device)
            data = torch.cat([local_q, q_vec, q_acc], axis=2)
            data = data.permute(0,2,1)
            output = model(data)
            optim.zero_grad()      
            loss = criterion(output, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False)
            optim.step()

        scheduler.step()
        # Log
        log_dict = {
            "Train/Loss": loss, 
        }
        wandb.log(log_dict)

        # Save model
        if (epoch % save_interval) == 0:
            ckpt = {'epoch': epoch,
                    'TCN': TCN.state_dict(),
                    'input_channels': input_channels,
                    'n_hid': n_hid,
                    'n_level': n_level,
                    'n_classes': n_classes,
                    'optimizer_state_dict': optim.state_dict(),
                    'loss': loss}
            torch.save(ckpt, os.path.join(wdir, f'train-{epoch}.pt'))
            print(f"[MODEL SAVED at {epoch} Epoch]")

    wandb.run.finish()
    torch.cuda.empty_cache()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='runs/train', help='project/name')
    parser.add_argument('--data_path', type=str, default='/Users/taehyun/workspace/childtoy/MotionReasoning/style_transfer/data/mocap_emotion_rig', help='Mat dataset path')
    parser.add_argument('--processed_data_dir', type=str, default='processed_data_mocam/', help='path to save pickled processed data')
    parser.add_argument('--window', type=int, default=40, help='window')
    parser.add_argument('--wandb_pj_name', type=str, default='MoCAM', help='project name')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', default='0', help='cuda device')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--exp_name', default='exp', help='save to project/name')
    parser.add_argument('--save_interval', type=int, default=10, help='Log model after every "save_period" epoch')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='generator_learning_rate')

    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    opt.save_dir = str(increment_path(Path(opt.project) / opt.exp_name))
    opt.exp_name = opt.save_dir.split('/')[-1]
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")
    train(opt, device)

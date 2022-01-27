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
from model.model import TCN, PURE3D
import torch.nn.functional as F


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
    input_channels = 3
    n_hid = 70
    n_level = 4
    n_channels = 64
    channel_sizes = [n_hid] * n_level
    kernel_size = 5
    model = PURE3D(input_channels, n_channels, n_classes, kernel_size=kernel_size, dropout=0)
    model.load_state_dict(ckpt['TCN'])
    model.eval()
    correct = 0
    pbar = tqdm(emotion_data_loader, position=1, desc="Batch")
    confusion_matrix = torch.zeros(7, 7)
    with torch.no_grad():
        for batch in pbar:
            local_q = batch["local_q"].to(device)
            q_vel = batch["q_vel"].to(device) 
            q_acc = batch["q_acc"].to(device) 
            labels = batch["labels"].to(device)
            data = torch.cat([local_q, q_vel, q_acc], axis=2)
            data = torch.cat([local_q.unsqueeze(3), q_vel.unsqueeze(3), q_acc.unsqueeze(3)], axis=3)
            data = data.permute(0,3,2,1)
            print(data.shape)
            output = model(data)
            prediction = output.data.max(1)[1]
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
    parser.add_argument('--weight', default='latest')
    parser.add_argument('--exp_name', default='exp148', help='experiment name')
    parser.add_argument('--data_path', type=str, default='/home/taehyun/workspace/childtoy/MotionReasoning/dataset/mocap_emotion_rig', help='BVH dataset path')
    parser.add_argument('--window', type=int, default=80, help='window')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')

    parser.add_argument('--processed_data_dir', type=str, default='processed_data_mocam/', help='path to save pickled processed data')
    parser.add_argument('--save_path', type=str, default='runs/test', help='path to save model')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    device = torch.device("cpu")
    test(opt, device)

import argparse
import os
from pathlib import Path
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import wandb
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from pytorch3d.transforms.rotation_conversions import (matrix_to_axis_angle,
                                                       rotation_6d_to_matrix)
from motion.dataset.humanact12 import HumanAct12Dataset, humanact12_label_map
from sklearn.preprocessing import LabelEncoder
from data_proc.utils import increment_path
import torch.nn.functional as F
import cv2
import numpy as np
import torch
from torchvision import models
import random
import colorsys
import requests
from io import BytesIO
from pathlib import Path

import skimage.io
from skimage.measure import find_contours
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image

from data_proc.utils import increment_path
import torch.nn.functional as F

from model.MoT import MoT_patch2_seg
from model.SImMIM import SimMIM
from mpl_toolkits import mplot3d
import sys
import math
from external_repos.human_body_prior.src.human_body_prior.models.ik_engine import IK_Engine
from typing import Union

joint_names = [
        'pelvis',
        'left_hip',
        'right_hip',
        'spine1',
        'left_knee',
        'right_knee',
        'spine2',
        'left_ankle',
        'right_ankle',
        'spine3',
        'left_foot',
        'right_foot',
        'neck',
        'left_collar',
        'right_collar',
        'head',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist']


selected_joint_names = []
selected_joint = list(range(22))
for i in selected_joint:
    selected_joint_names.append(joint_names[i])

import pathlib


def lerp_input_repr(input, valid_len, seq_len, device):
    output = input.clone()
    mask_start_frame = 0
    torch_pi = torch.acos(torch.zeros(1)).item() * 2
    # torch_pi = torch_pi.to(device)
    # torch.pi = torch.acos(torch.zeros(1)).item() * 2
    # data_low5 = input.clone()
    t = torch.arange(0, valid_len, 1).to(device)
    fs = valid_len
    dt = 1/fs
    x1 = torch.arange(0, 1, dt).to(device)
    # nfft = 샘플 개수
    nfft = torch.tensor(len(x1)).to(device)
    # df = 주파수 증가량
    df = torch.tensor(fs/nfft).to(device)
    k = torch.arange(nfft).to(device)
    # f = 0부터~최대주파수까지의 범위
    f = k*df 
    # 스펙트럼은 중앙을 기준으로 대칭이 되기 때문에 절반만 구함
    if valid_len % 2 :
        nfft_half = torch.trunc(nfft/2).to(device)
    else : 
        nfft_half = torch.trunc(nfft/2).to(device)+1
#     nfft_half = torch.trunc(nfft/2).to(device)
    f0 = f[(torch.range(0,nfft_half.long())).long()] 
    # 증폭값을 두 배로 계산(위에서 1/2 계산으로 인해 에너지가 반으로 줄었기 때문)    
    for jt in range(22):
        for ax in range(6):
                # joint rw1 : 8 , x축
            y1 = input[:valid_len,jt,ax] - torch.mean(input[:valid_len,jt,ax])
            fft_y = torch.fft.fft(y1)/nfft * 2 
            fft_y0 = fft_y[(torch.range(0,nfft_half.long())).long()]
            # 벡터(복소수)의 norm 측정(신호 강도)
            amp = torch.abs(fft_y0)
            idxy = torch.argsort(-amp)   
            y_low5 = torch.zeros(nfft).to(device)
            for i in range(1,int(valid_len/4)): 
                freq = f0[idxy[i]] 
                yx = fft_y[idxy[i]] 
                coec = yx.real 
                coes = yx.imag * -1 
                # if i < 8 and i >3:     
                y_low5 += (coec * torch.cos(2 * torch_pi * freq * x1) + coes * torch.sin(2 * torch_pi * freq * x1))                
       
            y_low5 = y_low5 + torch.mean(input[:valid_len,jt,ax])
                # print(torch.sum(input[:valid_len,jt,ax]-y_low5))
            data_low5 = y_low5
            if valid_len % 2:
                output[:valid_len, jt:jt+1,ax] = data_low5[:valid_len].unsqueeze(1)
            else : 
                output[:valid_len, jt:jt+1,ax] = data_low5[:valid_len].unsqueeze(1)
    return output


def plot_single_pose(
    pose,
    bm, 
    frame_idx,
    minx,
    maxx,
    miny,
    maxy,
    minz,
    maxz,
    save_dir,
    prefix,
    prefixxx,
):
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(10,75)  #각 변경
#     selected_joint = [2,3,4,5,7,8,9,11,12,13,14,16,17,18,20,21,22,23,25,26,27,28,29,30,31,33,34,35,36,37,38,39,41,42,43]
#     pose = pose[selected_joint]
    # heatmap = (heatmap - min_heat)/(max_heat - min_heat)
    for i, p in enumerate(bm.kintree_table[0][:22]):
        if i > 2 :
            ax.plot(
                [pose[i, 0], pose[p, 0]],
                [pose[i, 2], pose[p, 2]],
                [pose[i, 1], pose[p, 1]],
                c='k',
            )
        sc = ax.scatter(                
            pose[i, 0],
            pose[i, 2],
            pose[i, 1], 
            color='k',s=400)
    x_min = minx
    x_max = maxx
    y_min = miny
    y_max = maxy
    z_min = minz
    z_max = maxz

    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("$X$ Axis")
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("$Y$ Axis")
    ax.set_zlim(z_min, z_max)
    ax.set_zlabel("$Z$ Axis")
    plt.draw()

    title = f"{prefixxx}: {frame_idx}"
    plt.title(title)
    prefix = prefix
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(save_dir, prefix + str(frame_idx) + ".png"), dpi=60)
    plt.close()


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# build model
# Load LAFAN Dataset
test_dataset = HumanAct12Dataset(data_path="../dataset/experiment/HumanAct12Poses/humanact12poses.pkl", motion_length=150, dataset="test")
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
n_classes = 12
seq_len=150
num_joints=22

origin_data = iter(test_data_loader).next()
motions = origin_data["rotation_6d_pose_list"].to(device)
motions_ = motions.clone()

print('motions shape : ', motions_.shape)
valid_length = origin_data["valid_length_list"].to(device) 
labels = origin_data["labels"]
labels = le.transform(labels)
labels = torch.Tensor(labels)
labels = labels.long().to(device)

subject_gender = 'neutral'
bm_fname = os.path.join('smpl_model', subject_gender, 'model.npz')
dmpl_fname = os.path.join('dmpl_model', subject_gender, 'model.npz')
num_betas = 10
num_dmpls = 8
bm = BodyModel(bm_fname=bm_fname, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname).to(device)
for exam_idx in range(119):
    motion = motions_[exam_idx]
    motion1 = motion.clone()
    motion = lerp_input_repr(motion, valid_length[exam_idx], 150, device=device)
    print(torch.sum(motion1 - motion))
    smpl_param1 = matrix_to_axis_angle(rotation_6d_to_matrix(motion1[:valid_length[exam_idx]])).reshape(valid_length[exam_idx], -1)
    smpl_param = matrix_to_axis_angle(rotation_6d_to_matrix(motion[:valid_length[exam_idx]])).reshape(valid_length[exam_idx], -1)
    root_orient = smpl_param[:, :3]
    pose_body = smpl_param[:, 3:66]

    # root_orient1 = smpl_param1[:, :3]
    # pose_body1 = smpl_param1[:, 3:66]
    # body_params1 = {
    # 'root_orient': root_orient1, # controls the global root orientation
    # 'pose_body': pose_body1, # controls the body
    # # 'pose_hand': torch.Tensor(pose_hand).to(device), # controls the finger articulation
    # }
    # body_pose_beta1 = bm1(**{k:v for k,v in body_params1.items() if k in ['pose_body', 'betas']})
    # global_pose1 = body_pose_beta1.Jtr[:,:22,:]

    # pose_hand = smpl_param[:, 66:]
    body_params = {
    'root_orient': root_orient, # controls the global root orientation
    'pose_body': pose_body, # controls the body
    # 'pose_hand': torch.Tensor(pose_hand).to(device), # controls the finger articulation
    }
    body_pose_beta = bm(**{k:v for k,v in body_params.items() if k in ['pose_body', 'betas']})
    # save attentions heatmaps
    print('write start')
    from PIL import Image
    import imageio
    labels_name = full_proc_label_list
    prev_file = ''
    prefixx = str(exam_idx)+'GT:'+ le.inverse_transform([labels[exam_idx].cpu().numpy()])[0]
    i=0
    save_path = os.path.join('humanact12_6d')
    # if save_path == prev_file :
    #     continue
    Path(save_path).mkdir(parents=True, exist_ok=True)
    input_img_path = os.path.join(save_path, 'tmp')
    # Path(os.path.join(input_img_path, 'tmp')).mkdir(parents=True, exist_ok=True)

    img_aggr_list = []
    global_pose = body_pose_beta.Jtr[:,:22,:]
    # print(torch.sum(global_pose - global_pose1))
    minx = np.min(global_pose[:,:,0].cpu().numpy())
    maxx = np.max(global_pose[:,:,0].cpu().numpy())
    miny = np.min(global_pose[:,:,2].cpu().numpy())
    maxy = np.max(global_pose[:,:,2].cpu().numpy())
    minz = np.min(global_pose[:,:,1].cpu().numpy())
    maxz = np.max(global_pose[:,:,1].cpu().numpy())
    for t in range(global_pose.shape[0]):
        plot_single_pose(global_pose[t].cpu().numpy(),bm, t, minx,maxx,miny,maxy,minz,maxz,input_img_path, 'input', prefixx)
        input_img = Image.open(os.path.join(input_img_path, 'input'+str(t)+'.png'), 'r')
        img_aggr_list.append(input_img)
    # Save images

    prdstr = str(labels_name[labels[i].cpu().numpy()])
    gif_path = os.path.join('humanact12_6d','humanact12_6d'+str(prefixx)+'.gif')
    imageio.mimsave(gif_path, img_aggr_list, duration=0.1)
    print(f"ID {exam_idx}: test completed.")


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

from data_proc.emotionmocap_dataset import EmotionDataset
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
import pathlib

import skimage.io
from skimage.measure import find_contours
import matplotlib

import matplotlib.pyplot as plt

from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image

from data_proc.emotionmocap_dataset import EmotionDataset
from data_proc.utils import increment_path
import torch.nn.functional as F

from model.MoT import MoT
from model.SImMIM import SimMIM
from mpl_toolkits import mplot3d

# 
def make_attention(model_output):
    # we keep only the output patch attention
    threshold = None
    nh = model_output.shape[1]
    attentions_joint = model_output[0, :, 0,1:].reshape(nh, -1)
    w_featmap_joint = 1
    h_featmap_joint = 35
    # w_featmap_time = 80
    # h_featmap_time = 80 
#     if threshold is not None:
#         # we keep only a certain percentage of the mass
#         val, idx = torch.sort(attentions_joint)
#         val /= torch.sum(val, dim=1, keepdim=True)
#         cumval = torch.cumsum(val, dim=1)
#         th_attn = cumval > (1 - threshold)
#         idx2 = torch.argsort(idx)
#         for head in range(nh):
#             th_attn[head] = th_attn[head][idx2[head]]
#         th_attn = th_attn.reshape(nh, w_featmap_joint, h_featmap_joint).float()
        # interpolate
    #     th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()
#     print(attentions_joint.shape,'attentions shape')
    attentions_joint = attentions_joint.reshape(nh, w_featmap_joint, h_featmap_joint)
    attention_joint_inp = nn.functional.interpolate(attentions_joint.unsqueeze(0), scale_factor=(20,1), mode="nearest")[0].cpu().numpy()

    return attentions_joint, attention_joint_inp


def plot_pre_heatmap(data, jointnames, is_save=False, savepath=''):
    fig = plt.figure(figsize=(15,15),facecolor='white')
    ax = fig.add_subplot(111)
    heatmap = ax.pcolor(data)
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('Joint Number', fontsize=14)
    ax.set_yticks(list(range(35)))
    ax.set_yticklabels(jointnames)
    if is_save : 
        plt.savefig(savepath,facecolor=fig.get_facecolor(), transparent=False)
        plt.close()
    else : 
        plt.show()

def find_idx_topk_attention(data, seq_len=80, select_k=3):
    expand_seq = seq_len/data.shape[0]
    top_k_value = np.sort(data.reshape(-1).cpu().numpy())[::-1][:select_k]
    seq_list = []
    joint_list = []
    for i in top_k_value:
        idxs = np.where(i==data.cpu().numpy())
        joint_list.append(idxs[1][0])
        seq_list.append(int(0))
    return joint_list, seq_list, top_k_value

def lerp_input_repr(input, target_seqs, target_joints, seq_len):
    output = input.clone()
    mask_start_frame = 0
    for joint_idx, start_seq in zip(target_joints, target_seqs):
        minibatch_pose_input = input[start_seq:start_seq+seq_len, joint_idx: joint_idx+1]
        minibatch_pose_input = minibatch_pose_input.unsqueeze(0)
        seq_len = seq_len
        interpolated = torch.zeros_like(minibatch_pose_input, device=minibatch_pose_input.device)

        # if mask_start_frame == 0 or mask_start_frame == (seq_len -1):
        #     interpolate_start = minibatch_pose_input[:,0,:]
        #     interpolate_end = minibatch_pose_input[:,seq_len-1,:]

        #     for i in range(seq_len):
        #         dt = 1 / (seq_len-1)
        #         interpolated[:,i,:] = torch.lerp(interpolate_start, interpolate_end, dt * i)

        #     assert torch.allclose(interpolated[:,0,:], interpolate_start)
        #     assert torch.allclose(interpolated[:,seq_len-1,:], interpolate_end)
        # else:
        #     interpolate_start1 = minibatch_pose_input[:,0,:]
        #     interpolate_end1 = minibatch_pose_input[:,mask_start_frame,:]

        #     interpolate_start2 = minibatch_pose_input[:, mask_start_frame, :]
        #     interpolate_end2 = minibatch_pose_input[:, -1,:]

        #     for i in range(mask_start_frame+1):
        #         dt = 1 / mask_start_frame
        #         interpolated[:,i,:] = torch.lerp(interpolate_start1, interpolate_end1, dt * i)

        #     assert torch.allclose(interpolated[:,0,:], interpolate_start1)
        #     assert torch.allclose(interpolated[:,mask_start_frame,:], interpolate_end1)

        #     for i in range(mask_start_frame, seq_len):
        #         dt = 1 / (seq_len - mask_start_frame - 1)
        #         interpolated[:,i,:] = torch.lerp(interpolate_start2, interpolate_end2, dt * (i - mask_start_frame))

        #     assert torch.allclose(interpolated[:,mask_start_frame,:], interpolate_start2)
        #     assert torch.allclose(interpolated[:,-1,:], interpolate_end2)
        output[start_seq:start_seq+seq_len, joint_idx:joint_idx+1] =  torch.zeros([80,1])
    return output




patch_size = 8
pretrained_weights = ''
image_path = None
image_size = (480,480)
output_dir = 'SIM_MIM_MOT_patch20_2_cls_zero_all'
threshold = None
project = 'runs/train'
weight = 'latest'

exp_name='SIM_MIM_MOT_addR_joint_cls4'
data_path=''
window=80
batch_size=1

processed_data_dir='processed_data_mocam_80_All_Class_addR2'
save_path='runs/test'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# build model

save_dir = Path(os.path.join('runs', 'train', exp_name))
wdir = save_dir / 'weights'
weights = os.listdir(wdir)
weight = '200'
if weight == 'latest':
    weights_paths = [wdir / weight for weight in weights]
    print(weights_paths)
    weight_path = max(weights_paths , key = os.path.getctime)
else:
    weight_path = wdir / ('train-' + weight + '.pt')
print(weight_path)
ckpt = torch.load(weight_path, map_location=device)
print(f"Loaded weight: {weight_path}")
# Load LAFAN Dataset
Path(processed_data_dir).mkdir(parents=True, exist_ok=True)
emotion_dataset = EmotionDataset(data_dir=data_path, processed_data_dir=processed_data_dir, train=False, device=device, window=window)
emotion_data_loader = DataLoader(emotion_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
n_classes = 7
seq_len=window
num_joints=35
mot_dim = 1024
mot_depth = 6
mot_heads = 8 
mot_mlp_dim = 2048
mot_pool = 'cls'
mot_channels =1, 
mot_dim_head = 64
target_joint = 'joint'
model = MoT(seq_len = seq_len, num_joints=num_joints, num_classes=n_classes,  dim=mot_dim, depth=mot_depth, heads=mot_heads, mlp_dim = mot_mlp_dim)
# mim = SimMIM_patch( encoder = mot, masking_ratio = 0.5)
model.load_state_dict(ckpt['MoT'])
for p in model.parameters():
    p.requires_grad = False
model.eval()
model.to(device)
# pbar = tqdm(emotion_data_loader, position=1, desc="Batch")
origin_data = iter(emotion_data_loader).next()
local_q = origin_data["local_q"].to(device)
q_vel = origin_data["q_vel"].to(device) 
q_acc = origin_data["q_acc"].to(device) 
labels = origin_data["labels"].to(device)     
data = local_q
# data = data.permute(0,1,3,2)
attentions_joint = model(data.to(device), return_attention=True)
nh = attentions_joint.shape[1] # number of head
logits_joint = model(data.to(device))
output = logits_joint

joint_names = ['world','base','root1','root2','root3','spine','neck','rs1','rs2','rs3','re1','re2','rw1','rw2','rw3','rh','ls1','ls2','ls3'
,'le1','le2','lw1','lw2','lw3','lh','rp1','rp2','rp3','rk','ra1','ra2','ra3','rf','lp1','lp2','lp3','lk','la1','la2','la3','lf','head1','head2','head3']

selected_joint_names = []
selected_joint = [2,3,4,5,7,8,9,11,12,13,14,16,17,18,20,21,22,23,25,26,27,28,29,30,31,33,34,35,36,37,38,39,41,42,43]
for i in selected_joint:
    selected_joint_names.append(joint_names[i])
label_list = ['angry','disgust','fearful','happy','neutral','sad','surprise']
save_dir = './result_attention_patch2_val_zeropad_all'
pbar = tqdm(emotion_data_loader, position=1, desc="Batch")
correct = 0
correct1 = 0
correct2 = 0
correct3 = 0
correct_pred_top3 = []
correct_pred_top2 = []
correct_pred_top1 = []
pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
idxxx = 0 
with torch.no_grad():
    for batch in pbar:
        local_q = batch["local_q"].to(device)
        labels = batch["labels"].to(device)     
        filename = batch['filename']
        save_filename = filename[0][:-4]
        data = local_q
        attentions_joint = model(data.to(device), return_attention=True)
        nh = attentions_joint.shape[1] # number of head
        logits_joint = model(data.to(device))
        output = logits_joint
        pred_origin = output.data.max(1)[1]
        attn_org1, attn_inp1 = make_attention(attentions_joint)
        grayscale_cam = np.transpose(attn_inp1[0])   
        # if idxxx%10 == 0:
        #     print('ab',save_filename+'_'+label_list[pred_origin]+'_origin')
            # plot_pre_heatmap(grayscale_cam,selected_joint_names, is_save=True, savepath=os.path.join(save_dir,save_filename+'_'+label_list[pred_origin]+'_origin.png'))


        sjoint_idxs, sseq_idxs, sattn_values = find_idx_topk_attention(attn_org1[0])

        lerp_data = lerp_input_repr(data[0],sseq_idxs,sjoint_idxs,80)
        attentions_joint_lerp = model(lerp_data.unsqueeze(0).to(device), return_attention=True)
        output2 = model(lerp_data.unsqueeze(0).to(device))
        pred_top3 = output2.data.max(1)[1]
        attn_org2, attn_inp2 = make_attention(attentions_joint_lerp)
        grayscale_cam = np.transpose(attn_inp2[0])        
        # if idxxx%10 == 0:
        #     plot_pre_heatmap(grayscale_cam,selected_joint_names, is_save=True, savepath=os.path.join(save_dir,save_filename+'_'+label_list[pred_top3]+'_top3.png'))
        
        sjoint_idxs, sseq_idxs, sattn_values = find_idx_topk_attention(attn_org1[0], 80, 2)
        lerp_data = lerp_input_repr(data[0],sseq_idxs,sjoint_idxs,80)
        attentions_joint_lerp = model(lerp_data.unsqueeze(0).to(device), return_attention=True)
        output3 = model(lerp_data.unsqueeze(0).to(device))
        pred_top2 = output3.data.max(1)[1]
        attn_org2, attn_inp2 = make_attention(attentions_joint_lerp)
        grayscale_cam = np.transpose(attn_inp2[0])        
        # if idxxx%10 == 0:
        #     plot_pre_heatmap(grayscale_cam,selected_joint_names, is_save=True, savepath=os.path.join(save_dir,save_filename+'_'+label_list[pred_top2]+'_top2.png'))
        sjoint_idxs, sseq_idxs, sattn_values = find_idx_topk_attention(attn_org1[0], 80, 1)
        lerp_data = lerp_input_repr(data[0],sseq_idxs,sjoint_idxs,80)
        attentions_joint_lerp = model(lerp_data.unsqueeze(0).to(device), return_attention=True)
        output4 = model(lerp_data.unsqueeze(0).to(device))
        pred_top1 = output4.data.max(1)[1]
        attn_org2, attn_inp2 = make_attention(attentions_joint_lerp)
        grayscale_cam = np.transpose(attn_inp2[0])        
        # if idxxx%10 == 0:
        #     plot_pre_heatmap(grayscale_cam,selected_joint_names, is_save=True, savepath=os.path.join(save_dir,save_filename+'_'+label_list[pred_top1]+'_top1.png'))
                         
        correct += pred_origin.eq(labels).sum()
        correct1 += pred_top1.eq(labels).sum()
        correct2 += pred_top2.eq(labels).sum()
        correct3 += pred_top3.eq(labels).sum()

        correct_pred_top3.append(pred_origin.eq(pred_top3).cpu().numpy()[0])
        correct_pred_top2.append(pred_origin.eq(pred_top2).cpu().numpy()[0])
        correct_pred_top1.append(pred_origin.eq(pred_top1).cpu().numpy()[0])
        idxxx +=1

print('correct_pred_top3',2512-sum(correct_pred_top3))
print('correct_pred_top2', 2512-sum(correct_pred_top2))
print('correct_pred_top1', 2512-sum(correct_pred_top1))
print('correct',correct/2512)
print('correct',correct1/2512)
print('correct',correct2/2512)
print('correct',correct3/2512)


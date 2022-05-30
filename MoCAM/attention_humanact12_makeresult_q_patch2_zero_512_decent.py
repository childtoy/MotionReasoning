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

from motion.dataset.human36m import Human36mDataset, human36m_label_map
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
import pathlib

import skimage.io
from skimage.measure import find_contours
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import sys
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

from model.MoT import MoT_patch2_seg
from model.SImMIM import SimMIM
from mpl_toolkits import mplot3d

# 

def make_attention(model_output):
    # we keep only the output patch attention
    threshold = None
    nh = model_output.shape[1]
    attentions_joint = model_output[0, :, 0,1:].reshape(nh, -1)
    w_featmap_joint = 5
    h_featmap_joint = 22
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
    ax.set_yticks(list(range(140)))
    ax.set_yticklabels(jointnames)
    if is_save : 
        plt.savefig(savepath,facecolor=fig.get_facecolor(), transparent=False)
        plt.close()
    else : 
        plt.show()

def find_idx_topk_attention(data, seq_len=80, select_k=3):
    expand_seq = seq_len/data.shape[0]
    # sorted_data = np.sort(data.cpu().numpy(), axis=0)[0,:]
    # top_k_value = np.sort(sorted_data.reshape(-1))[:select_k]
    # top_k_value = np.sort(data.reshape(-1).cpu().numpy())[:select_k]
    top_k_value = np.sort(data.reshape(-1).cpu().numpy())[::-1][:select_k]
    print(top_k_value)
    seq_list = []
    joint_list = []
    for i in top_k_value:
        idxs = np.where(i==data.cpu().numpy())
        joint_list.append(idxs[1][0]*6)
        seq_list.append(int(idxs[0][0]*expand_seq))
    return joint_list, seq_list, top_k_value

def lerp_input_repr(input, target_seqs, target_joints, seq_len):
    output = input.clone()
    mask_start_frame = 0
    print(target_joints)
    print(target_seqs)
    for joint_idx, start_seq in zip(target_joints, target_seqs):
        minibatch_pose_input = input[0:1,start_seq:start_seq+seq_len, joint_idx: joint_idx+1]
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
        output[:,start_seq:start_seq+seq_len, joint_idx:joint_idx+6] = torch.Tensor([1,0,0,0,1,0]).repeat(seq_len,1).unsqueeze(0)
    return output


patch_size = 8
pretrained_weights = ''
image_path = None
image_size = (480,480)
output_dir = 'humanact12_SIM_MIM_MOT_patch2_512_cls2'
threshold = None
project = 'runs/train'
weight = '500'

exp_name='humanact12_SIM_MIM_MOT_patch2_512_cls2'
data_path=''
window=80
batch_size=1

processed_data_dir='processed_data_mocam_80_All_Class_addR2'
save_path='runs/test'

device = torch.device("cuda:3") if torch.cuda.is_available() else torch.device("cpu")
# build model

save_dir = Path(os.path.join('runs', 'train', exp_name))
wdir = save_dir / 'weights'
weights = os.listdir(wdir)
weight = '850'
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
# num_pair = 105
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

model = MoT_patch2_seg(seq_len = seq_len, num_joints=num_joints, sub_seq=30, num_classes=n_classes,  dim=mot_dim, depth=mot_depth, heads=mot_heads, mlp_dim = mot_mlp_dim)
# mim = SimMIM_patch( encoder = mot, masking_ratio = 0.5)
model.load_state_dict(ckpt['MoT'])
for p in model.parameters():
    p.requires_grad = False
model.eval()
model.to(device)
# pbar = tqdm(emotion_data_loader, position=1, desc="Batch")
origin_data = iter(test_data_loader).next()
motions = origin_data["rotation_6d_pose_list"].to(device)
valid_length = origin_data['valid_length_list'].to(device)
labels = origin_data["labels"]
print(motions.shape)
labels = le.fit_transform(labels)
labels = torch.Tensor(labels)
labels = labels.long().to(device)
B,T,J,C = motions.shape
motions = motions.view(B,T,J*C)
motions = motions.unsqueeze(1)

attentions_joint = model(motions.to(device), valid_length.to(device), return_attention=True)
# data = data.permute(0,1,3,2)
nh = attentions_joint.shape[1] # number of head
logits_joint = model(motions.to(device), valid_length.to(device))
output = logits_joint

joint_names = ['Pelvis','L_hip','R_hip','Spine1','L_Knee','R_Knee','Spine2','L_Ankle','R_Ankle','Spine3','L_Foot','R_Foot','Neck','L_Collar','R_Collar','Head','L_Shoulder','R_Shoulder','L_Elbow'
,'R_Elbow','L_Wrist','R_Wrist']

selected_joint_names = []
selected_joint = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
for i in selected_joint:
    selected_joint_names.append(joint_names[i])

label_list = ["warm up","walk","run","jump","drink","lift dumbbell","sit","eat","turn steering wheel","phone","boxing","throw"]
save_dir = './result_attention_q_patch2_val_zero_pad_512'
pbar = tqdm(test_data_loader, position=1, desc="Batch")
correct = 0
correct1 = 0
correct2 = 0
correct3 = 0
correct4 = 0
correct5 = 0
correct6 = 0
correct7 = 0
correct8 = 0
correct9 = 0
correct10 = 0
correct11 = 0
correct12 = 0
correct13 = 0
correct14 = 0
correct15 = 0
correct_pred_top3 = []
correct_pred_top2 = []
correct_pred_top1 = []
pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
idxxx = 0 
joint_list = torch.Tensor([0,3,4,7,8,11,14,15,18,21,22,25,28,29,32]).long()
# comblist = torch.combinations(joint_list)
# for i in comblist:
#     selected_joint_names.append(str(joint_names[i[0]])+'-'+str(joint_names[i[1]]))

with torch.no_grad():
    for batch in pbar:
        # if idxxx%100 == 0:
        motions = batch["rotation_6d_pose_list"].to(device)
        valid_length = batch['valid_length_list'].to(device)
        labels = batch["labels"]
        labels = le.fit_transform(labels)
        labels = torch.Tensor(labels)
        labels = labels.long().to(device)
        B,T,J,C = motions.shape
        motions = motions.view(B,T,J*C)
        motions = motions.unsqueeze(1)
        data = motions
        attentions_joint = model(motions.to(device), valid_length.to(device), return_attention=True)
        nh = attentions_joint.shape[1] # number of head

        logits_joint = model(motions.to(device), valid_length.to(device))
        output = logits_joint
        pred_origin = output.data.max(1)[1]
        attn_org1, attn_inp1 = make_attention(attentions_joint)
        grayscale_cam = np.transpose(attn_inp1[0])   

        sjoint_idxs, sseq_idxs, sattn_values = find_idx_topk_attention(attn_org1[0], 150, 1)
        lerp_data = lerp_input_repr(data[0],sseq_idxs,sjoint_idxs,30)
        attentions_joint_lerp = model(lerp_data.unsqueeze(0).to(device), valid_length.to(device), return_attention=True)
        output2 = model(lerp_data.unsqueeze(0).to(device), valid_length.to(device))
        pred_top1 = output2.data.max(1)[1]
        
        sjoint_idxs, sseq_idxs, sattn_values = find_idx_topk_attention(attn_org1[0], 150, 2)
        lerp_data = lerp_input_repr(data[0],sseq_idxs,sjoint_idxs,30)
        attentions_joint_lerp = model(lerp_data.unsqueeze(0).to(device), valid_length.to(device), return_attention=True)
        output2 = model(lerp_data.unsqueeze(0).to(device), valid_length.to(device))
        pred_top2 = output2.data.max(1)[1]
        
        sjoint_idxs, sseq_idxs, sattn_values = find_idx_topk_attention(attn_org1[0], 150, 3)
        lerp_data = lerp_input_repr(data[0],sseq_idxs,sjoint_idxs,30)
        attentions_joint_lerp = model(lerp_data.unsqueeze(0).to(device), valid_length.to(device), return_attention=True)
        output2 = model(lerp_data.unsqueeze(0).to(device), valid_length.to(device))
        pred_top3 = output2.data.max(1)[1]        
        
        sjoint_idxs, sseq_idxs, sattn_values = find_idx_topk_attention(attn_org1[0], 150, 4)
        lerp_data = lerp_input_repr(data[0],sseq_idxs,sjoint_idxs,30)
        attentions_joint_lerp = model(lerp_data.unsqueeze(0).to(device), valid_length.to(device), return_attention=True)
        output2 = model(lerp_data.unsqueeze(0).to(device), valid_length.to(device))
        pred_top4 = output2.data.max(1)[1]

        sjoint_idxs, sseq_idxs, sattn_values = find_idx_topk_attention(attn_org1[0], 150, 5)
        lerp_data = lerp_input_repr(data[0],sseq_idxs,sjoint_idxs,30)
        attentions_joint_lerp = model(lerp_data.unsqueeze(0).to(device), valid_length.to(device), return_attention=True)
        output2 = model(lerp_data.unsqueeze(0).to(device), valid_length.to(device))
        pred_top5 = output2.data.max(1)[1]

        sjoint_idxs, sseq_idxs, sattn_values = find_idx_topk_attention(attn_org1[0], 150, 60)
        lerp_data = lerp_input_repr(data[0],sseq_idxs,sjoint_idxs,30)
        attentions_joint_lerp = model(lerp_data.unsqueeze(0).to(device), valid_length.to(device), return_attention=True)
        output2 = model(lerp_data.unsqueeze(0).to(device), valid_length.to(device))
        pred_top6 = output2.data.max(1)[1]

        sjoint_idxs, sseq_idxs, sattn_values = find_idx_topk_attention(attn_org1[0], 150, 80)
        lerp_data = lerp_input_repr(data[0],sseq_idxs,sjoint_idxs,30)
        attentions_joint_lerp = model(lerp_data.unsqueeze(0).to(device), valid_length.to(device), return_attention=True)
        output2 = model(lerp_data.unsqueeze(0).to(device), valid_length.to(device))
        pred_top7 = output2.data.max(1)[1]


        # attn_org2, attn_inp2 = make_attention(attentions_ qjoint_lerp)
        # grayscale_cam = np.transpose(attn_inp2[0])        
        
        # plot_pre_heatmap(grayscale_cam,selected_joint_names, is_save=True, savepath=os.path.join(save_dir,save_filename+'_'+label_list[pred_top3]+'_top3.png'))
        
        correct += pred_origin.eq(labels).sum()
        correct1 += pred_top1.eq(labels).sum()
        correct2 += pred_top2.eq(labels).sum()
        correct3 += pred_top3.eq(labels).sum()       
        correct4 += pred_top4.eq(labels).sum()       
        correct5 += pred_top5.eq(labels).sum()       
        correct6 += pred_top6.eq(labels).sum()        
        correct7 += pred_top7.eq(labels).sum()       
        idxxx +=1
        # else : 
        #     idxxx +=1
        #     continue
        
# print('correct_pred_top3',correct_pred_top3)
# print('correct_pred_top2',correct_pred_top2)
# print('correct_pred_top1',correct_pred_top1)
# print('correct',correct)

print('correct',correct/119)
print('correct',correct1/119)
print('correct',correct2/119)
print('correct',correct3/119)
print('correct',correct4/119)
print('correct',correct5/119)
print('correct',correct6/119)
print('correct',correct7/119)


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

from model.MoT import MoT_patch2
from model.SImMIM import SimMIM
from mpl_toolkits import mplot3d

# 

def make_attention(model_output):
    # we keep only the output patch attention
    threshold = None
    nh = model_output.shape[1]
    attentions_joint = model_output[0, :, 0,1:].reshape(nh, -1)
    w_featmap_joint = 4
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
    ax.set_yticks(list(range(140)))
    ax.set_yticklabels(jointnames)
    if is_save : 
        plt.savefig(savepath,facecolor=fig.get_facecolor(), transparent=False)
        plt.close()
    else : 
        plt.show()

def find_idx_topk_attention(data, seq_len=80, select_k=3):
    expand_seq = seq_len/data.shape[0]
    sorted_data = np.sort(data.cpu().numpy(), axis=0)[0,:]
    top_k_value = np.sort(sorted_data.reshape(-1))[:select_k]
    seq_list = []
    joint_list = []
    for i in top_k_value:
        idxs = np.where(i==sorted_data)
        joint_list.append(idxs[0][0])
        seq_list.append(0)
    return joint_list, seq_list, top_k_value

def lerp_input_repr(input, target_seqs, target_joints, seq_len):
    output = input.clone()
    mask_start_frame = 0
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
        output[:,:, joint_idx:joint_idx+1] = torch.zeros([1,80,1])
    return output


patch_size = 8
pretrained_weights = ''
image_path = None
image_size = (480,480)
output_dir = 'SIM_MIM_MOT_patch2_512_cls8'
threshold = None
project = 'runs/train'
weight = '500'

exp_name='SIM_MIM_MOT_patch2_512_cls8'
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
weight = '350'
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
gp = emotion_dataset.data['global_p']

mean_p = torch.Tensor(np.asarray(gp).mean(axis=(0,1,2))).to(device)
std_p = torch.Tensor(np.std(np.asarray(gp),axis=(0,1,2))).to(device)

n_classes = 7
seq_len=80
num_joints=35
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

model = MoT_patch2(seq_len = seq_len, num_joints=num_joints, sub_seq=20, num_classes=n_classes,  dim=mot_dim, depth=mot_depth, heads=mot_heads, mlp_dim = mot_mlp_dim)
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
data = data.unsqueeze(1)
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
save_dir = './result_attention_q_patch2_val_zero_pad_512'
pbar = tqdm(emotion_data_loader, position=1, desc="Batch")
correct = 0
correct1 = 0
correct2 = 0
correct3 = 0
correct4 = 0
correct5 = 0
correct6 = 0
correct7 = 0
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
        local_q = batch["local_q"].to(device)
        global_p = batch["global_p"].to(device)    
        labels = batch["labels"].to(device)
        batch_size = local_q.shape[0]
        # jointdist = torch.zeros([batch_size,seq_len,num_pair]).to(device)
        # for idx, i in enumerate(comblist):
        #     jointdist[:,:,idx] = (torch.sqrt(torch.sum((torch.square(((global_p-mean_p)/std_p)[:,:,i[0],:] - ((global_p-mean_p)/std_p)[:,:,i[1],:])),dim=2)))
        # data = torch.cat([local_q,jointdist],dim=2)
        data = local_q
        data = data.unsqueeze(1)
        filename = batch['filename']
        save_filename = filename[0][:-4]
        # data = local_q
        # data = data.unsqueeze(1)
        attentions_joint = model(data.to(device), return_attention=True)
        nh = attentions_joint.shape[1] # number of head
        logits_joint = model(data.to(device))
        output = logits_joint
        pred_origin = output.data.max(1)[1]
        attn_org1, attn_inp1 = make_attention(attentions_joint)
        # grayscale_cam = np.transpose(attn_inp1[0])   
        
        # plot_pre_heatmap(grayscale_cam,selected_joint_names, is_save=True, savepath=os.path.join(save_dir,save_filename+'_'+label_list[pred_origin]+'_origin.png'))
        sjoint_idxs, sseq_idxs, sattn_values = find_idx_topk_attention(attn_org1[0], 80, 5)
        lerp_data = lerp_input_repr(data[0],sseq_idxs,sjoint_idxs,80)
        attentions_joint_lerp = model(lerp_data.unsqueeze(0).to(device), return_attention=True)
        output2 = model(lerp_data.unsqueeze(0).to(device))
        pred_top1 = output2.data.max(1)[1]
        # attn_org2, attn_inp2 = make_attention(attentions_joint_lerp)
        # grayscale_cam = np.transpose(attn_inp2[0])        
        
        # plot_pre_heatmap(grayscale_cam,selected_joint_names, is_save=True, savepath=os.path.join(save_dir,save_filename+'_'+label_list[pred_top3]+'_top3.png'))
        
        sjoint_idxs, sseq_idxs, sattn_values = find_idx_topk_attention(attn_org1[0], 80, 10)
        lerp_data = lerp_input_repr(data[0],sseq_idxs,sjoint_idxs,80)
        attentions_joint_lerp = model(lerp_data.unsqueeze(0).to(device), return_attention=True)
        output3 = model(lerp_data.unsqueeze(0).to(device))
        pred_top2 = output3.data.max(1)[1]
        # attn_org2, attn_inp2 = make_attention(attentions_joint_lerp)
        # grayscale_cam = np.transpose(attn_inp2[0])        
        
        # plot_pre_heatmap(grayscale_cam,selected_joint_names, is_save=True, savepath=os.path.join(save_dir,save_filename+'_'+label_list[pred_top2]+'_top2.png'))
        sjoint_idxs, sseq_idxs, sattn_values = find_idx_topk_attention(attn_org1[0], 80, 15)
        lerp_data = lerp_input_repr(data[0],sseq_idxs,sjoint_idxs,80)
        attentions_joint_lerp = model(lerp_data.unsqueeze(0).to(device), return_attention=True)
        output4 = model(lerp_data.unsqueeze(0).to(device))
        pred_top3 = output4.data.max(1)[1]
        # attn_org2, attn_inp2 = make_attention(attentions_joint_lerp)
        # grayscale_cam = np.transpose(attn_inp2[0])        

        sjoint_idxs, sseq_idxs, sattn_values = find_idx_topk_attention(attn_org1[0], 80, 20)
        lerp_data = lerp_input_repr(data[0],sseq_idxs,sjoint_idxs, 80)
        attentions_joint_lerp = model(lerp_data.unsqueeze(0).to(device), return_attention=True)
        output4 = model(lerp_data.unsqueeze(0).to(device))
        pred_top4 = output4.data.max(1)[1]
        # attn_org2, attn_inp2 = make_attention(attentions_joint_lerp)
        # grayscale_cam = np.transpose(attn_inp2[0])        

        sjoint_idxs, sseq_idxs, sattn_values = find_idx_topk_attention(attn_org1[0], 80, 25)
        lerp_data = lerp_input_repr(data[0],sseq_idxs,sjoint_idxs, 80)
        attentions_joint_lerp = model(lerp_data.unsqueeze(0).to(device), return_attention=True)
        output5 = model(lerp_data.unsqueeze(0).to(device))
        pred_top5 = output5.data.max(1)[1]
        # attn_org2, attn_inp2 = make_attention(attentions_joint_lerp)
        # grayscale_cam = np.transpose(attn_inp2[0])        

        sjoint_idxs, sseq_idxs, sattn_values = find_idx_topk_attention(attn_org1[0], 80, 30)
        lerp_data = lerp_input_repr(data[0],sseq_idxs,sjoint_idxs, 80)
        attentions_joint_lerp = model(lerp_data.unsqueeze(0).to(device), return_attention=True)
        output6 = model(lerp_data.unsqueeze(0).to(device))
        pred_top6= output6.data.max(1)[1]
        # attn_org2, attn_inp2 = make_attention(attentions_joint_lerp)
        # grayscale_cam = np.transpose(attn_inp2[0])        

        sjoint_idxs, sseq_idxs, sattn_values = find_idx_topk_attention(attn_org1[0], 80, 35)
        lerp_data = lerp_input_repr(data[0],sseq_idxs,sjoint_idxs, 80)
        attentions_joint_lerp = model(lerp_data.unsqueeze(0).to(device), return_attention=True)
        output7 = model(lerp_data.unsqueeze(0).to(device))
        pred_top7 = output7.data.max(1)[1]
        # attn_org2, attn_inp2 = make_attention(attentions_joint_lerp)
        # grayscale_cam = np.transpose(attn_inp2[0])        
        # plot_pre_heatmap(grayscale_cam,selected_joint_names, is_save=True, savepath=os.path.join(save_dir,save_filename+'_'+label_list[pred_top1]+'_top1.png'))
                        
        correct += pred_origin.eq(labels).sum()
        correct1 += pred_top1.eq(labels).sum()
        correct2 += pred_top2.eq(labels).sum()
        correct3 += pred_top3.eq(labels).sum()        
        correct4 += pred_top4.eq(labels).sum()        
        correct5 += pred_top5.eq(labels).sum()        
        correct6 += pred_top6.eq(labels).sum()        
        correct7 += pred_top7.eq(labels).sum()        
        
        # correct_pred_top7.append(pred_origin.eq(pred_top7).cpu().numpy()[0])
        # correct_pred_top6.append(pred_origin.eq(pred_top6).cpu().numpy()[0])
        # correct_pred_top5.append(pred_origin.eq(pred_top5).cpu().numpy()[0])
        # correct_pred_top4.append(pred_origin.eq(pred_top4).cpu().numpy()[0])
        # correct_pred_top3.append(pred_origin.eq(pred_top3).cpu().numpy()[0])
        # correct_pred_top2.append(pred_origin.eq(pred_top2).cpu().numpy()[0])
        # correct_pred_top1.append(pred_origin.eq(pred_top1).cpu().numpy()[0])
        idxxx +=1
        # else : 
        #     idxxx +=1
        #     continue
        
# print('correct_pred_top3',correct_pred_top3)
# print('correct_pred_top2',correct_pred_top2)
# print('correct_pred_top1',correct_pred_top1)
# print('correct',correct)

# print('correct_pred_top3',2512-sum(correct_pred_top3))
# print('correct_pred_top2', 2512-sum(correct_pred_top2))
# print('correct_pred_top1', 2512-sum(correct_pred_top1))
print('correct',correct/2512)
print('correct',correct1/2512)
print('correct',correct2/2512)
print('correct',correct3/2512)
print('correct',correct4/2512)
print('correct',correct5/2512)
print('correct',correct6/2512)
print('correct',correct7/2512)



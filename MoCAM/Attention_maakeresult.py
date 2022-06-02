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

from data_proc.emotionmocap_dataset import EmotionDataset
from data_proc.utils import increment_path
import torch.nn.functional as F

from model.MoT import MoT
from model.SImMIM import SimMIM
from mpl_toolkits import mplot3d

# 


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask,(10,10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    print(f"{fname} saved.")
    return

patch_size = 8
pretrained_weights = ''
image_path = None
image_size = (480,480)
output_dir = 'SIM_MIM_MOT_jointtime_cls5_900'
threshold = None
project = 'runs/train'
weight = 'latest'

exp_name='SIM_MIM_MOT_jointtime_cls5'
data_path=''
window=80
batch_size=128

processed_data_dir='processed_data_mocam_80_All_Class_addR'
save_path='runs/test'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# build model

save_dir = Path(os.path.join('runs', 'train', exp_name))
wdir = save_dir / 'weights'
weights = os.listdir(wdir)

if weight == 'latest':
    weights_paths = [wdir / weight for weight in weights]
    print(weights_paths)
    weight_path = max(weights_paths , key = os.path.getctime)
else:
    weight_path = wdir / ('train-' + weight + '.pt')
ckpt = torch.load(weight_path, map_location=device)
print(f"Loaded weight: {weight_path}")
# Load LAFAN Dataset
Path(processed_data_dir).mkdir(parents=True, exist_ok=True)
emotion_dataset = EmotionDataset(data_dir=data_path, processed_data_dir=processed_data_dir, train=True, device=device, window=window)
emotion_data_loader = DataLoader(emotion_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
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
mot_time = MoT(seq_len = num_joints, num_joints=seq_len, num_classes=n_classes,  dim=mot_dim, depth=mot_depth, heads=mot_heads, mlp_dim = mot_mlp_dim)
model_time = SimMIM( encoder = mot_time, masking_ratio = 0.5)
model_time.load_state_dict(ckpt['SimMIM_time'])
mot_joint = MoT(seq_len = seq_len, num_joints=num_joints, num_classes=n_classes,  dim=mot_dim, depth=mot_depth, heads=mot_heads, mlp_dim = mot_mlp_dim)
model_joint = SimMIM( encoder = mot_joint, masking_ratio = 0.5)
model_joint.load_state_dict(ckpt['SimMIM_joint'])
# if target_joint == 'joint':
#     model = model_joint
# else : 
#     model = model_time
for p in model_joint.parameters():
    p.requires_grad = False
for p in model_time.parameters():
    p.requires_grad = False        
model_joint.eval()
model_joint.to(device)
model_time.eval()
model_time.to(device)
pbar = tqdm(emotion_data_loader, position=1, desc="Batch")

origin_data = iter(emotion_data_loader).next()
local_q = origin_data["local_q"].to(device)
q_vel = origin_data["q_vel"].to(device) 
q_acc = origin_data["q_acc"].to(device) 
labels = origin_data["labels"].to(device)     
data = local_q




attentions_joint = model_joint(data.to(device), return_attention=True)
data = local_q.permute(0,2,1)
attentions_time = model_time(data.to(device), return_attention=True)
print(attentions_joint.shape)
nh = attentions_joint.shape[1] # number of head
data = local_q.permute(0,1,2)
logits_joint = model_joint(data.to(device))
data_time = data.permute(0,2,1)
logits_time = model_time(data_time.to(device))
output = (logits_joint+logits_time)/2
# we keep only the output patch attention
attentions_joint = attentions_joint[0, :, :].reshape(nh, -1)
attentions_time = attentions_time[0, :, :].reshape(nh, -1)
w_featmap_joint = 35
h_featmap_joint = 35
w_featmap_time = 80
h_featmap_time = 80 
if threshold is not None:
    # we keep only a certain percentage of the mass
    val, idx = torch.sort(attentions_joint)
    val /= torch.sum(val, dim=1, keepdim=True)
    cumval = torch.cumsum(val, dim=1)
    th_attn = cumval > (1 - threshold)
    idx2 = torch.argsort(idx)
    for head in range(nh):
        th_attn[head] = th_attn[head][idx2[head]]
    th_attn = th_attn.reshape(nh, w_featmap_joint, h_featmap_joint).float()
    # interpolate
#     th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()
print(attentions_joint.shape,'attentions shape')
attentions_joint = attentions_joint.reshape(nh, w_featmap_joint, h_featmap_joint)
# attentions_joint = nn.functional.interpolate(attentions_joint.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()
attentions_time = attentions_time.reshape(nh, w_featmap_time, h_featmap_time)
# attentions_time = nn.functional.interpolate(attentions_time.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()
print(attentions_joint.shape)
# save attentions heatmaps
os.makedirs(output_dir, exist_ok=True)
print(data.shape)
# torchvision.utils.save_image(torchvision.utils.make_grid(data, normalize=True, scale_each=True), os.path.join(args.output_dir, "img.png"))
# for j in range(nh):
#     fname = os.path.join(output_dir, "joint-jt-attn-head" + str(j) + ".png")
#     plt.imsave(fname=fname, arr=attentions_joint[j], format='png')
#     fname = os.path.join(output_dir, "time-jt-attn-head" + str(j) + ".png")
#     plt.imsave(fname=fname, arr=attentions_time[j], format='png')
#     print(f"{fname} saved.")

# if threshold is not None:
#     image = skimage.io.imread(os.path.join(output_dir, "img.png"))
#     for j in range(nh):
#             display_instances(image, th_attn[j], fname=os.path.join(args.output_dir, "mask_th" + str(args.threshold) + "_head" + str(j) +".png"), blur=False)
joint_names = ['world'
,'base'
,'root1'
,'root2'
,'root3'
,'spine'
,'neck'
,'rs1'
,'rs2'
,'rs3'
,'re1'
,'re2'
,'rw1'
,'rw2'
,'rw3'
,'rh'
,'ls1'
,'ls2'
,'ls3'
,'le1'
,'le2'
,'lw1'
,'lw2'
,'lw3'
,'lh'
,'rp1'
,'rp2'
,'rp3'
,'rk'
,'ra1'
,'ra2'
,'ra3'
,'rf'
,'lp1'
,'lp2'
,'lp3'
,'lk'
,'la1'
,'la2'
,'la3'
,'lf'
,'head1'
,'head2'
,'head3']

selected_joint_names = []
selected_joint = [2,3,4,5,7,8,9,11,12,13,14,16,17,18,20,21,22,23,25,26,27,28,29,30,31,33,34,35,36,37,38,39,41,42,43]
for i in selected_joint:
    selected_joint_names.append(joint_names[i])

import pathlib
def plot_single_pose_heatmap(
    pose,
    frame_idx,
    heatmap,
    min_heat,
    max_heat,
    save_dir,
    prefix,
):
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(10,10)  #각 변경
#     selected_joint = [2,3,4,5,7,8,9,11,12,13,14,16,17,18,20,21,22,23,25,26,27,28,29,30,31,33,34,35,36,37,38,39,41,42,43]
#     pose = pose[selected_joint]
    cmap = plt.cm.get_cmap('viridis', 100)
    heatmap = (heatmap - min_heat)/(max_heat - min_heat)

    for i, p in enumerate([-1,-1,-1,0,3,3,3,4,7,7,7,3,3,3,11,14,14,14,0,0,0,18,21,21,21,0,0,0,25,28,28,28,3,3,3]):
        #,8,15,22,30
        #,15,24,32,40
        if heatmap[i] < 0.33 : 
            ccc = 0
        elif heatmap[i] >= 0.33 and heatmap[i] < 0.66 :
            ccc = 50
        else : 
            ccc = 100
        if i > 2 :
            
            sp = ax.plot(
                [pose[i, 0], pose[p, 0]],
                [pose[i, 1], pose[p, 1]],
                [pose[i, 2], pose[p, 2]],
                dash_capstyle='round', linewidth=50, c=(cmap(ccc)[0],cmap(ccc)[1]) + (cmap(ccc)[2], heatmap[i]),
            )
            ax.plot(
                [pose[i, 0], pose[p, 0]],
                [pose[i, 1], pose[p, 1]],
                [pose[i, 2], pose[p, 2]],
                c='k',
            )
        sc = ax.scatter(                
            pose[i, 0],
            pose[i, 1],
            pose[i, 2], 
            color=cmap(ccc),s=400)
        print('b')
# c=(heatmap-min_heat)/(max_heat-min_heat), cmap='viridis'
    x_min = -0.5
    x_max = 0.5
    y_min = -0.5
    y_max = 0.5
    
    z_min = 0
    z_max = 1.5
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("$X$ Axis")

    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("$Y$ Axis")

    ax.set_zlim(z_min, z_max)
    ax.set_zlabel("$Z$ Axis")
#     plt.show()
    plt.draw()
    print('c')

    title = f"{prefix}: {frame_idx}"
    plt.title(title)
    prefix = prefix
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(save_dir, prefix + str(frame_idx) + ".png"), dpi=60)
    plt.close()
def plot_heatmap(data, savepath, jointnames):
    fig = plt.figure(figsize=(15,15),facecolor='white')

    ax = fig.add_subplot(111)
    heatmap = ax.pcolor(data)
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('Joint Number', fontsize=14)
    ax.set_yticks(list(range(35)))
    ax.set_yticklabels(jointnames)
#     ax3 = fig.add_subplot(1,1,1)
#     fig.colorbar(heatmap)
    plt.savefig(savepath,facecolor=fig.get_facecolor(), transparent=False)
#     plt.show()
    plt.close()
def plot_heatmap_joint(data, savepath, jointnames):
    fig = plt.figure(figsize=(15,15),facecolor='white')

    ax = fig.add_subplot(111)
    heatmap = ax.pcolor(data)
    ax.set_xlabel('Joint Number', fontsize=14)
    ax.set_ylabel('Joint Number', fontsize=14)
    ax.set_yticks(list(range(35)))
    ax.set_yticklabels(jointnames)
    ax.set_xticks(list(range(35)))
    ax.set_xticklabels(jointnames)    
    ax3 = fig.add_subplot(1,1,1)
    fig.colorbar(heatmap)
    plt.savefig(savepath,facecolor=fig.get_facecolor(), transparent=False)
#     plt.show()
    plt.close()    
def plot_heatmap_time(data, savepath, jointnames):
    fig = plt.figure(figsize=(15,15),facecolor='white')

    ax = fig.add_subplot(111)
    heatmap = ax.pcolor(data)
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('Joint Number', fontsize=14)
    ax.set_yticks(list(range(35)))
    ax.set_yticklabels(jointnames)
    ax3 = fig.add_subplot(1,1,1)
    fig.colorbar(heatmap)
    plt.savefig(savepath,facecolor=fig.get_facecolor(), transparent=False)
#     plt.show()
    plt.close()    
print('write start')
print(origin_data.keys())
from PIL import Image
import imageio
# emotions = ['angry','happy']
emotions = ['angry','disgust','fearful','happy','neutral','sad','surprise']
prev_file = ''
selected_joint_names = []
selected_joint = [2,3,4,5,7,8,9,11,12,13,14,16,17,18,20,21,22,23,25,26,27,28,29,30,31,33,34,35,36,37,38,39,41,42,43]
for i in selected_joint:
    selected_joint_names.append(joint_names[i])
prefixx = 1
jo = [0,1]
ti = [0,1]
for jo in [0,1]:
    for ti in [0,1]:
        i=0
        save_path = os.path.join('result_2classAH_3c_val_heatmap'+str(prefixx)+str(jo)+str(ti), origin_data['filename'][i].split('.mat')[0])
        # if save_path == prev_file :
        #     continue
        Path(save_path).mkdir(parents=True, exist_ok=True)
        img_aggr_list = []
        grayscale_cam = torch.zeros([35,80])
        j = 0
        for ll in torch.mean(attentions_joint[0], axis=jo):
            grayscale_cam[j] = ((ll* torch.mean(attentions_time[0], axis=ti)))
            j +=1
        heatmap_img_path = os.path.join(save_path,'heatmap.png')
        heat_img = plot_heatmap(grayscale_cam, heatmap_img_path,selected_joint_names)
        heatmap_img = Image.open(heatmap_img_path, 'r')
        heat_img = heatmap_img.resize((900,900))
        for t in range(80):
            input_img_path = os.path.join(save_path, 'tmp')
            plot_single_pose_heatmap(origin_data['global_p'][i,t],t,grayscale_cam[:,t].numpy(),np.min(grayscale_cam.numpy()),np.max(grayscale_cam.numpy()),input_img_path, 'input')
            input_img = Image.open(os.path.join(input_img_path, 'input'+str(t)+'.png'), 'r')
            img_aggr_list.append(np.concatenate([input_img, heat_img], 1))
        # Save images

        prdstr = str(emotions[labels[i].cpu().numpy()])
        gif_path = os.path.join('result_2classAH_3c_val_heatmap'+str(prefixx)+str(jo)+str(ti), str(origin_data['filename'][i].split('.mat')[0])+'_'+str(origin_data['start_frame'][i].numpy())+'_'+prdstr+'.gif')
        imageio.mimsave(gif_path, img_aggr_list, duration=0.1)
        print(f"ID {origin_data['filename'][i].split('.mat')[0]}: test completed.")



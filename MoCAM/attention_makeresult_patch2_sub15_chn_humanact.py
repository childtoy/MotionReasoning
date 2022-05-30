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

from model.MoT import MoT_patch2_seg_chn
from model.SImMIM import SimMIM
from mpl_toolkits import mplot3d
import sys
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
def plot_single_pose_heatmap(
    pose,
    bm, 
    frame_idx,
    minx,
    maxx,
    miny,
    maxy,
    minz,
    maxz,
    heatmap,
    min_heat,
    max_heat,
    save_dir,
    prefix,
    prefixxx,
):
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(10,75)  #각 변경
#     selected_joint = [2,3,4,5,7,8,9,11,12,13,14,16,17,18,20,21,22,23,25,26,27,28,29,30,31,33,34,35,36,37,38,39,41,42,43]
#     pose = pose[selected_joint]
    cmap = plt.cm.get_cmap('viridis', 100)
    # heatmap = (heatmap - min_heat)/(max_heat - min_heat)
    for i, p in enumerate(bm.kintree_table[0][:22]):
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
                [pose[i, 2], pose[p, 2]],
                [pose[i, 1], pose[p, 1]],
                dash_capstyle='round', linewidth=50, c=(cmap(ccc)[0],cmap(ccc)[1]) + (cmap(ccc)[2], heatmap[i]),
            )
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
            color=cmap(ccc),s=400)

# c=(heatmap-min_heat)/(max_heat-min_heat), cmap='viridis'
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
#     plt.show()
    plt.draw()

    title = f"{prefixxx}: {frame_idx}"
    plt.title(title)
    prefix = prefix
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(save_dir, prefix + str(frame_idx) + ".png"), dpi=60)
    plt.close()
def plot_heatmap(data, savepath, jointnames):
    fig = plt.figure(figsize=(15,15),facecolor='white')

    ax = fig.add_subplot(111)
    heatmap = ax.pcolor(data, vmin=0, vmax=1)
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('Joint Number', fontsize=14)
    ax.set_yticks(list(range(22)))
    ax.set_yticklabels(jointnames)
#     ax3 = fig.add_subplot(1,1,1)
    fig.colorbar(heatmap)
    plt.savefig(savepath,facecolor=fig.get_facecolor(), transparent=False)
#     plt.show()
    plt.close()

patch_size = 8
pretrained_weights = ''
image_path = None
image_size = (480,480)
output_dir = 'humanact12_SIM_MIM_MOT_patch2_512_cls_chn_head2'
threshold = None
project = 'runs/train'
weight = '1000'

exp_name='humanact12_SIM_MIM_MOT_patch2_512_cls_chn_head2'
data_path=''
window=150
batch_size=128
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
mot_dim = 256
mot_depth = 6
mot_heads = 1
mot_mlp_dim = 512
mot_pool = 'cls'
mot_channels =1, 
mot_dim_head = 64
target_joint = 'joint'
model = MoT_patch2_seg_chn(seq_len = seq_len, num_joints=num_joints, sub_seq=15, num_classes=n_classes,  dim=mot_dim, depth=mot_depth, heads=mot_heads, mlp_dim = mot_mlp_dim)
# mim = SimMIM_patch( encoder = mot, masking_ratio = 0.5)
model.load_state_dict(ckpt['MoT'])
for p in model.parameters():
    p.requires_grad = False
model.eval()
model.to(device)
pbar = tqdm(test_data_loader, position=1, desc="Batch")
origin_data = iter(test_data_loader).next()
motions = origin_data["rotation_6d_pose_list"].to(device)
motions_ = motions.clone()
valid_length = origin_data["valid_length_list"].to(device) 
labels = origin_data["labels"]
labels = le.transform(labels)
labels = torch.Tensor(labels)
labels = labels.long().to(device)
B,T,J,C = motions.shape
motions = motions.permute(0,3,1,2)
# motions = motions.unsqueeze(1)
attentions_joints = model(motions.to(device), valid_length.to(device), return_attention=True)
# data = local_q.permute(0,2,1)
# attentions_time = model_time(data.to(device), return_attention=True)
# print(attentions_joint.shape)
nh = attentions_joints.shape[1] # number of head
# data = local_q.permute(0,1,2)
logits_joint = model(motions.to(device), valid_length.to(device))
# data_time = data.permute(0,2,1)
# logits_time = model_time(data_time.to(device))
# output = (logits_joint+logits_time)/2
# we keep only the output patch attention
output = logits_joint
pred_ = output.data.max(1)[1]

subject_gender = 'neutral'
bm_fname = os.path.join('smpl_model', subject_gender, 'model.npz')
dmpl_fname = os.path.join('dmpl_model', subject_gender, 'model.npz')
num_betas = 10
num_dmpls = 8
bm = BodyModel(bm_fname=bm_fname, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname).to(device)
faces = c2c(bm.f)
for exam_idx in range(119):
    attentions_joint = attentions_joints[exam_idx, :, 0,1:].reshape(nh, -1)
    w_featmap_joint = 10
    h_featmap_joint = 22
    attentions_joint = attentions_joint.reshape(nh, w_featmap_joint, h_featmap_joint)
    attentions_joint = nn.functional.interpolate(attentions_joint.unsqueeze(0), scale_factor=(15,1), mode="nearest")[0].cpu().numpy()

    motion = motions_[exam_idx]
    smpl_param = matrix_to_axis_angle(rotation_6d_to_matrix(motion[:valid_length[exam_idx]])).reshape(valid_length[exam_idx], -1)
    root_orient = smpl_param[:, :3]
    pose_body = smpl_param[:, 3:66]
    # pose_hand = smpl_param[:, 66:]
    body_params = {
    'root_orient': root_orient, # controls the global root orientation
    'pose_body': pose_body, # controls the body
    # 'pose_hand': torch.Tensor(pose_hand).to(device), # controls the finger articulation
    }
    body_pose_beta = bm(**{k:v for k,v in body_params.items() if k in ['pose_body', 'betas']})
    # save attentions heatmaps
    os.makedirs(output_dir, exist_ok=True)
    
    print('write start')
    from PIL import Image
    import imageio
    labels_name = full_proc_label_list
    prev_file = ''
    selected_joint_names = []
    selected_joint = list(range(22))
    for i in selected_joint:
        selected_joint_names.append(joint_names[i])
    prefixx = str(exam_idx)+'GT:'+ le.inverse_transform([labels[exam_idx].cpu().numpy()])[0] + '   Pred:'+ le.inverse_transform([pred_[exam_idx].cpu().numpy()])[0] + ' Seq_Len:' +str(valid_length[exam_idx].cpu().numpy())
    i=0
    save_path = os.path.join('humanact12_patch_sub15_head1_chn_val')
    # if save_path == prev_file :
    #     continue
    Path(save_path).mkdir(parents=True, exist_ok=True)
    input_img_path = os.path.join(save_path, 'tmp')
    # Path(os.path.join(input_img_path, 'tmp')).mkdir(parents=True, exist_ok=True)

    img_aggr_list = []
    # grayscale_cam = torch.zeros([22,80])
    grayscale_cam = attentions_joint[0]
    # grayscale_cam = grayscale_cam.transpose(1,0)
    grayscale_cam = np.transpose(grayscale_cam)
    heatmap_img_path = os.path.join(save_path,'heatmap.png')
    heat_img = plot_heatmap(grayscale_cam, heatmap_img_path,selected_joint_names)
    heatmap_img = Image.open(heatmap_img_path, 'r')
    heat_img = heatmap_img.resize((900,900))
    global_pose = body_pose_beta.Jtr[:,:22,:]
    minx = np.min(global_pose[:,:,0].cpu().numpy())
    maxx = np.max(global_pose[:,:,0].cpu().numpy())
    miny = np.min(global_pose[:,:,2].cpu().numpy())
    maxy = np.max(global_pose[:,:,2].cpu().numpy())
    minz = np.min(global_pose[:,:,1].cpu().numpy())
    maxz = np.max(global_pose[:,:,1].cpu().numpy())
    for t in range(global_pose.shape[0]):
        plot_single_pose_heatmap(global_pose[t].cpu().numpy(),bm, t, minx,maxx,miny,maxy,minz,maxz, grayscale_cam[:,t],np.min(grayscale_cam),np.max(grayscale_cam),input_img_path, 'input', prefixx)
        input_img = Image.open(os.path.join(input_img_path, 'input'+str(t)+'.png'), 'r')
        img_aggr_list.append(np.concatenate([input_img, heat_img], 1))
    # Save images

    prdstr = str(labels_name[labels[i].cpu().numpy()])
    gif_path = os.path.join('humanact12_patch_sub15_head1_chn_val','humanact12_patch_sub15_head1_chn_val'+str(prefixx)+'.gif')
    imageio.mimsave(gif_path, img_aggr_list, duration=0.1)
    print(f"ID {exam_idx}: test completed.")


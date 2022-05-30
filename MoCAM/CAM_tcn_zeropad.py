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

from mpl_toolkits import mplot3d

# 

import numpy as np
import torch
import ttach as tta
from typing import Callable, List, Tuple
from CAM.activations_and_gradients import ActivationsAndGradients
from CAM.utils.svd_on_activations import get_2d_projection
from CAM.utils.image import scale_cam_image
from CAM.utils.model_targets import ClassifierOutputTarget

import matplotlib.pyplot as plt

class BaseCAM:
    def __init__(self,
                 model: torch.nn.Module,
                 target_layers: List[torch.nn.Module],
                 use_cuda: bool = False,
                 reshape_transform: Callable = None,
                 compute_input_gradient: bool = False,
                 uses_gradients: bool = True) -> None:
        self.model = model.eval()
        self.target_layers = target_layers
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.reshape_transform = reshape_transform
        self.compute_input_gradient = compute_input_gradient
        self.uses_gradients = uses_gradients
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    def get_cam_weights(self,
                        input_tensor: torch.Tensor,
                        target_layers: List[torch.nn.Module],
                        targets: List[torch.nn.Module],
                        activations: torch.Tensor,
                        grads: torch.Tensor) -> np.ndarray:
        raise Exception("Not Implemented")

    def get_cam_image(self,
                      input_tensor: torch.Tensor,
                      target_layer: torch.nn.Module,
                      targets: List[torch.nn.Module],
                      activations: torch.Tensor,
                      grads: torch.Tensor,
                      eigen_smooth: bool = False) -> np.ndarray:

        weights = self.get_cam_weights(input_tensor,
                                       target_layer,
                                       targets,
                                       activations,
                                       grads)
        weighted_activations = weights[:, :, None, None] * activations
        print('lajfs;djasl;kxzcv.n,zvxnxcvzn,n')
        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)
        return cam

    def forward(self,
                input_tensor: torch.Tensor,
                targets: List[torch.nn.Module],
                eigen_smooth: bool = False) -> np.ndarray:

        if self.cuda:
            input_tensor = input_tensor.cuda()

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor,
                                                   requires_grad=True)

        outputs = self.activations_and_grads(input_tensor)
        if targets is None:
            target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
            targets = [ClassifierOutputTarget(category) for category in target_categories]

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(output) for target, output in zip(targets, outputs)])
            loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor,
                                                   targets,
                                                   eigen_smooth)
        print(len(cam_per_layer))
        return self.aggregate_multi_layers(cam_per_layer)

    def get_target_width_height(self,
                                input_tensor: torch.Tensor) -> Tuple[int, int]:
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def compute_cam_per_layer(
            self,
            input_tensor: torch.Tensor,
            targets: List[torch.nn.Module],
            eigen_smooth: bool) -> np.ndarray:
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam_image(input_tensor,
                                     target_layer,
                                     targets,
                                     layer_activations,
                                     layer_grads,
                                     eigen_smooth)
            cam = np.maximum(cam, 0)
            scaled = scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer: np.ndarray) -> np.ndarray:
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return scale_cam_image(result)

    def forward_augmentation_smoothing(self,
                                       input_tensor: torch.Tensor,
                                       targets: List[torch.nn.Module],
                                       eigen_smooth: bool = False) -> np.ndarray:
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.Multiply(factors=[0.9, 1, 1.1]),
            ]
        )
        cams = []
        for transform in transforms:
            augmented_tensor = transform.augment_image(input_tensor)
            cam = self.forward(augmented_tensor,
                               targets,
                               eigen_smooth)

            # The ttach library expects a tensor of size BxCxHxW
            cam = cam[:, None, :, :]
            cam = torch.from_numpy(cam)
            cam = transform.deaugment_mask(cam)

            # Back to numpy float32, HxW
            cam = cam.numpy()
            cam = cam[:, 0, :, :]
            cams.append(cam)

        cam = np.mean(np.float32(cams), axis=0)
        return cam

    def __call__(self,
                 input_tensor: torch.Tensor,
                 targets: List[torch.nn.Module] = None,
                 aug_smooth: bool = False,
                 eigen_smooth: bool = False) -> np.ndarray:

        # Smooth the CAM result with test time augmentation
        if aug_smooth is True:
            return self.forward_augmentation_smoothing(
                input_tensor, targets, eigen_smooth)
        
        return self.forward(input_tensor,
                            targets, eigen_smooth)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True

class EigenCAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None):
        super(EigenCAM, self).__init__(model, target_layers, use_cuda,
                                       reshape_transform)

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth):
        return get_2d_projection(activations)

class GradCAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None):
        super(
            GradCAM,
            self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        return np.mean(grads, axis=(2, 3))


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
def find_idx_topk_cam(data, seq_len=80, select_k=3):
    expand_seq = seq_len/data.shape[0]
    top_k_value = np.sort(data[:,0])[::-1][:select_k]
    print(top_k_value)
    seq_list = []
    joint_list = []
    for i in top_k_value:
        idxs = np.where(i==data[:,0])
        joint_list.append(idxs[0][0])
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
        output[start_seq:start_seq+seq_len, joint_idx:joint_idx+1] = torch.zeros([1,80,1])
    return output


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
from model.model import TCN, PURE1D
import torch.nn.functional as F
import argparse
import cv2
import numpy as np
import torch
from torchvision import models
from CAM.eigen_cam import EigenCAM

from CAM.guided_backprop import GuidedBackpropReLUModel
from CAM.utils.image import show_cam_on_image, deprocess_image, preprocess_image
from CAM.utils.model_targets import ClassifierOutputTarget


device = torch.device("cpu")
parser = argparse.ArgumentParser()
project='runs/train'
weight='latest'
exp_name='TCN_localq3'
data_path='/home/taehyun/workspace/childtoy/MotionReasoning/dataset/mocap_emotion_rig'
window=80
batch_size=1

processed_data_dir='processed_data_mocam_80_All_Class_addR2/'

save_dir = Path(os.path.join('runs', 'train', exp_name))
wdir = save_dir / 'weights'
weights = os.listdir(wdir)
weight = '250'

if weight == 'latest':
    weights_paths = [wdir / weight for weight in weights]
    weight_path = max(weights_paths , key = os.path.getctime)
else:
    weight_path = wdir / ('train-' + weight + '.pt')
ckpt = torch.load(weight_path, map_location=device)
print(f"Loaded weight: {weight_path}")


# Load LAFAN Dataset
Path(processed_data_dir).mkdir(parents=True, exist_ok=True)
emotion_dataset = EmotionDataset(data_dir=data_path, processed_data_dir=processed_data_dir, train=False, device=device, window=window)
emotion_data_loader = DataLoader(emotion_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
n_classes = ckpt['n_classes']
input_channels = ckpt['input_channels']
n_classes = 7
input_channels = 35
seq_length = 80
n_hid = 128
n_level = 6
channel_sizes = [n_hid] * n_level
kernel_size = 5
model = TCN(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=0)
model.load_state_dict(ckpt['TCN'])
model.eval()
correct = 0
seq_length = 40
n_hid = ckpt['n_hid']
n_level = ckpt['n_level']
n_classes = 7
input_channels = 105
origin_data = iter(emotion_data_loader).next()# confusion_matrix = torch.zeros(7, 7)
local_q = origin_data["local_q"].to(device)
q_vel = origin_data["q_vel"].to(device) 
q_acc = origin_data["q_acc"].to(device) 
labels = origin_data["labels"].to(device)
data = local_q
data = data.permute(0,2,1)
output = model(data)


# 
joint_names = ['world','base','root1','root2','root3','spine','neck','rs1','rs2','rs3','re1','re2','rw1','rw2','rw3','rh','ls1','ls2','ls3'
,'le1','le2','lw1','lw2','lw3','lh','rp1','rp2','rp3','rk','ra1','ra2','ra3','rf','lp1','lp2','lp3','lk','la1','la2','la3','lf','head1','head2','head3']

selected_joint_names = []
selected_joint = [2,3,4,5,7,8,9,11,12,13,14,16,17,18,20,21,22,23,25,26,27,28,29,30,31,33,34,35,36,37,38,39,41,42,43]
for i in selected_joint:
    selected_joint_names.append(joint_names[i])
label_list = ['angry','disgust','fearful','happy','neutral','sad','surprise']
save_dir = './result_attention_patch2_val_lerp_all'
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
cam = EigenCAM(model=model, target_layers = model.tcn.network, use_cuda=False)

# with torch.no_grad():
for batch in pbar:
    local_q = batch["local_q"].to(device)
    labels = batch["labels"].to(device)     
    filename = batch['filename']
    save_filename = filename[0][:-4]
    data = local_q
    data = data.permute(0,2,1)
    output = model(data)
    pred_origin = output.data.max(1)[1]
    print(data[0:].shape)
    grayscale_cam = cam(data[0:], targets=None)        
    if idxxx%10 == 0:
        plot_pre_heatmap(grayscale_cam[0],selected_joint_names, is_save=True, savepath=os.path.join(save_dir,save_filename+'_'+label_list[pred_origin]+'_origin.png'))
    sjoint_idxs, sseq_idxs, sattn_values = find_idx_topk_cam(grayscale_cam[0])
    lerp_data = lerp_input_repr(data[0].transpose(1,0),sseq_idxs,sjoint_idxs,80).transpose(1,0).unsqueeze(0)
    output2 = model(lerp_data.to(device))
    pred_top3 = output2.data.max(1)[1]
    grayscale_cam = cam(lerp_data[0:1], targets=None)


    if idxxx%10 == 0:
        plot_pre_heatmap(grayscale_cam[0],selected_joint_names, is_save=True, savepath=os.path.join(save_dir,save_filename+'_'+label_list[pred_top3]+'_top3.png'))

    sjoint_idxs, sseq_idxs, sattn_values = find_idx_topk_cam(grayscale_cam[0], 80, 2)
    lerp_data = lerp_input_repr(data[0].transpose(1,0),sseq_idxs,sjoint_idxs,80).transpose(1,0).unsqueeze(0)
    output3 = model(lerp_data.to(device))
    pred_top2 = output3.data.max(1)[1]
    grayscale_cam = cam(lerp_data[0:1], targets=None)

    if idxxx%10 == 0:
        plot_pre_heatmap(grayscale_cam[0],selected_joint_names, is_save=True, savepath=os.path.join(save_dir,save_filename+'_'+label_list[pred_top2]+'_top2.png'))
    sjoint_idxs, sseq_idxs, sattn_values = find_idx_topk_cam(grayscale_cam[0], 80, 1)
    lerp_data = lerp_input_repr(data[0].transpose(1,0),sseq_idxs,sjoint_idxs,80).transpose(1,0).unsqueeze(0)
    output4 = model(lerp_data.to(device))
    pred_top1 = output4.data.max(1)[1]
    grayscale_cam = cam(lerp_data[0:1], targets=None)
    if idxxx%10 == 0:
        plot_pre_heatmap(grayscale_cam[0],selected_joint_names, is_save=True, savepath=os.path.join(save_dir,save_filename+'_'+label_list[pred_top1]+'_top1.png'))

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


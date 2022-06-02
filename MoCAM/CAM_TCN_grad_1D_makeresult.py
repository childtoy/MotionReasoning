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
weight='250'
exp_name='TCN_localq3'
data_path='/home/taehyun/workspace/childtoy/MotionReasoning/dataset/mocap_emotion_rig'
window=80
batch_size=300

processed_data_dir='processed_data_mocam_80_All_Class_addR2/'

save_dir = Path(os.path.join('runs', 'train', exp_name))
wdir = save_dir / 'weights'
weights = os.listdir(wdir)

if weight == 'latest':
    weights_paths = [wdir / weight for weight in weights]
    weight_path = max(weights_paths , key = os.path.getctime)
else:
    weight_path = wdir / ('train-' + weight + '.pt')
ckpt = torch.load(weight_path, map_location=device)
print(f"Loaded weight: {weight_path}")


# Load LAFAN Dataset
# Path(processed_data_dir).mkdir(parents=True, exist_ok=True)
# emotion_dataset = EmotionDataset(data_dir=data_path, processed_data_dir=processed_data_dir, train=False, device=device, window=window)
# emotion_data_loader = DataLoader(emotion_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
# Load LAFAN Dataset
Path(processed_data_dir).mkdir(parents=True, exist_ok=True)
emotion_dataset = EmotionDataset(data_dir=data_path, processed_data_dir=processed_data_dir, train=False, device=device, window=window)
emotion_data_loader = DataLoader(emotion_dataset, batch_size=batch_size, shuffle=True, num_workers=0)


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
n_classes = ckpt['n_classes']
input_channels = ckpt['input_channels']
n_hid = ckpt['n_hid']
n_level = ckpt['n_level']
origin_data = iter(emotion_data_loader).next()# confusion_matrix = torch.zeros(7, 7)
local_q = origin_data["local_q"].to(device)
q_vel = origin_data["q_vel"].to(device) 
q_acc = origin_data["q_acc"].to(device) 
labels = origin_data["labels"].to(device)
# data = torch.cat([local_q, q_vel, q_acc], axis=2)
data = local_q
data = data.permute(0,2,1)
output = model(data)

# device = torch.device("cpu")
# parser = argparse.ArgumentParser()
# project='runs/train'
# weight='latest'
# exp_name='emotionmocam_mat_7class3'
# data_path='/home/taehyun/workspace/childtoy/MotionReasoning/dataset/mocap_emotion_rig'
# window=80
# batch_size=300

# processed_data_dir='processed_data_mocam_80_All_Class_global_p/'

# save_dir = Path(os.path.join('runs', 'train', exp_name))
# wdir = save_dir / 'weights'
# weights = os.listdir(wdir)

# if weight == 'latest':
#     weights_paths = [wdir / weight for weight in weights]
#     weight_path = max(weights_paths , key = os.path.getctime)
# else:
#     weight_path = wdir / ('train-' + weight + '.pt')
# ckpt = torch.load(weight_path, map_location=device)
# print(f"Loaded weight: {weight_path}")


# Load LAFAN Dataset
# Path(processed_data_dir).mkdir(parents=True, exist_ok=True)
# emotion_dataset = EmotionDataset(data_dir=data_path, processed_data_dir=processed_data_dir, train=False, device=device, window=window)
# emotion_data_loader = DataLoader(emotion_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
# n_classes = 7
# input_channels = 3
# n_hid = 70
# n_level = 4
# n_channels = 64
# channel_sizes = [n_hid] * n_level
# kernel_size = 5
# model = PURE3D(input_channels, n_channels, n_classes, kernel_size=kernel_size, dropout=0)
# model.load_state_dict(ckpt['TCN'])
# model.eval()
# correct = 0
# n_classes = ckpt['n_classes']
# input_channels = ckpt['input_channels']
# n_hid = ckpt['n_hid']
# n_level = ckpt['n_level']
# origin_data = iter(emotion_data_loader).next()# confusion_matrix = torch.zeros(7, 7)
# local_q = origin_data["local_q"].to(device)
# q_vel = origin_data["q_vel"].to(device) 
# q_acc = origin_data["q_acc"].to(device) 
# labels = origin_data["labels"].to(device)
# data = torch.cat([local_q, q_vel, q_acc], axis=2)
# data = torch.cat([local_q.unsqueeze(3), local_q.unsqueeze(3), local_q.unsqueeze(3)], axis=3)
# data = data.permute(0,3,2,1)
# output = model(data)

# 

import numpy as np
import torch
import ttach as tta
from typing import Callable, List, Tuple
import numpy as np
import torch
import ttach as tta
from typing import Callable, List, Tuple
from CAM.activations_and_gradients import ActivationsAndGradients
from CAM.utils.svd_on_activations import get_2d_projection
from CAM.utils.image import scale_cam_image
from CAM.utils.model_targets import ClassifierOutputTarget
import matplotlib 
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

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
        weighted_activations = weights[:, :, None] * activations
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
#             loss = targets[0](output[0])
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
        print(grads.shape)
        return np.mean(grads, axis=(2))
    
class GradCAMPlusPlus(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None):
        super(GradCAMPlusPlus, self).__init__(model, target_layers, use_cuda,
                                              reshape_transform)

    def get_cam_weights(self,
                        input_tensor,
                        target_layers,
                        target_category,
                        activations,
                        grads):
        grads_power_2 = grads**2
        grads_power_3 = grads_power_2 * grads
        # Equation 19 in https://arxiv.org/abs/1710.11063
        sum_activations = np.sum(activations, axis=(2))
        eps = 0.000001
        aij = grads_power_2 / (2 * grads_power_2 +
                               sum_activations[:, :, None] * grads_power_3 + eps)
        # Now bring back the ReLU from eq.7 in the paper,
        # And zero out aijs where the activations are 0
        aij = np.where(grads != 0, aij, 0)

        weights = np.maximum(grads, 0) * aij
        weights = np.sum(weights, axis=(2))
        return weights
class ClassifierOutputTarget:
    def __init__(self, category):
        self.category = category
    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]    

# grayscale_cam = cam(input_tensor, targets=targets)
# # Take the first image in the batch:
# grayscale_cam = grayscale_cam[0, :]
# cam_image = show_cam_on_image(image_float_np, grayscale_cam, use_rgb=True)
# # And lets draw the boxes again:
# image_with_bounding_boxes = draw_boxes(boxes, labels, classes, cam_image)
# Image.fromarray(image_with_bounding_boxes)
print('CAM Start')
cam = GradCAM(model,model.tcn.network,
               use_cuda=False)
grayscale_cam = cam(data[:15], targets=None)
print('CAM END')
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
    ax.view_init(10,10)   #각 변경
    selected_joint = [2,3,4,5,7,8,9,11,12,13,14,16,17,18,20,21,22,23,25,26,27,28,29,30,31,33,34,35,36,37,38,39,41,42,43]
    pose = pose
    cmap = plt.cm.get_cmap('viridis', 100)
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
    ax3 = fig.add_subplot(1,1,1)
    fig.colorbar(heatmap)
    plt.savefig(savepath,facecolor=fig.get_facecolor(), transparent=False)
#     plt.show()
    plt.close()

print('write start')
from PIL import Image
import imageio
# emotions = ['angry','happy']
emotions = ['angry','disgust','fearful','happy','neutral','sad','surprise']
prev_file = ''
selected_joint_names = []
selected_joint = [2,3,4,5,7,8,9,11,12,13,14,16,17,18,20,21,22,23,25,26,27,28,29,30,31,33,34,35,36,37,38,39,41,42,43]
for i in selected_joint:
    selected_joint_names.append(joint_names[i])
for i in range(15):
    save_path = os.path.join('result_grad_val_heatmap', origin_data['filename'][i].split('.mat')[0])
    if save_path == prev_file :
        continue
    Path(save_path).mkdir(parents=True, exist_ok=True)
    img_aggr_list = []
    
    heatmap_img_path = os.path.join(save_path,'heatmap.png')
    heat_img = plot_heatmap(grayscale_cam[i], heatmap_img_path,selected_joint_names)
    heatmap_img = Image.open(heatmap_img_path, 'r')
    heat_img = heatmap_img.resize((900,900))
    for t in range(80):
        input_img_path = os.path.join(save_path, 'tmp')
        plot_single_pose_heatmap(origin_data['global_p'][i,t],t,grayscale_cam[i,:,t],np.min(grayscale_cam[i]),np.max(grayscale_cam[i]),input_img_path, 'input')
        input_img = Image.open(os.path.join(input_img_path, 'input'+str(t)+'.png'), 'r')
        img_aggr_list.append(np.concatenate([input_img, heat_img], 1))
    # Save images

    prdstr = str(emotions[labels[i].numpy()])+"_pred_"+str(emotions[output.data.max(1)[1][i].numpy()])
    gif_path = os.path.join('result_grad_val_heatmap', str(origin_data['filename'][i].split('.mat')[0])+'_'+prdstr+'.gif')
    imageio.mimsave(gif_path, img_aggr_list, duration=0.1)
    print(f"ID {origin_data['filename'][i].split('.mat')[0]}: test completed.")
    prev_file = save_path
# for j in range(80):
#     plot_single_pose_heatmap(origin_data['global_p'][0,j],i,grayscale_cam[0,:,j],'tmp','input')
import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO
from pathlib import Path

import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from tqdm import tqdm

from data_proc.emotionmocap_dataset import EmotionDataset
from data_proc.utils import increment_path
import torch.nn.functional as F

from model.MoT import MoT
from model.SImMIM import SimMIM

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str,
        help="Path to pretrained weights to load.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--image_path", default=None, type=str, help="Path of the image to load.")
    parser.add_argument("--image_size", default=(480, 480), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output_dir', default='.', help='Path where to save visualizations.')
    parser.add_argument("--threshold", type=float, default=None, help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    parser.add_argument('--project', default='runs/train', help='project/name')
    parser.add_argument('--weight', default='500')
    parser.add_argument('--exp_name', default='SIM_MIM_MOT_jointtime_cls5', help='experiment name')
    parser.add_argument('--data_path', type=str, default='/home/taehyun/workspace/childtoy/MotionReasoning/dataset/mocap_emotion_rig', help='BVH dataset path')

    parser.add_argument('--window', type=int, default=80, help='window')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')

    parser.add_argument('--processed_data_dir', type=str, default='processed_data_mocam_80_All_Class/', help='path to save pickled processed data')
    parser.add_argument('--save_path', type=str, default='runs/test', help='path to save model')    
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # build model

    save_dir = Path(os.path.join('runs', 'train', args.exp_name))
    wdir = save_dir / 'weights'
    weights = os.listdir(wdir)

    if args.weight == 'latest':
        weights_paths = [wdir / weight for weight in weights]
        print(weights_paths)
        weight_path = max(weights_paths , key = os.path.getctime)
    else:
        weight_path = wdir / ('train-' + args.weight + '.pt')
    ckpt = torch.load(weight_path, map_location=device)
    print(f"Loaded weight: {weight_path}")


    # Load LAFAN Dataset
    Path(args.processed_data_dir).mkdir(parents=True, exist_ok=True)
    emotion_dataset = EmotionDataset(data_dir=args.data_path, processed_data_dir=args.processed_data_dir, train=False, device=device, window=args.window)
    emotion_data_loader = DataLoader(emotion_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    n_classes = 7
    seq_len=args.window
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
    if target_joint == 'joint':
        model = model_joint
    else : 
        model = model_time
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)
    pbar = tqdm(emotion_data_loader, position=1, desc="Batch")
    confusion_matrix = torch.zeros(7, 7)
    with torch.no_grad():
        for batch in pbar:
            local_q = batch["local_q"].to(device)
            batch_size = local_q.shape[0]
            q_vel = batch["q_vel"].to(device) 
            q_acc = batch["q_acc"].to(device) 
            # data = local_q.permute(0,2,1)
            data = local_q
            break
    # make the image divisible by the patch size

    attentions = model(data.to(device), return_attention=True)
    print(attentions.shape)
    nh = attentions.shape[1] # number of head

    # we keep only the output patch attention
    attentions = attentions[0, :, :].reshape(nh, -1)
    w_featmap = 35
    h_featmap = 35
    if args.threshold is not None:
        # we keep only a certain percentage of the mass
        val, idx = torch.sort(attentions)
        val /= torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)
        th_attn = cumval > (1 - args.threshold)
        idx2 = torch.argsort(idx)
        for head in range(nh):
            th_attn[head] = th_attn[head][idx2[head]]
        th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
        # interpolate
        th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()
    print(attentions.shape,'attentions shape')
    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()
    print(attentions.shape)
    # save attentions heatmaps
    os.makedirs(args.output_dir, exist_ok=True)
    print(data.shape)
    # torchvision.utils.save_image(torchvision.utils.make_grid(data, normalize=True, scale_each=True), os.path.join(args.output_dir, "img.png"))
    for j in range(nh):
        fname = os.path.join(args.output_dir, target_joint+"-jt-attn-head" + str(j) + ".png")
        plt.imsave(fname=fname, arr=attentions[j], format='png')
        print(f"{fname} saved.")

    if args.threshold is not None:
        image = skimage.io.imread(os.path.join(args.output_dir, "img.png"))
        for j in range(nh):
            display_instances(image, th_attn[j], fname=os.path.join(args.output_dir, "mask_th" + str(args.threshold) + "_head" + str(j) +".png"), blur=False)
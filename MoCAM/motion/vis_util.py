import os

os.environ['PYOPENGL_PLATFORM'] = 'egl'

from pathlib import Path
from typing import List

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from body_visualizer.mesh.mesh_viewer import MeshViewer
from body_visualizer.tools.vis_tools import colors
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from PIL import Image


def visualize_motion(data_sample, sample_id: int, prediction: List, device: str="cuda:0"):
    device = torch.device(device)
    imw, imh = 400, 400
    mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
    
    # Visualization settings
    subject_gender = 'neutral'
    bm_fname = os.path.join('smpl_model', subject_gender, 'model.npz')
    dmpl_fname = os.path.join('dmpl_model', subject_gender, 'model.npz')
    num_betas = 16
    num_dmpls = 8

    bm = BodyModel(bm_fname=bm_fname, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname).to(device)
    faces = c2c(bm.f)

    valid_length = data_sample['valid_length_list']
    pose_details = data_sample['pose_list'][:valid_length]
    trans = data_sample['trans_list'][:valid_length]
    betas = data_sample['betas_list']
    dmpls = data_sample['dmpl_list'][:valid_length]

    body_params = {
        'root_orient': torch.Tensor(pose_details[:, :3]).to(device), # controls the global root orientation
        'pose_body': torch.Tensor(pose_details[:, 3:66]).to(device), # controls the body
        'pose_hand': torch.Tensor(pose_details[:, 66:]).to(device), # controls the finger articulation
        'trans': torch.Tensor(trans).to(device), # controls the global body position
        'betas': torch.Tensor(np.repeat(betas[:num_betas][np.newaxis], repeats=valid_length, axis=0)).to(device), # controls the body shape. Body shape is static
        'dmpls': torch.Tensor(dmpls[:, :num_dmpls]).to(device) # controls soft tissue dynamics
    }

    body_pose_beta = bm(**{k:v for k,v in body_params.items() if k in ['pose_body', 'betas']})

    vis_path = Path('vis_out/')
    vis_image_path = vis_path / f'{sample_id}'
    vis_image_path.mkdir(exist_ok=True, parents=True)

    for fId in range(valid_length):
        gt = data_sample['proc_label_list']
        body_mesh = trimesh.Trimesh(vertices=c2c(body_pose_beta.v[fId]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
        mv.set_static_meshes([body_mesh])
        body_image = mv.render(render_wireframe=False)
        save_image_with_gt_pred(body_image, frame_id=fId, gt=gt, prediction=prediction, target_dir=vis_image_path)

    # Make gif from images
    image_files = [os.path.join(vis_image_path, img) for img in sorted(os.listdir(vis_image_path))]
    images = [Image.open(i) for i in image_files]
    imageio.mimsave(f'{vis_path}/{sample_id}.gif', images, duration=0.1)


def save_image_with_gt_pred(img_ndarray: str, frame_id: int, gt: str, prediction: List, target_dir: str):

    fig = plt.figure(figsize=(4, 4), dpi=300)
    ax = fig.gca()

    img = img_ndarray.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    plt.axis('off')
    title = f'GT: {gt} \n PRED: {prediction}'
    plt.title(title)
    save_fn = os.path.join(target_dir, '{:06d}.png'.format(frame_id))
    plt.savefig(save_fn, bbox_inches='tight', pad_inches=0)
    plt.close()


def save_image(img_ndarray: str, frame_id: int, title: str, target_dir: str):

    fig = plt.figure(figsize=(4, 4), dpi=300)
    ax = fig.gca()

    img = img_ndarray.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    plt.axis('off')
    title = f'{title}: [{frame_id}]'
    plt.title(title)
    save_fn = os.path.join(target_dir, '{:06d}.png'.format(frame_id))
    plt.savefig(save_fn, bbox_inches='tight', pad_inches=0)
    plt.close()
import argparse
import os
from pathlib import Path
import json
import pandas as pd
import torch.backends.cudnn as cudnn
import torch.distributed as dist


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from motion.dataset.human36m import Human36mDataset, human36m_label_map
from motion.dataset.humanact12 import HumanAct12Dataset, humanact12_label_map

from data_proc.utils import increment_path
import torch
import model.motion_transformer as mits
from model.motion_transformer import DINOHead
from model.utils import bool_flag
import model.utils as utils


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

def flip(motions):
    motion_flip = motions.clone()
    for idx, i in enumerate([0,2,1,3,5,4,6,8,7,9,11,10,12,14,13,15,17,16,19,18,21,20]):
        motion_flip[:,:,idx,:] = motions[:,:,i,:]
    return motion_flip

def train(opt):
    utils.init_distributed_mode(opt)
    utils.fix_random_seeds(opt.seed)
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(opt)).items())))
    cudnn.benchmark = True

    # Prepare Directories
    save_dir = Path(opt.save_dir)
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)

    # Save run settings
    # with open(save_dir / 'opt.yaml', 'w') as f:
    #     yaml.safe_dump(vars(opt), f, sort_keys=True)

    epochs = opt.epochs
    save_interval = opt.save_interval
                          
    # Loggers
    #init wandb
    wandb.init(config=opt, project=opt.wandb_pj_name, entity=opt.entity, name=opt.exp_name, dir=opt.save_dir)

    # Load EmotionMoCap Dataset
    train_dataset = HumanAct12Dataset(data_path="../dataset/experiment/HumanAct12Poses/humanact12poses.pkl", motion_length=150, dataset="train")
    full_proc_label_list = list(humanact12_label_map.values())
    label_map = humanact12_label_map
    data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=opt.batch_size_per_gpu,
        num_workers=0,
        shuffle=True,
        pin_memory=True)

    student = mits.__dict__['vit_tiny'](
        patch_size=opt.patch_size,
        drop_path_rate=0.1,  # stochastic depth
    )
    teacher = mits.__dict__['vit_tiny'](patch_size=[10,1])
    embed_dim = student.embed_dim
    
    student = utils.MultiCropWrapper(student, DINOHead(
            embed_dim,
            opt.out_dim,
            use_bn=opt.use_bn_in_head,
            norm_last_layer=opt.norm_last_layer,
            ))
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, opt.out_dim, opt.use_bn_in_head),
        )
    student, teacher = student.cuda(), teacher.cuda()

    teacher_without_ddp = teacher

    student = nn.parallel.DistributedDataParallel(student, device_ids=[0])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {opt.arch} network.")

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        opt.out_dim,
        16 + 16,  # total number of crops = 2 global crops + local_crops_number
        opt.warmup_teacher_temp,
        opt.teacher_temp,
        opt.warmup_teacher_temp_epochs,
        opt.epochs,
    ).cuda()

    params_groups = utils.get_params_groups(student)

    optimizer = AdamW(params=params_groups, lr=opt.lr)

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        opt.lr * (opt.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        opt.min_lr,
        opt.epochs, len(data_loader),
        warmup_epochs=opt.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        opt.weight_decay,
        opt.weight_decay_end,
        opt.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(opt.momentum_teacher, 1,
                                               opt.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")
    for epoch in range(0, epochs):

        pbar = tqdm(data_loader, position=1, desc="Batch")
        for it, batch in enumerate(pbar):
            it = len(data_loader) * epoch + it  # global training iteration
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_schedule[it]
                if i == 0:  # only the first group is regularized
                    param_group["weight_decay"] = wd_schedule[it]
            motions = batch["rotation_6d_pose_list"].cuda(non_blocking=True)
            motions_flip = flip(motions)

            valid_length = batch['valid_length_list'].cuda(non_blocking=True)
            B,T,J,C = motions.shape

            motions = motions.permute(0,3,1,2)
            # motions = motions.unsqueeze(1)
            motions_flip = motions_flip.permute(0,3,1,2)
            # motions_flip = motions_flip.unsqueeze(1)

            teacher_output = teacher(motions)  # only the 2 global views pass through the teacher
            student_output = student(torch.cat([motions,motions_flip], axis=0))
            # student_output = student(motions)
            # print(teacher_output.shape)
            # print(student_output.shape)
            loss = dino_loss(student_output, teacher_output, epoch)

            # student update
            optimizer.zero_grad()

            loss.backward()
            if opt.clip_grad:
                param_norms = utils.clip_gradients(student, opt.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                            opt.freeze_last_layer)
            optimizer.step()

            # EMA update for the teacher
            with torch.no_grad():
                m = momentum_schedule[it]  # momentum parameter
                for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
            torch.cuda.synchronize()

        # Log
        log_dict = {
            "Train/Loss": loss, 
        }
        wandb.log(log_dict)

        # Save model
        if (epoch % save_interval) == 0:
            ckpt = {'student': student.state_dict(),
                'teacher': teacher.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'opt': opt,
                'dino_loss': dino_loss.state_dict(),
                }
            torch.save(ckpt, os.path.join(wdir, f'train-{epoch}.pt'))
            print(f"[MODEL SAVED at {epoch} Epoch]")

    wandb.run.finish()
    torch.cuda.empty_cache()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='runs/train', help='project/name')
    parser.add_argument('--data_path', type=str, default='/home/taehyun/workspace/childtoy/MotionReasoning/dataset/mocap_emotion_rig', help='Mat dataset path')
    parser.add_argument('--processed_data_dir', type=str, default='processed_data_mocam_80_All_Class_norm3/', help='path to save pickled processed data')
    parser.add_argument('--window', type=int, default=150, help='window')
    parser.add_argument('--wandb_pj_name', type=str, default='MoCAM', help='project name')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=100)
    # parser.add_argument('--device', default='2', help='cuda device')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--exp_name', default='humanact12_dino_patch2_512', help='save to project/name')
    parser.add_argument('--save_interval', type=int, default=5, help='Log model after every "save_period" epoch')
    parser.add_argument('--lr', type=float, default=0.001, help='generator_learning_rate')
    parser.add_argument('--patch_size', default=[10,1], type=list, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=16384, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")
    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--batch_size_per_gpu', default=16, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument('--arch', default='vit_small', type=str,
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")        
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")        
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")        
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")        
    opt = parser.parse_args()

    return opt

if __name__ == "__main__":
    opt = parse_opt()
    opt.save_dir = str(increment_path(Path(opt.project) / opt.exp_name))
    opt.exp_name = opt.save_dir.split('/')[-1]
    # device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")
    train(opt)

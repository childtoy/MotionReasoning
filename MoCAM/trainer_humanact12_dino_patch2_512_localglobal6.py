import argparse
import os
from pathlib import Path
import json
import pandas as pd
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import time
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
from motion.dataset.humanact12 import HumanAct12Dataset2, humanact12_label_map

from data_proc.utils import increment_path
import torch
import model.motion_transformer as mits
from model.motion_transformer import DINOHead
from model.utils import bool_flag
import model.utils2 as utils
from collections import defaultdict, deque
import math
import datetime
class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)



def fft_joint(data, target_joint, min_amplitude):
    output = input.clone()


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def lerp_input_repr(input, valid_len, seq_len, mode):
    if mode == 'global':
        amp_const = np.random.randint(8,12,[1])/10
        select_const = np.random.randint(1,5,[1])
    else :
        amp_const = np.random.randint(0,5,[1])/10
        select_const = np.random.randint(5,10,[1])

    dataset = input.copy()
    data = dataset.copy()
    output = data.copy()
    mask_start_frame = 0
    # torch_pi = np.acos(np.zeros(1)).item() * 2
    # torch_pi = torch_pi.to(device)
    # torch.pi = torch.acos(torch.zeros(1)).item() * 2
    # data_low5 = input.clone()
    t = np.arange(0, valid_len, 1)
    fs = valid_len
    dt = 1/fs
    x1 = np.arange(0, 1, dt)
    # nfft = 샘플 개수
    nfft = len(x1)
    # df = 주파수 증가량
    df = fs/nfft
    k = np.arange(nfft)
    # f = 0부터~최대주파수까지의 범위
    f = k*df 
    # 스펙트럼은 중앙을 기준으로 대칭이 되기 때문에 절반만 구함
    if valid_len % 2 :
        nfft_half = math.trunc(nfft/2)
    else : 
        nfft_half = math.trunc(nfft/2)+1
#     nfft_half = torch.trunc(nfft/2).to(device)
    f0 = f[range(0,nfft_half)] 
    # 증폭값을 두 배로 계산(위에서 1/2 계산으로 인해 에너지가 반으로 줄었기 때문) 
    for jt in range(22):
        for ax in range(6):
                # joint rw1 : 8 , x축
            y1 = data[:valid_len,jt,ax] - np.mean(data[:valid_len,jt,ax])
            fft_y = np.fft.fft(y1)/nfft * 2 
            fft_y0 = fft_y[(range(0,nfft_half))]
            # 벡터(복소수)의 norm 측정(신호 강도)
            amp = np.abs(fft_y0)
            idxy = np.argsort(-amp)   
            y_low5 = np.zeros(nfft)
            for i in range(len(idxy)): 
                freq = f0[idxy[i]] 
                yx = fft_y[idxy[i]] 
                coec = yx.real 
                coes = yx.imag * -1 
                if i < select_const :     
                    y_low5 += amp_const*(coec * np.cos(2 * np.pi * freq * x1) + coes * np.sin(2 * np.pi * freq * x1))                
                else : 
                    y_low5 += (coec * np.cos(2 * np.pi * freq * x1) + coes * np.sin(2 * np.pi * freq * x1))                                        
    
            y_low5 = y_low5 + np.mean(data[:valid_len,jt,ax])
                # print(torch.sum(input[:valid_len,jt,ax]-y_low5))
            data_low5 = y_low5

            if valid_len % 2:
                output[:valid_len, jt:jt+1,ax] = np.expand_dims(data_low5[:valid_len], axis=1)
            else : 
                output[:valid_len, jt:jt+1,ax] = np.expand_dims(data_low5[:valid_len], axis=1)
    dataset = output    
    return dataset
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def flip(motions):
    motion_flip = motions.copy()
    if torch.rand(1)<0.3:
        for idx, i in enumerate([0,2,1,3,5,4,6,8,7,9,11,10,12,14,13,15,17,16,19,18,21,20]):
            motion_flip[:,idx] = motions[:,i]
    return motion_flip

class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.6f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))
        

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
class DataAugmentationDINO(object):
    def __init__(self, local_crops_number, seq_len):
       self.local_crops_number = local_crops_number
       self.seq_len = seq_len
    def __call__(self, query):
        motions, valid_len, labels, proc_labels = query['rotation_6d_pose_list'], query['valid_length_list'], query['labels'], query['proc_label_list'] 
        crops = []
        valid_len_list = []
        labels_list = []
        proc_label_list = []
        self.global_transfo1 = flip(motions)
        self.global_transfo1 = lerp_input_repr(self.global_transfo1, valid_len, self.seq_len, mode='global')
        self.global_transfo2 = flip(motions)
        self.global_transfo2 = lerp_input_repr(self.global_transfo2, valid_len, self.seq_len, mode='global')
        crops.append(self.global_transfo1.transpose(2,0,1))
        crops.append(self.global_transfo2.transpose(2,0,1))
        valid_len_list.append(valid_len)
        valid_len_list.append(valid_len)
        labels_list.append(labels)
        labels_list.append(labels)
        proc_label_list.append(proc_labels)
        proc_label_list.append(proc_labels)
        for _ in range(self.local_crops_number):
            self.local_transfo = flip(motions)
            self.local_transfo = lerp_input_repr(self.local_transfo, valid_len, self.seq_len, mode='local')
            crops.append(self.local_transfo.transpose(2,0,1))
            valid_len_list.append(valid_len)
            labels_list.append(labels)
            proc_label_list.append(proc_labels)
        return {'rotation_6d_pose_list': crops, 'valid_length_list': valid_len_list,'labels': labels_list, 'proc_label_list':proc_label_list }



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
    transform = DataAugmentationDINO(local_crops_number=opt.local_crops_number, seq_len=opt.window)
    train_dataset = HumanAct12Dataset2(data_path="../dataset/experiment/HumanAct12Poses/humanact12poses.pkl", motion_length=150, dataset="train", transform=transform)
    sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=True)
    full_proc_label_list = list(humanact12_label_map.values())
    label_map = humanact12_label_map

    
    data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        sampler=sampler,
        batch_size=opt.batch_size_per_gpu,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True)

    student = mits.__dict__['vit_tiny'](
        patch_size=opt.patch_size,
        drop_path_rate=opt.drop_path_rate,  # stochastic depth
    )
    teacher = mits.__dict__['vit_tiny'](patch_size=opt.patch_size)
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

    student = nn.parallel.DistributedDataParallel(student, device_ids=[1])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {opt.arch} network.")

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        opt.out_dim,
        opt.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        opt.warmup_teacher_temp,
        opt.teacher_temp,
        opt.warmup_teacher_temp_epochs,
        opt.epochs,
    ).cuda()
    fp16_scaler = None
    if opt.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    params_groups = utils.get_params_groups(student)

    optimizer = AdamW(params=params_groups)

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
        data_loader.sampler.set_epoch(epoch)
        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, opt)
        # ============ writing logs ... ============

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

def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args):
    # pbar = tqdm(data_loader, position=1, desc="Batch")
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, datas in enumerate(metric_logger.log_every(data_loader, 10, header)):
    # for it, batch in enumerate(pbar):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        # motions = batch["rotation_6d_pose_list"].cuda(non_blocking=True)
        # valid_length = batch["valid_length_list"].cuda(non_blocking=True)
        # motions = batch["rotation_6d_pose_list"].cuda(non_blocking=True)
        # valid_length = batch["valid_length_list"].cuda(non_blocking=True)        
        # teacher and student forward passes + compute dino loss
        motions =[im.cuda(non_blocking=True) for im in datas['rotation_6d_pose_list']]
        valid_length = [vl.cuda(non_blocking=True) for vl in datas['valid_length_list']]

        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(motions[:2], valid_length[:2],30)  # only the 2 global views pass through the teacher
            student_output = student(motions, valid_length, 30)
            loss = dino_loss(student_output, teacher_output, epoch)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    log_dict = {'Train/Loss':loss}
    wandb.log(log_dict)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='runs/train', help='project/name')
    parser.add_argument('--data_path', type=str, default='/home/taehyun/workspace/childtoy/MotionReasoning/dataset/mocap_emotion_rig', help='Mat dataset path')
    parser.add_argument('--processed_data_dir', type=str, default='processed_data_mocam_80_All_Class_norm3/', help='path to save pickled processed data')
    parser.add_argument('--window', type=int, default=150, help='window')
    parser.add_argument('--wandb_pj_name', type=str, default='MoCAM', help='project name')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--epochs', type=int, default=200)
    # parser.add_argument('--device', default='2', help='cuda device')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--exp_name', default='humanact12_dino_patch2_512', help='save to project/name')
    parser.add_argument('--save_interval', type=int, default=5, help='Log model after every "save_period" epoch')
    parser.add_argument('--lr', type=float, default=0.0005, help='generator_learning_rate')
    parser.add_argument('--patch_size', default=[30,1], type=list, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--use_fp16', type=bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")        
    parser.add_argument('--out_dim', default=4096, type=int, help="""Dimensionality of
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
    parser.add_argument('--batch_size_per_gpu', default=256, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument('--arch', default='vit_small', type=str,
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")        
    parser.add_argument("--warmup_epochs", default=0, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")        
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)    
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')

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

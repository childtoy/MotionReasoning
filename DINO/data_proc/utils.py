import glob
import json
import os
from pathlib import Path
import re
import numpy as np


def drop_end_quat(quaternions, skeleton):
    """
    quaternions: [N,T,Joints,4]
    """

    return quaternions[:, :, skeleton.has_children()]


def write_json(filename, local_q, root_pos, joint_names):
    json_out = {}
    json_out["root_pos"] = root_pos.tolist()
    json_out["local_quat"] = local_q.tolist()
    json_out["joint_names"] = joint_names
    with open(filename, "w") as outfile:
        json.dump(json_out, outfile)



def increment_path(path, exist_ok=False, sep="", mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix("")
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir = path if path.suffix == "" else path.parent  # directory
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path


def process_seq_names(seq_names, dataset):

    if dataset in ['HumanEva', 'HUMAN4D', 'MPI_HDM05']:
        processed_seqname = [x[:-1] for x in seq_names]
    elif dataset == 'PosePrior':
        processed_seqname = []
        for seq in seq_names:
            if 'lar' in seq:
                pr_seq = 'lar'
            elif 'op' in seq:
                pr_seq = 'op'
            elif 'rom' in seq:
                pr_seq = 'rom'
            elif 'uar' in seq:
                pr_seq = 'uar'
            elif 'ulr' in seq:
                pr_seq = 'ulr'
            else:
                ValueError('Invlaid seq name')
            processed_seqname.append(pr_seq)
    else:
        ValueError('Invalid dataset name')
    return processed_seqname
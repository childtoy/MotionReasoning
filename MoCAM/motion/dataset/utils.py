import logging
import os
import pickle
from typing import Callable

import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d


def mask_motion(data, start_ind: int, mask_length: int, gpu: int):
    B, T, D = data.shape
    noise = torch.randn(B, mask_length, D).cuda(gpu)
    data[:, start_ind:start_ind + mask_length, :] = noise
    return data

def load_pickle(pickle_path: str, target_dir: str, func: Callable):

    if pickle_path not in os.listdir(target_dir):
        return_data = func()
        with open(target_dir / pickle_path, "wb") as f:
            pickle.dump(
                return_data, f, protocol=4
            )  # Need to use protocol 4 for serializing large files
        logging.info(f"{pickle_path} saved at {target_dir}.")
    else:
        logging.info(f"{pickle_path} already exists. Loading from disk.")
        return_data = pickle.load(open(target_dir / pickle_path, "rb"))

    return return_data


def calc_stats(dataset, target_fps=30):

    seq_lengths = []

    for k in dataset.keys():
        dataset[k]["proc_frame_ann"]
        for frame_ann in dataset[k]["proc_frame_ann"]:
            seq_lengths.append(frame_ann[f"fps{target_fps}_poses"].shape[0])

    stats = {
        "seq_lengths": seq_lengths,
        "min_seq_length": min(seq_lengths),
        "max_seq_length": max(seq_lengths),
        "mean_seq_length": sum(seq_lengths) / len(seq_lengths),
    }
    return stats


def preprocess_dataset(dataset: str, target_fps: int, data, max_length: int):
    # make fps consistent
    logging.info(f"[{dataset}] FPS Conversion to {target_fps}")
    fps_list = []

    for k in data.keys():
        mocap_fps = data[k]["mocap_framerate"]

        fps_list.append(mocap_fps)

        interval = int(mocap_fps / target_fps)
        # Stage 1: Make sequences to have same fps / drop 'transition' labels
        drop_list = []
        for frame_ann in data[k]["proc_frame_ann"]:
            frame_length = frame_ann["poses"][::interval, :].shape[0]

            # Drop zero-framed sequence or labeled as transition
            if (frame_length == 0) or (frame_ann["proc_label"] == "transition"):
                drop_list.append(frame_ann)

            frame_ann[f"fps{target_fps}_frame_length"] = frame_length
            frame_ann[f"fps{target_fps}_poses"] = frame_ann["poses"][::interval, :]

        for drop_item in drop_list:
            data[k]["proc_frame_ann"].remove(drop_item)

    # support multiple rotation fromat from axis-angle to rotation matrix, quaternion, and 6D.
    logging.info(f"[{dataset}] Expanding rotation format")
    for k in data.keys():
        for frame_ann in data[k]["proc_frame_ann"]:
            num_frames = frame_ann[f"fps{target_fps}_frame_length"]
            poses = frame_ann[f"fps{target_fps}_poses"]
            axis_angles = poses[:, :66].reshape(num_frames, -1, 3)  # 0~3: global_orient, 3~66: controls the body
            reference_rot_mat = axis_angle_to_matrix(torch.Tensor(axis_angles))
            rotation_6d = matrix_to_rotation_6d(reference_rot_mat).numpy()
            frame_ann[f"fps{target_fps}_rotation_6d_pose"] = rotation_6d

    padded_dataset = prepare_padded_set(data, max_length=max_length, fps=target_fps)
    return padded_dataset


def prepare_padded_set(dataset, max_length=150, fps=30, offset=10) -> dict:

    padded_return = {
        "valid_length_list": [],
        "pose_list": [],
        "betas_list": [],
        "raw_label_list": [],
        "proc_label_list": [],
        "rotation_6d_pose_list": [],
    }

    for k in dataset.keys():
        for frame_ann in dataset[k]["proc_frame_ann"]:
            if frame_ann[f"fps{fps}_frame_length"] <= max_length:

                padded_return["valid_length_list"].append(frame_ann[f"fps{fps}_frame_length"])
                padded_return["pose_list"].append(
                    np.pad(
                        frame_ann[f"fps{fps}_poses"],
                        pad_width=(
                            (0, max_length - frame_ann[f"fps{fps}_poses"].shape[0]),
                            (0, 0),
                        ),
                        mode="constant",
                        constant_values=0,
                    ).astype(np.float32)
                )
                padded_return["rotation_6d_pose_list"].append(
                    np.pad(
                        frame_ann[f"fps{fps}_rotation_6d_pose"],
                        pad_width=(
                            (
                                0,
                                max_length
                                - frame_ann[f"fps{fps}_rotation_6d_pose"].shape[0],
                            ),
                            (0, 0),
                            (0, 0),
                        ),
                        mode="constant",
                        constant_values=0,
                    ).astype(np.float32)
                )
                padded_return["betas_list"].append(frame_ann["betas"])
                padded_return["raw_label_list"].append(frame_ann["raw_label"])
                padded_return["proc_label_list"].append(frame_ann["proc_label"])

            else:
                i = 0
                while i + max_length < len(frame_ann[f"fps{fps}_poses"]):
                    padded_return["valid_length_list"].append(
                        max_length
                    )
                    padded_return["pose_list"].append(
                        frame_ann[f"fps{fps}_poses"][i:i+max_length]
                    )
                    padded_return["rotation_6d_pose_list"].append(
                        frame_ann[f"fps{fps}_rotation_6d_pose"][i:i+max_length]
                    )
                    padded_return["betas_list"].append(frame_ann["betas"])
                    padded_return["raw_label_list"].append(frame_ann["raw_label"])
                    padded_return["proc_label_list"].append(frame_ann["proc_label"])

                    i += offset
    return padded_return
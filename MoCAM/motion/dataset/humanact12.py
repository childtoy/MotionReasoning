import pickle
from pathlib import Path

import numpy as np
import torch
from motion.dataset.utils import load_pickle
from pytorch3d.transforms.rotation_conversions import (axis_angle_to_matrix,
                                                       matrix_to_rotation_6d)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

humanact12_label_map = {
    0: "warm up",
    1: "walk",
    2: "run",
    3: "jump",
    4: "drink",
    5: "lift dumbbell",
    6: "sit",
    7: "eat",
    8: "turn steering wheel",
    9: "phone",
    10: "boxing",
    11: "throw",
}


class HumanAct12Dataset(Dataset):
    """
    https://jimmyzou.github.io/publication/2020-PHSPDataset
    """
    def __init__(self, data_path, motion_length=150, dataset="train"):
        self.data_path = Path(data_path)
        self.motion_length = motion_length
        self.tmp_path = Path("tmp/humanact12/")
        self.tmp_path.mkdir(exist_ok=True, parents=True)
        self.upsample_rate = 2
        self.start_offset = 0
        self.transform = transform

        with open(self.data_path, "rb") as f:
            datafile = pickle.load(f)
        X, y = datafile['poses'], datafile['y']
        np.random.seed(0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=119, random_state=0)

        if dataset == "train":
            self.data_dict = {'poses': X_train, 'y': y_train}
        else:
            self.data_dict = {'poses': X_test, 'y': y_test}

        self.padded_dataset = load_pickle(
            f"{dataset}_humanact12_dataset.pkl", self.tmp_path, self._process
        )

    def _process(self):

        self.data_dict['valid_length_list'] = []
        self.data_dict['rotation_6d_pose_list'] = []
        self.data_dict['labels'] = []

        for idx, pose in enumerate(self.data_dict['poses']):

            # FPS
            interpolated_pose = [pose[0]]
            for i in range(1, len(pose)):
                cur_pose = pose[i]
                upsamples = []
                for i in range(1, self.upsample_rate):
                    upsamples.append(interpolated_pose[-1] + i * (cur_pose - interpolated_pose[-1]) / self.upsample_rate)
                interpolated_pose.extend(upsamples)
                interpolated_pose.append(cur_pose)

            interpolated_pose = np.array(interpolated_pose)[self.start_offset:, :]
            interpolated_num_frames = len(np.array(interpolated_pose))

            if interpolated_num_frames > self.motion_length:
                interpolated_num_frames = self.motion_length
            self.data_dict['valid_length_list'].append(interpolated_num_frames)

            if interpolated_num_frames < self.motion_length:
                pose_padded = np.pad(interpolated_pose, ((0, self.motion_length - interpolated_pose.shape[0]), (0, 0)), mode="constant", constant_values=0)
            else:
                pose_padded = interpolated_pose[:self.motion_length, :]
            axis_angles = pose_padded[:, :66].reshape(self.motion_length, -1, 3)
            rot_mat = axis_angle_to_matrix(torch.Tensor(axis_angles))
            rotation_6d = matrix_to_rotation_6d(rot_mat).numpy()
            self.data_dict['rotation_6d_pose_list'].append(rotation_6d)
            self.data_dict['labels'].append(humanact12_label_map[self.data_dict['y'][idx]])
        return self.data_dict

    def __len__(self):
        return len(self.padded_dataset['labels'])

    def __getitem__(self, idx):
        query = {}
        query['rotation_6d_pose_list'] = self.padded_dataset['rotation_6d_pose_list'][idx]
        query['valid_length_list'] = self.padded_dataset['valid_length_list'][idx]
        query['proc_label_list'] = self.padded_dataset['labels'][idx]
        query['labels'] = self.padded_dataset['labels'][idx]
        return query

class HumanAct12Dataset2(Dataset):
    """
    https://jimmyzou.github.io/publication/2020-PHSPDataset
    """
    def __init__(self, data_path, motion_length=150, dataset="train", transform=None):
        self.data_path = Path(data_path)
        self.motion_length = motion_length
        self.tmp_path = Path("tmp/humanact12/")
        self.tmp_path.mkdir(exist_ok=True, parents=True)
        self.upsample_rate = 2
        self.start_offset = 0
        self.transform = transform
        with open(self.data_path, "rb") as f:
            datafile = pickle.load(f)
        X, y = datafile['poses'], datafile['y']
        np.random.seed(0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=119, random_state=0)

        if dataset == "train":
            self.data_dict = {'poses': X_train, 'y': y_train}
        else:
            self.data_dict = {'poses': X_test, 'y': y_test}

        self.padded_dataset = load_pickle(
            f"{dataset}_humanact12_dataset.pkl", self.tmp_path, self._process
        )

    def _process(self):

        self.data_dict['valid_length_list'] = []
        self.data_dict['rotation_6d_pose_list'] = []
        self.data_dict['labels'] = []

        for idx, pose in enumerate(self.data_dict['poses']):

            # FPS
            interpolated_pose = [pose[0]]
            for i in range(1, len(pose)):
                cur_pose = pose[i]
                upsamples = []
                for i in range(1, self.upsample_rate):
                    upsamples.append(interpolated_pose[-1] + i * (cur_pose - interpolated_pose[-1]) / self.upsample_rate)
                interpolated_pose.extend(upsamples)
                interpolated_pose.append(cur_pose)

            interpolated_pose = np.array(interpolated_pose)[self.start_offset:, :]
            interpolated_num_frames = len(np.array(interpolated_pose))

            if interpolated_num_frames > self.motion_length:
                interpolated_num_frames = self.motion_length
            self.data_dict['valid_length_list'].append(interpolated_num_frames)

            if interpolated_num_frames < self.motion_length:
                pose_padded = np.pad(interpolated_pose, ((0, self.motion_length - interpolated_pose.shape[0]), (0, 0)), mode="constant", constant_values=0)
            else:
                pose_padded = interpolated_pose[:self.motion_length, :]
            axis_angles = pose_padded[:, :66].reshape(self.motion_length, -1, 3)
            rot_mat = axis_angle_to_matrix(torch.Tensor(axis_angles))
            rotation_6d = matrix_to_rotation_6d(rot_mat).numpy()
            self.data_dict['rotation_6d_pose_list'].append(rotation_6d)
            self.data_dict['labels'].append(humanact12_label_map[self.data_dict['y'][idx]])
        return self.data_dict

    def __len__(self):
        return len(self.padded_dataset['labels'])

    def __getitem__(self, idx):
        query = {}
        query['rotation_6d_pose_list'] = self.padded_dataset['rotation_6d_pose_list'][idx]
        query['valid_length_list'] = self.padded_dataset['valid_length_list'][idx]
        query['proc_label_list'] = self.padded_dataset['labels'][idx]
        query['labels'] = self.padded_dataset['labels'][idx]
        
        if self.transform is not None: 
            sample = self.transform(query)
        else : 
            sample = query
        return sample        
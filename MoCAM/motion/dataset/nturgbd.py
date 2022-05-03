import os
from pathlib import Path

import joblib
import numpy as np
import torch
from motion.dataset.utils import load_pickle
from pytorch3d.transforms.rotation_conversions import (axis_angle_to_matrix,
                                                       matrix_to_rotation_6d)
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

nturgbd_label_map = {
    6  : 'pick up',
    7  : 'throw',
    8  : 'sit down',
    9  : 'stand up',
    22 : 'cheer up',
    23 : 'hand wave',
    24 : 'kick something',
    38 : 'salute',
    80 : 'squat',
    93 : 'shake fist',
    99 : 'run on the spot',
    100: 'butt kick',
    102: 'side kick',
}

class NTURGBDDataset(Dataset):

    def __init__(self, data_path, motion_length=150, dataset="train"):
        self.data_path = Path(data_path)
        self.data_files = [f for f in os.listdir(self.data_path)]
        self.motion_length = motion_length
        self.tmp_path = Path("tmp/nturgbd/")
        self.tmp_path.mkdir(exist_ok=True, parents=True)

        datafile = {'poses': [], 'y': []}
        for fn in self.data_files:
            file_path = self.data_path / fn
            action_label = int(fn.split('.')[0][-3:])
            with open(file_path, 'rb') as f:
                vibe_data = joblib.load(f)
                if not vibe_data:
                    continue
                smpl_param = vibe_data[1]['pose']
                datafile['poses'].append(smpl_param)
                datafile['y'].append(nturgbd_label_map[action_label])
        X, y = datafile['poses'], datafile['y']
        np.random.seed(0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=0)

        if dataset == "train":
            self.data_dict = {'poses': X_train, 'y': y_train}
        else:
            self.data_dict = {'poses': X_test, 'y': y_test}

        self.padded_dataset = load_pickle(
            f"{dataset}_nturgbd_dataset.pkl", self.tmp_path, self._process
        )


    def _process(self):

        self.data_dict['valid_length_list'] = []
        self.data_dict['rotation_6d_pose_list'] = []
        self.data_dict['labels'] = []

        for pose, label in zip(self.data_dict['poses'], self.data_dict['y']):
            num_frames = pose.shape[0]

            if num_frames > self.motion_length:
                num_frames = self.motion_length
            self.data_dict['valid_length_list'].append(num_frames)
            
            if num_frames < self.motion_length:
                pose_padded = np.pad(pose, ((0, self.motion_length - pose.shape[0]), (0, 0)), mode="constant", constant_values=0)
            else:
                pose_padded = pose[:self.motion_length, :]
            axis_angles = pose_padded[:, :66].reshape(self.motion_length, -1, 3)
            rot_mat = axis_angle_to_matrix(torch.Tensor(axis_angles))
            rotation_6d = matrix_to_rotation_6d(rot_mat).numpy()
            self.data_dict['rotation_6d_pose_list'].append(rotation_6d)
            self.data_dict['labels'].append(label)
        return self.data_dict
        
    def __len__(self):
        return len(self.padded_dataset['labels'])

    def __getitem__(self, idx):
        query = {}
        query['rotation_6d_pose_list'] = self.padded_dataset['rotation_6d_pose_list'][idx]
        query['labels'] = self.padded_dataset['labels'][idx]
        query['valid_length_list'] = self.padded_dataset['valid_length_list'][idx]
        query['proc_label_list'] = self.padded_dataset['labels'][idx]
        return query
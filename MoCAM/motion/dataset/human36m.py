import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from motion.dataset.utils import load_pickle
from pytorch3d.transforms.rotation_conversions import (axis_angle_to_matrix,
                                                       matrix_to_rotation_6d)
from torch.utils.data import Dataset

human36m_label_map = {
    'Directions': 'direct',
    'Discussion': 'discuss',
    'Eating': 'eat',
    'Greeting': 'greet',
    'Phoning': 'phone',
    'Photo': 'photo',
    'Posing': 'pose',
    'Purchases': 'purchase',
    'Sitting': 'sit',
    'SittingDown': 'sit down',  # TODO: Should this be treat as 'sit'?
    'Smoking': 'smoke',
    'Waiting': 'wait',
    'WalkDog': 'walk dog',
    'WalkTogether': 'walk together',  # TODO: Might be integrated into larger categories
    'Walking': 'walk',
}


class Human36mDataset(Dataset):

    def __init__(self, data_path, motion_length=150, dataset="train"):
        self.data_path = Path(data_path)
        self.motion_length = motion_length
        self.tmp_path = Path("tmp/human36m/")
        self.tmp_path.mkdir(exist_ok=True, parents=True)
        self.upsample_rate = 4
        self.start_offset = 10

        # Split protocol 1
        if dataset == "train":
            self.subjects = ["subject1", "subject5", "subject6", "subject7", "subject8", "subject9"]
        else:
            self.subjects = ["subject11"]

        self.padded_dataset = load_pickle(
            f"{dataset}_human36m_dataset.pkl", self.tmp_path, self._process
        )


    def _process(self):
        self.data_dict = {}
        self.data_dict['valid_length_list'] = []
        self.data_dict['rotation_6d_pose_list'] = []
        self.data_dict['labels'] = []

        for subject in self.subjects:
            data_json_fn = self.data_path / "annotations" / f"Human36M_{subject}_data.json"
            with open(data_json_fn, 'r') as f:
                data_json = json.load(f)
            
            smpl_json_fn = self.data_path / "smpl" / f"Human36M_{subject}_smpl_param.json"
            with open(smpl_json_fn, 'r') as f:
                smpl_json = json.load(f)
            
            data_df = pd.DataFrame(data_json['images'])
            data_df['video_id'] = data_df['file_name'].apply(lambda x: x.split('/')[0])
            unique_video_ids = data_df['video_id'].unique()

            for unique_video_id in unique_video_ids:
                video_df = data_df[data_df['video_id'] == unique_video_id].sort_values('frame_idx')
                assert len(video_df["action_name"].unique()) == 1
                assert len(video_df['action_idx'].unique()) == 1
                assert len(video_df['subaction_idx'].unique()) == 1

                action_id = video_df['action_idx'].unique()[0]
                action_name = video_df["action_name"].unique()[0]
                subaction_id = video_df['subaction_idx'].unique()[0]

                self.data_dict['labels'].append(human36m_label_map[action_name])

                # Interpolation
                poses = [np.array(smpl_json[str(action_id)][str(subaction_id)][str(0)]['pose'])]
                for idx in range(1, video_df.shape[0]):
                    if str(idx) in smpl_json[str(action_id)][str(subaction_id)].keys():
                        cur_pose = np.array(smpl_json[str(action_id)][str(subaction_id)][str(idx)]['pose'])

                        upsamples = []
                        for i in range(1, self.upsample_rate):
                            upsamples.append(poses[-1] + i * (cur_pose - poses[-1]) / self.upsample_rate)

                        poses.extend(upsamples)
                        poses.append(cur_pose)
                        
                # Remove the first t-pose
                poses = np.array(poses)[self.start_offset:, :]
                motion_length = poses.shape[0]

                # Curtail full sequence length to be self.motion_length
                if motion_length > self.motion_length:
                    motion_length = self.motion_length
                self.data_dict['valid_length_list'].append(motion_length)

                # Zero-padding
                if motion_length < self.motion_length:
                    pose_padded = np.pad(poses, ((0, self.motion_length - poses.shape[0]), (0, 0)), mode="constant", constant_values=0)
                else:
                    pose_padded = poses[:self.motion_length, :]
                
                axis_angles = pose_padded[:, :66].reshape(self.motion_length, -1, 3)
                rot_mat = axis_angle_to_matrix(torch.Tensor(axis_angles))
                rotation_6d = matrix_to_rotation_6d(rot_mat).numpy()
                self.data_dict['rotation_6d_pose_list'].append(rotation_6d)
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
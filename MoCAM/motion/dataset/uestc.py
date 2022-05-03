import numpy as np
from torch.utils.data import Dataset
from motion.dataset.utils import load_pickle
from pytorch3d.transforms.rotation_conversions import axis_angle_to_matrix, matrix_to_rotation_6d
import torch
from pathlib import Path
import pickle


train_subject_ids = [
        1, 2, 6, 12, 13, 16, 21,
        24, 28, 29, 30, 31, 33, 35, 39, 41, 42, 45, 47, 50, 52, 54, 55, 57, 59,
        61, 63, 64, 67, 69, 70, 71, 73, 77, 81, 84, 86, 87, 88, 90, 91, 93, 96,
        99, 102, 103, 104, 107, 108, 112, 113
    ]
                    
uestc_label_map = {
    0: "punch and knee lift",
    1: "mark time and knee lift",
    2: "jump jack",
    3: "squat",
    4: "forward lunge",
    5: "left lunge",
    6: "left stretch",
    7: "raise hand and jump",
    8: "left kick",
    9: "rotate clap",
    10: "front raise",
    11: "pull chest expand",
    12: "punch",
    13: "wrist circle",
    14: "single dumbbell raise",
    15: "shoulder raise",
    16: "elbow circle",
    17: "dumbbell one arm shoulder press",
    18: "arm circle",
    19: "dumbbell shrug",
    20: "pinch back",
    21: "head counterclockwise circle",
    22: "shoulder abduct",
    23: "deltoid muscle stretch",
    24: "straight forward flexion",
    25: "spinal stretch",
    26: "dumbbell side bend",
    27: "stand opposite elbow to knee crunch",
    28: "stand rotate",
    29: "overhead stretch",
    30: "upper back stretch",
    31: "knee to chest",
    32: "knee circle",
    33: "alternate knee lift",
    34: "bend over twist",
    35: "rope skip",
    36: "stand toe touch",
    37: "stand gastrocnemius calf",
    38: "single leg lateral hop",
    39: "high knee run"
}

class UESTCDataset(Dataset):
    def __init__(self, data_path, motion_length=150, dataset="train"):
        self.data_path = Path(data_path)
        self.data_vibe_pickle = self.data_path / "vibe_cache_refined.pkl"
        self.info_names = self.data_path / "info" / "names.txt"
        self.motion_length = motion_length
        self.tmp_path = Path("tmp/uestc/")
        self.tmp_path.mkdir(exist_ok=True, parents=True)

        with open(self.data_vibe_pickle, "rb") as f:
            uestc = pickle.load(f)

        action_labels = []
        subject_labels = []

        with open(self.info_names, "r") as f:
            for line in f:
                action_label = int(line.strip().split('_')[0][1:])
                subject_label = int(line.strip().split('_')[2][1:])
                action_labels.append(action_label)
                subject_labels.append(subject_label)

        assert len(uestc['pose']) == len(action_labels) == len(subject_labels)

        self.data_dict = {'poses': [], 'y': [], 'subject': []}

        if dataset == "train":
            for pose, label, subject in zip(uestc['pose'], action_labels, subject_labels):
                if subject in train_subject_ids:
                    self.data_dict['poses'].append(pose)
                    self.data_dict['y'].append(label)
                    self.data_dict['subject'].append(subject)
        else:
            for pose, label, subject in zip(uestc['pose'], action_labels, subject_labels):
                if subject not in train_subject_ids:
                    self.data_dict['poses'].append(pose)
                    self.data_dict['y'].append(label)
                    self.data_dict['subject'].append(subject)

        self.padded_dataset = load_pickle(
            f"{dataset}_uestc_dataset.pkl", self.tmp_path, self._process
        )

    def _process(self):

        self.data_dict['valid_length_list'] = []
        self.data_dict['rotation_6d_pose_list'] = []
        self.data_dict['labels'] = []

        for idx, pose in enumerate(self.data_dict['poses']):
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
            self.data_dict['labels'].append(uestc_label_map[self.data_dict['y'][idx]])
        return self.data_dict

    
    def __len__(self):
        return len(self.data_dict['poses'])

    def __getitem__(self, idx):
        query = {}
        query['rotation_6d_pose_list'] = self.padded_dataset['rotation_6d_pose_list'][idx]
        query['labels'] = self.padded_dataset['labels'][idx]
        query['valid_length_list'] = self.padded_dataset['valid_length_list'][idx]
        # query['subject'] = self.padded_dataset['subject'][idx]
        query['proc_label_list'] = self.padded_dataset['labels'][idx]
        return query
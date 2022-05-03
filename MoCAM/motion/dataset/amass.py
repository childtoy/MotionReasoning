import os
from pathlib import Path
import numpy as np
from motion.dataset.parser import BABELParser
from motion.dataset.utils import load_pickle, preprocess_dataset
from torch.utils.data import Dataset
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d
import pandas as pd
import torch
from einops import rearrange

class MotionRetrievalDataset:
    def __init__(self, dataset, cache_path="tmp/amass_dataset/"):
        self.dataset = dataset
        self.tmp_path = Path(cache_path)
        self.retrieval_labels = [
            'sit',
            'walk',
            'dance',
            'tpose',
            'wipe',
            'move around with bent knees',
            'apose',
            'stand',
            'crouch',
            'jog',
            'fight stance',
            'eat food',
            'creep forward',
            'move in a right angle triangular path',
            'jump'
            ]
        self.problem_set = load_pickle(
            f"test_motion_retrieval.pkl", self.tmp_path, self._process
        )

    def __len__(self):
        return len(self.problem_set)


    def _process(self):
        labels = self.dataset.padded_dataset['proc_label_list']
        motions = self.dataset.padded_dataset['rotation_6d_pose_list']
        valid_lengths = self.dataset.padded_dataset['valid_length_list']

        retrieval_dataset = {k: [] for k in self.retrieval_labels}

        for label, motion, valid_length in zip(labels, motions, valid_lengths):
            if label in self.retrieval_labels:
                retrieval_dataset[label].append({'motion': motion, 'valid_length': valid_length})

        problem_set = []
        for label in self.retrieval_labels:
            for _ in range(20):
                problem = {}
                problem['gt'] = label
                problem['query'] = {}
                for label_problem in self.retrieval_labels:
                    chosen_motion_idx = np.random.randint(0, len(retrieval_dataset[label_problem]))
                    problem['query'][label_problem] = retrieval_dataset[label_problem][chosen_motion_idx]
                problem_set.append(problem)
                
        return problem_set

class AMASSDataset(Dataset):
    def __init__(self, dataset_path, babel_dir, cache_path="tmp/amass_dataset/", dataset="train", target_fps=30, motion_length=150, setaside=['HumanEva', 'SFU']):
        self.dataset_path = dataset_path
        self.babel_dir = babel_dir
        self.dataset = dataset
        self.setaside = setaside

        self.tmp_path = Path(cache_path)
        self.tmp_path.mkdir(exist_ok=True, parents=True)

        self.target_fps = target_fps
        self.max_length = motion_length

        self.data = load_pickle(
            f"{dataset}_amass_babel.pkl", self.tmp_path, self._integrate_babel
        )
        self.num_seq = len(self.data)
        self.padded_dataset = load_pickle(
            f"{dataset}_padded.pkl", self.tmp_path, self._process
        )
        self.num_ann_seq = len(self.padded_dataset["pose_list"])

    def __len__(self):
        return self.num_ann_seq

    def __getitem__(self, idx):
        query = {}
        query["valid_length_list"] = self.padded_dataset["valid_length_list"][idx]
        query["proc_label_list"] = self.padded_dataset["proc_label_list"][idx]
        query["rotation_6d_pose_list"] = self.padded_dataset["rotation_6d_pose_list"][idx]
        query["labels"] = self.padded_dataset["proc_label_list"][idx]
        return query

    def _process(self):
        preproccessed = preprocess_dataset(self.dataset, self.target_fps, self.data, self.max_length)
        return preproccessed

    def _integrate_babel(self):

        self.parser = BABELParser(
            self.dataset_path, self.babel_dir, dataset=self.dataset, setaside=self.setaside
        )
        json_data = self.parser.integrate_amass()
        return json_data


class AMASSFullDataset:
    def __init__(self, dataset_path, babel_dir, target_fps=30, motion_length=150, offset=10, setaside=['HumanEva', 'SFU']):
        self.dataset_path = dataset_path
        self.babel_dir = babel_dir
        self.setaside = setaside

        self.tmp_path = Path("tmp/amass_dataset/")
        self.tmp_path.mkdir(exist_ok=True, parents=True)

        self.target_fps = target_fps
        self.motion_length = motion_length
        self.offset = offset

        self.motion_file_dirs = load_pickle(
            f"amass_full_file_dirs.pkl", self.tmp_path, self._load
        )
        self.processed_motion = load_pickle(
            f"amass_full_processed_motion.pkl", self.tmp_path, self._process
        )

    def __getitem__(self, idx):
        query = {}
        query["processed_motion"]  = self.processed_motion["processed_motion"][idx]
        query["valid_length"] = self.processed_motion["valid_length"][idx]
        return query

    def __len__(self):
        return len(self.processed_motion["processed_motion"])

    def _process(self):

        valid_length = []
        processed_motion = []
        for file in self.motion_file_dirs:
            motion = np.load(file)
            # Stage 1 Adjust framerate
            mocap_fps = motion["mocap_framerate"]
            interval = int(mocap_fps / self.target_fps)
            target_fps_motion = motion["poses"][::interval, :66]
            adjust_len = len(target_fps_motion)
            axis_angles = target_fps_motion.reshape(adjust_len, -1, 3)
            ref_rot_mat = axis_angle_to_matrix(torch.Tensor(axis_angles))
            rotation_6d = matrix_to_rotation_6d(ref_rot_mat).numpy()
            # Stage 2 Sliding window for fixed length (self.motion_length)
            if len(rotation_6d) <= self.motion_length:
                padded_motion = np.pad(
                    rotation_6d,
                    ((0, self.motion_length - len(target_fps_motion)), (0, 0), (0,0)),
                    "constant",
                )
                # padded_motion rearrange(padded_motion)
                padded_motion = rearrange(padded_motion, 'b l d -> b (l d)')
                valid_length.append(len(rotation_6d))
                processed_motion.append(padded_motion)
            else:
                i = 0
                while i + self.motion_length < len(target_fps_motion):
                    padded_motion = rotation_6d[i : i + self.motion_length, :, :]
                    i += self.offset
                    padded_motion = rearrange(padded_motion, 'b l d -> b (l d)')
                    valid_length.append(self.motion_length)
                    processed_motion.append(padded_motion)

        assert len(processed_motion) == len(valid_length)
        return {'processed_motion': np.array(processed_motion).astype(np.float32), 'valid_length': np.array(valid_length)}

    def _load(self):
        datasets = os.listdir(self.dataset_path)
        datasets.remove(self.babel_dir)

        motion_file_dirs = []
        for dataset in datasets:
            if dataset in self.setaside:  # Do not use setaside dataset
                continue
            dataset_dir = os.path.join(self.dataset_path, dataset)
            for dirpath, dirnames, filenames in os.walk(dataset_dir):
                if not dirnames:
                    for filename in filenames:
                        if filename == "shape.npz":
                            continue
                        motion_file_dirs.append(os.path.join(dirpath, filename))

        return motion_file_dirs
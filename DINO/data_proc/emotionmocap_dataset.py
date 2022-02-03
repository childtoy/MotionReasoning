from torch.utils.data import Dataset
from emotion import extract, utils
import numpy as np
import pickle
import os
import torch


class EmotionDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        processed_data_dir: str,
        train: bool,
        device: str,
        window: int = 40,
    ):
        self.data_dir = data_dir
        self.teacher_len = 60
        self.student_len = 30

        self.train = train
        # 4.3: ... The training statistics for normalization are computed on windows of 50 frames offset by 20 frames.
        self.window = window
        # 4.3: Given the larger size of ... we sample our test windows from Subject 5 at every 40 frames.
        # The training statistics for normalization are computed on windows of 50 frames offset by 20 frames.
        self.offset = 10 
        self.device = device
        pickle_name = "processed_train_data.pkl" if train else "processed_test_data.pkl"

        if pickle_name in os.listdir(processed_data_dir):
            with open(os.path.join(processed_data_dir, pickle_name), "rb") as f:
                self.data = pickle.load(f)
        else:
            self.data = self.load_emotion()  # Call this last
            with open(os.path.join(processed_data_dir, pickle_name), "wb") as f:
                pickle.dump(self.data, f, pickle.HIGHEST_PROTOCOL)

    @property
    def local_q_dim(self):
        return self.data["local_q"].shape[2] * self.data["local_q"].shape[3]

    @property
    def num_joints(self):
        return self.data["root_p"].shape[2]

    def load_emotion(self):
        # This uses method provided with Emotion MoCap.
        # Refer to paper 3.1 Data formatting
        X, Q, Q_vel, Q_acc, seq_names, labels, subjects, scenes, repeats, start_frame, file_name = extract.get_emotion_set(
            self.data_dir, self.window, self.offset, self.train
        )
        input_data = {}
        input_data["root_p"] = X
        input_data["local_q"] = Q  # q_{t}
        input_data["q_vel"] = Q_vel
        input_data["q_acc"] = Q_acc
        input_data["subjects"] = subjects
        input_data["scenes"] = scenes
        input_data["repeats"] = repeats
        input_data["seq_names"] = seq_names
        input_data["labels"] = labels
        input_data["start_frame"] = start_frame
        input_data["filename"] = file_name
        return input_data

    def __len__(self):
        return self.data["root_p"].shape[0]

    def __getitem__(self, index):
        query = {}
        query["local_q"] = self.data["local_q"][index].astype(np.float32)
        local_q = query["local_q"]
        local_start_idx = np.random.randint(0, self.teacher_len - self.student_len,8)
        global_start_idx = [10, 20]
        # expand_rate = int(self.teacher_len/self.student_len)
        interpolated = torch.zeros([self.teacher_len, 35])

        data_seq = []
        data_seq.append(local_q[global_start_idx[0]:global_start_idx[0]  + self.teacher_len])
        data_seq.append(local_q[global_start_idx[1]:global_start_idx[1]  + self.teacher_len])
        local_dt_list = np.linspace(0, self.student_len-1, self.teacher_len)
        for k in range(8):
            start_idx = local_start_idx[k]

            minibatch_pose_input = torch.Tensor(local_q[start_idx:start_idx + self.student_len])
            j = 0 
            for i in range(self.teacher_len-1):
                interpolate_start = minibatch_pose_input[int(local_dt_list[i])].unsqueeze(0)
                interpolate_end = minibatch_pose_input[int(local_dt_list[i])+1].unsqueeze(0)
                interpolated[i] = slerp(torch.Tensor(interpolate_start), torch.Tensor(interpolate_end), local_dt_list[i]-int(local_dt_list[i]))
                j+=1
            interpolated[-1] = minibatch_pose_input[-1]
            data_seq.append(interpolated)
        return data_seq

def slerp(x, y, a):
    """
    Perfroms spherical linear interpolation (SLERP) between x and y, with proportion a

    :param x: quaternion tensor
    :param y: quaternion tensor
    :param a: indicator (between 0 and 1) of completion of the interpolation.
    :return: tensor of interpolation results
    """
    device = x.device
    len = torch.sum(x * y, dim=-1)

    neg = len < 0.0
    len[neg] = -len[neg]
    y[neg] = -y[neg]

    a = torch.zeros_like(x[..., 0]) + a
    amount0 = torch.zeros(a.shape, device=device)
    amount1 = torch.zeros(a.shape, device=device)

    linear = (1.0 - len) < 0.01
    omegas = torch.arccos(len[~linear])
    sinoms = torch.sin(omegas)

    amount0[linear] = 1.0 - a[linear]
    amount0[~linear] = torch.sin((1.0 - a[~linear]) * omegas) / sinoms

    amount1[linear] = a[linear]
    amount1[~linear] = torch.sin(a[~linear] * omegas) / sinoms
    # res = amount0[..., np.newaxis] * x + amount1[..., np.newaxis] * y

    res = amount0.unsqueeze(1) * x + amount1.unsqueeze(1) * y
    return res

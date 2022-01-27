from torch.utils.data import Dataset
from emotion import extract_bvh, utils
import numpy as np
import pickle
import os


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
        X, Q, Q_vel, Q_acc, seq_names, labels, subjects, scenes, repeats = extract_bvh.get_emotion_set(
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
        return input_data

    def __len__(self):
        return self.data["root_p"].shape[0]

    def __getitem__(self, index):
        query = {}
        query["local_q"] = self.data["local_q"][index].astype(np.float32)
        query["q_vel"] = self.data["q_vel"][index].astype(np.float32)
        query["q_acc"] = self.data["q_acc"][index].astype(np.float32)
        # query["root_p"] = self.data["root_p"][index].astype(np.float32)
        # query["subjects"] = self.data["subjects"][index].astype(np.float32)
        # query["scenes"] = self.data["scenes"][index].astype(np.float32)
        # query["repeats"] = self.data["repeats"][index].astype(np.float32)
        # query["seq_names"] = self.data["seq_names"][index].astype(np.float32)
        query["labels"] = self.data["labels"][index].astype(np.int64)

        return query

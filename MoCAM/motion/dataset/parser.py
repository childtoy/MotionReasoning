import json
import math
import os
from typing import List
import numpy as np


class BABELParser:
    def __init__(self, dataset_path: str, babel_dir: str, dataset: str, setaside: List=['HumanEva', 'SFU']):
        """
        Initialize the AMASSBase class.
        dataset should be one of a ['train', 'val', 'test']
        """
        self.dataset_path = dataset_path
        self.dataset = dataset
        self.babel_dir = os.path.join(dataset_path, babel_dir)
        self.setaside = setaside

        self.json_path = os.path.join(self.babel_dir, f"{dataset}.json")
        self.json_data = self._load()

    def _load(self):
        with open(self.json_path, "r") as f:
            data = json.load(f)
        return data

    def integrate_amass(self) -> List[dict]:
        """
        Integrate AMASS into the BABEL information. BABEL dictionary is updated from corresponding AMASS meta.
        It updates self.train, self.val, and self.test. Processesd data is stored in 'proc_frame_ann'.
        """

        dataset_accessed = []
        missing_dataset = []
        setaside_ids = []

        for i in self.json_data:
            feat_p = self.json_data[i]["feat_p"]
            amass_path = os.path.join(self.dataset_path, feat_p)

            dataset_name = amass_path.split("/")[1]
            if dataset_name in self.setaside:  # Do not include setaside dataset
                setaside_ids.append(i)
                continue
            dataset_accessed.append(dataset_name)

            if os.path.exists(amass_path):
                with open(amass_path, "rb") as f:
                    amass_motion = np.load(f)
                    amass_meta = {
                        "gender": amass_motion["gender"],
                        "mocap_framerate": amass_motion["mocap_framerate"],
                        "betas": amass_motion["betas"].astype(np.float32),
                        "poses": amass_motion["poses"].astype(np.float32),
                    }
                self.json_data[i].update(amass_meta)

                self.json_data[i]["proc_frame_ann"] = []

                if self.json_data[i]["frame_ann"] is not None:
                    for frame_ann in self.json_data[i]["frame_ann"]["labels"]:
                        start_frame = math.floor(
                            frame_ann["start_t"] * amass_meta["mocap_framerate"]
                        )
                        end_frame = math.ceil(
                            frame_ann["end_t"] * amass_meta["mocap_framerate"]
                        )
                        self.json_data[i]["proc_frame_ann"].append(
                            {
                                "start_frame": start_frame,
                                "end_frame": end_frame,
                                "raw_label": frame_ann["raw_label"],
                                "proc_label": frame_ann["proc_label"],
                                "act_cat": frame_ann["act_cat"],
                                "betas": self.json_data[i]["betas"].astype(np.float32),
                                "poses": self.json_data[i]["poses"][
                                    start_frame : end_frame + 1
                                ].astype(np.float32),
                            }
                        )

            else:
                missing_dataset.append(dataset_name)
        print("Integration Completed.")
        print(f"Deleting ids (setaside set): {setaside_ids}")
        for i in setaside_ids:
            del self.json_data[i]
        return self.json_data
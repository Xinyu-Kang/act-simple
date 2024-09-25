import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

class FollowCubeDataset(Dataset):
    def __init__(self, dataset_path, sequence_length=5, transform=None, normalize_robot=True):
        super(FollowCubeDataset, self).__init__()
        self.dataset = h5py.File(dataset_path, 'r')
        self.image_keys = sorted([key for key in self.dataset["images"].keys() if key.startswith("image_")], key=lambda x: int(x.split("_")[1]))
        self.qpos_keys = sorted([key for key in self.dataset["status"].keys() if key.startswith("status_")], key=lambda x: int(x.split("_")[1]))
        self.action_keys = sorted([key for key in self.dataset["actions"].keys() if key.startswith("action_")], key=lambda x: int(x.split("_")[1]))

        assert len(self.image_keys) == len(self.qpos_keys) == len(self.action_keys), "Mismatch in dataset lengths."

        self.transform = transform
        self.normalize_robot = normalize_robot
        self.sequence_length = sequence_length

        # Compute normalization statistics if needed
        if self.normalize_robot:
            qpos_list = []
            action_list = []
            for key in self.qpos_keys:
                qpos = self.dataset["status"][key][()]
                qpos_list.append(qpos)
            self.qpos_mean = np.mean(np.array(qpos_list), axis=0)
            self.qpos_std = np.std(np.array(qpos_list), axis=0) + 1e-5  # Prevent division by zero

            for key in self.action_keys:
                action = self.dataset["actions"][key][()]
                action_list.append(action)
            self.action_mean = np.mean(np.array(action_list), axis=0)
            self.action_std = np.std(np.array(action_list), axis=0) + 1e-5

    def __len__(self):
        # Adjust length to account for sequence length
        return len(self.image_keys) - self.sequence_length + 1

    def __getitem__(self, idx):
        # Ensure idx does not exceed dataset bounds
        if idx + self.sequence_length > len(self.image_keys):
            idx = len(self.image_keys) - self.sequence_length

        imgs = []
        qposes = []
        actions = []

        for i in range(idx, idx + self.sequence_length):
            # Load image
            img_encoded = self.dataset["images"][self.image_keys[i]][()]
            img_decoded = cv2.imdecode(np.frombuffer(img_encoded, np.uint8), cv2.IMREAD_COLOR)
            img_decoded = cv2.cvtColor(img_decoded, cv2.COLOR_BGR2RGB)

            if self.transform:
                img = self.transform(img_decoded)
            else:
                img = torch.from_numpy(img_decoded).permute(2, 0, 1).float() / 255.0  # [C, H, W]

            imgs.append(img)

            # Load robot status
            qpos = self.dataset["status"][self.qpos_keys[i]][()]
            if self.normalize_robot:
                qpos = (qpos - self.qpos_mean) / self.qpos_std
            qpos = torch.from_numpy(qpos).float()
            qposes.append(qpos)

            # Load action
            action = self.dataset["actions"][self.action_keys[i]][()]
            if self.normalize_robot:
                action = (action - self.action_mean) / self.action_std
            action = torch.from_numpy(action).float()
            actions.append(action)

        # Stack sequences
        imgs = torch.stack(imgs)  # [sequence_length, C, H, W]
        qposes = torch.stack(qposes)  # [sequence_length, num_joints + 3]
        actions = torch.stack(actions)  # [sequence_length, 3]

        return imgs, qposes, actions
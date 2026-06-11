import os
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

def load_pose_line(line):
    vals = np.fromstring(line, sep=' ')
    T = np.eye(4, dtype=np.float32)
    T[:3, :4] = vals.reshape(3, 4)
    return T

class KITTIOdometryDataset(Dataset):
    def __init__(self, root, sequence="00", image_folder="image_2", transform=None):
        self.root = root
        self.sequence = sequence
        self.transform = transform

        seq_dir = os.path.join(root, "sequences", sequence, image_folder)
        pose_file = os.path.join(root, "poses", f"{sequence}.txt")

        self.image_paths = sorted(glob.glob(os.path.join(seq_dir, "*.png")))

        with open(pose_file, "r") as f:
            self.poses = [load_pose_line(line.strip()) for line in f.readlines()]

        assert len(self.image_paths) == len(self.poses)

    def __len__(self):
        return len(self.image_paths) - 1

    def __getitem__(self, idx):
        img1 = Image.open(self.image_paths[idx]).convert("RGB")
        img2 = Image.open(self.image_paths[idx + 1]).convert("RGB")
    
        T1 = self.poses[idx]
        T2 = self.poses[idx + 1]
        T_rel = np.linalg.inv(T1) @ T2
    
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
    
        target = torch.tensor(T_rel[:3, :4].reshape(-1), dtype=torch.float32)
    
        return {
            "img1": img1,
            "img2": img2,
            "target": target
        }
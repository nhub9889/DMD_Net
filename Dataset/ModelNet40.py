import os
import glob
import torch
from torch.utils.data import Dataset

class ModelNet40(Dataset):
    def __init__(self, dir, noise_std= 0.01, split= 'train'):
        self.noise_std = noise_std
        self.files = sorted(glob.glob(os.path.join(dir, "*", "*.pt"), recursive= True))
        
        total = len(self.files)
        if split == 'train':
            self.files = self.files[:int(total * 0.9)]
        elif split == 'val':
            self.files = self.files[int(total * 0.9):]
        
        print(f"[{split.upper()}] loaded {len(self.files)}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        clean_verts = data['verts']
        faces = data['faces']
        
        noise = torch.randn_like(clean_verts) * self.noise_std
        noise = torch.clamp(noise, -0.05, 0.05)
        noise_verts = clean_verts + noise
        
        return{
            "clean_verts": clean_verts,
            "noise_verts": noise_verts,
            "faces": faces,
            "A_p": data['A_p'],
            "A_d": data['A_d'],
            "A": data['A']
        }


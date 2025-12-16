import glob
import os
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import DMD
from Dataset.utils import preprocess
from Dataset.ModelNet40 import ModelNet40, VertexLoss

parser = argparse.ArgumentParser(description= "Processing some parameters.")
parser.add_argument('processed_root', type= str, help= "Direction to processed dataset", dest= 'PROCESSED_ROOT', default= "/content/drive/MyDrive/ModelNet40_Processed")
parser.add_argument('device', type= str, default= 'cuda', dest= 'DEVICE')
parser.add_argument('checkpoints_root', type= str, default='checkpoints', help= "Direction to save checkpoint", dest= 'CHECKPONTS_ROOT')
parser.add_argument('batch_size', type= int, default= 1, dest= "BATCH_SIZE")
parser.add_argument('num_workers', type= int, default= 0, dest= "NUM_WORKERS")
parser.add_argument('epochs', type= int, default= 5, dest= "EPOCHS")
parser.add_argument('noise_std', type= float, default= 0.02, dest= "NOISE_STD", help= "Noise std")

EPOCHS = parser.parse_args(['EPOCHS'])
DEVICE = parser.parse_args(['DEVICE'])
NOISE_STD = parser.parse_args(['NOISE_STD'])
BATCH_SIZE = parser.parse_args(['BATCH_SIZE'])
NUM_WORKERS = parser.parse_args(['NUM_WORKERS'])
PROCESSED_ROOT = parser.parse_args(['PROCESSED_ROOT'])
CHECKPOINTS_ROOT = parser.parse_args(['CHECKPOINTS_ROOT'])

train_dataset =ModelNet40(PROCESSED_ROOT, split='train', noise_std=0.02)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers= NUM_WORKERS)

model = DMD(in_dim=512).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = VertexLoss()

print("Starting Training on ModelNet40...")
model.train()

for epoch in range(EPOCHS):
    total_loss = 0
    for i, batch in enumerate(train_loader):
        # Move to GPU
        clean = batch['clean_verts'].squeeze(0).to(DEVICE)
        noisy = batch['noise_verts'].squeeze(0).to(DEVICE) 
        
        A_p = batch['A_p'].squeeze(0).to(DEVICE)
        A_d = batch['A_d'].squeeze(0).to(DEVICE)
        A = batch['A'].squeeze(0).to(DEVICE)
        
        faces = batch['faces'].squeeze(0).to(DEVICE)
        
        # Forward
        optimizer.zero_grad()
        denoised = model(noisy, A_p, A_d, A, faces)
        
        # Loss
        loss = loss_fn(denoised, clean)
        
        # Backward
        loss.backward()
        
        # Clip Gradients (Essential for GCN stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        if i % 100 == 0:
            print(f"Epoch {epoch} | Step {i} | Loss: {loss.item():.6f}")

    print(f"--- Epoch {epoch} Avg Loss: {total_loss / len(train_loader):.6f} ---")
    
    # Save Model
    torch.save(model.state_dict(), CHECKPOINTS_ROOT)



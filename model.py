import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modules import Denoiser, Transformer

class DMD(nn.Module):
    def __init__(self, in_dim= 512):
        self.feature_extractor = Denoiser(in_dim= 512, out_dim= 5, hidden_dim= 2)
        self.denoiser = Denoiser(in_dim= in_dim, hidden_dim= 5, out_dim= 3)
        self.transformer = Transformer(in_dim= in_dim)
        self.fc = nn.Linear(3, in_dim)
    
    def forward(self, mesh, A_primal, A_dual, A, faces):
        """
        mesh: [B, N, 3] Input
        A*: Adjacency matrices
        faces: Topology
        """
        X = self.fc(mesh)
        feature = self.feature_extractor(X, A_primal, A_dual, A, faces)
        out_tranformer = self.transformer(mesh, feature, A_primal)
        out = self.denoiser(out_tranformer, A_primal, A_dual, A, faces)
        return out
    
class VertexLoss(nn.Module):
    def __init__(self, w= 1.0):
        self.w = w
        self.l2 = nn.L2Loss()
        
    def forward(self, pred_mesh, gt_mesh):
        loss = self.l2(pred_mesh, gt_mesh)
        return self.w*loss
        
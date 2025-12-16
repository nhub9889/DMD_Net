import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



def dual_primal(dual, A):
    if A.is_sparse:
        return torch.sparse.mm(A, dual)
    return A@dual

def primal_dual(primal, A):
    return (1.0/3.0)*torch.sparse.mm(A.T, primal) if A.is_sparse else (1.0/3.0)*(A.T@primal)

def dap(Xv, Xf, faces):
    P = Xv[faces]
    d = Xf.unsqueeze(1)
    L = torch.abs(P - d)
    return torch.mean(L, dim= 1)

class GAGG(nn.Module):
    def __init__(self, in_features, out_features, dropout= 0.5):
        super(GAGG, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.dropout = dropout
        
    def forward(self, X, A):
        X = F.dropout(X, self.dropout, training= self.training)
        H = torch.mm(A, X) if A.is_sparse else torch.mm(A, X)
        H = self.fc(H)
        return H

class AGG(nn.Module):
    def __init__(self, in_dim = 512, out_dim = 512):
        super(AGG, self).__init__() 
        self.fc = nn.Linear(in_dim, out_dim)
        self.gagg = GAGG(in_dim, out_dim)
        self.relu = nn.ReLU()
    def forward(self, X, A):
        """
        X: (N, in_dim): node features
        A: (N, N): adjacency matrix
        """
        out_fc = self.fc(torch.mm(A, X) if not A.is_sparse else torch.sparse.mm(A, X))
        out_gagg = self.gagg(out_fc, A)
        return self.relu(out_fc + out_gagg)
    
class PDF(nn.Module):
    def __init__(self, dim= 512):
        super(PDF, self).__init__()
        self.dim = dim
        self.fc_primal = nn.Linear(dim*2, dim)
        self.fc_dual = nn.Linear(dim*2, dim)
        self.relu = nn.ReLU()
        
    def forward(self, primal, dual, A, faces):
        """
        primal: (N_v, dim) primal node features
        dual: (N_f, dim) dual face features
        A: (N_v, N_f) degree-nomalized vertex-face adjacency matrix
        faces: (N_f, 3) mesh topology
        """
        f = dap(primal, dual, faces)
        
        dual_cat = torch.cat([dual, f], dim= 1)
        out_dual = self.relu(self.fc_dual(dual_cat))
        
        mapped = dual_primal(f, A)
        
        primal_cat = torch.cat([primal, mapped], dim= 1)
        out_primal = self.relu(self.fc_primal(primal_cat))
        return out_primal, out_dual

class TwoStreamNet(nn.Module):
    def __init__(self, dim):
        super(TwoStreamNet, self).__init__()
        self.dim = dim
        self.primal_aggs = nn.ModuleList([
            AGG(dim, dim),
            AGG(dim, dim),
            AGG(dim, dim)
        ])
        self.dual_aggs = nn.ModuleList([
            AGG(dim, dim),
            AGG(dim, dim),
            AGG(dim, dim)
        ])
        self.pdf = PDF(dim)
    
    def forward(self,primal, A_primal, A_dual, A, faces):
        """
        primal: (N_v, dim) Initial vertex features
        A_primal: (N_v, N_v) Vertex-Vertex adjacency
        A_dual: (N_f, N_f) Face-Face adjacency
        A: (N_v, N_f) Vertex-Face adjacency
        faces: (N_f, 3) Topology for DAP
        """
        x_p = primal
        x_d = primal_dual(primal, A)
        
        primal_outs = []
        dual_outs = []
        
        for agg in self.primal_aggs:
            primal_outs.append(x_p)
            x_p = agg(x_p, A_primal) + x_p
            
        for agg in self.dual_aggs:
            dual_outs.append(x_d)
            x_d = agg(x_d, A_dual) + x_d
        out_primal, out_dual = self.pdf(x_p, x_d, A, faces)
        out_primal = out_primal + x_p
        out_dual = out_dual + x_d
        return out_primal, out_dual, primal_outs, dual_outs
    
class Denoiser(nn.Module):
    def __init__(self, in_dim= 512, out_dim= 3, hidden_dim= 5):
        super(Denoiser, self).__init__()
        self.fc1 = nn.Linear(in_dim, in_dim//2)
        self.fc2 = nn.Linear(in_dim, in_dim//2)
        self.fc3 = nn.Linear(in_dim//2, in_dim//4)
        self.fc4 = nn.Linear(in_dim//4, out_dim)
        self.hidden = hidden_dim
        self.in_dim = in_dim
    
        self.tsn_block = nn.ModuleList([
            TwoStreamNet(in_dim) for _ in range(hidden_dim)
        ])
        concat_dim = hidden_dim*3*in_dim
        self.dual_mlp = nn.Sequential(
            nn.Linear(concat_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        
        self.primal_mlp = nn.Sequential(
            nn.Linear(concat_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )
        
        self.final_mlp = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )
        self.relu = nn.ReLU()
    
    def forward(self, primal, A_primal, A_dual, A, faces):
        """
        primal: [N, in_dim] or [N, 3]
        A: [N, F] Vertex-Face adjacency
        """
        x_p = primal
        primal_out = []
        dual_out = []
        
        for block in self.tsn_block:
            x_p, _, p, d = block(
                x_p, A_primal, A_dual, A, faces
            )
            
            primal_out.extend(p)
            dual_out.extend(d)
            
        cat_dual = torch.cat(dual_out, dim= 1)
        cat_primal = torch.cat(primal_out, dim= 1)
        feat_dual = self.dual_mlp(cat_dual)
        feat_dual = dual_primal(feat_dual, A)
        feat_primal = self.primal_mlp(cat_primal)
        out = torch.cat([feat_primal, feat_dual], dim= 1)
        out = self.final_mlp(out)
        
        return out
           
        
class Transformer(nn.Module):
    def __init__(self, in_dim= 512, out_dim = 4096):
        super(Transformer, self).__init__()
        self.encoder = nn.Linear(5, in_dim)
        self.fc = nn.Linear(in_dim, out_dim)
        self.relu = nn.ReLU()
        self.agg1 = AGG(in_dim, in_dim)
        self.agg2 = AGG(in_dim, in_dim)
        self.agg3 = AGG(in_dim, in_dim)
        self.agg4 = AGG(in_dim, in_dim)
    
    def forward(self, vertices, features, A):
        """
        vertices: [Batch, N, 3]
        features: [Batch, N, 5]
        A: [Batch, N, N]
        """
        B, N = vertices.shape
        X = self.relu(self.encoder(features))
        
        X1 = self.agg1(X, A)
        X2 = self.agg2(X + X1, A)
        X3 = self.agg3(X + X1 + X2, A)
        out_X = self.fc(X + X1 + X2 + X3)
        out_X = torch.mean(out_X, dim= 1)
        out_X = out_X.view(B, 8, 512)
        combined = torch.cat([vertices, features], dim= 2)
        transformer = torch.bmm(combined, out_X)
        out = self.agg4(transformer, A)
        return self.relu(out)
       
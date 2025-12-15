import torch
import numpy as np
import trimesh
from scipy import sparse
from  fast_simplification import simplify
import glob
import os

TARGET_FACES= 4000

def normalize(mx):
    row = np.array(mx.sum(1))
    r_inv = np.power(row, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat = sparse.diags(r_inv)
    mx = r_mat.dot(mx)
    return mx

def mx_tensor(mx):
    mx = mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((mx.row, mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(mx.data)
    shape = torch.Size(mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

def get_matrices(path):
    mesh = trimesh.load(path, process= False, force= 'mesh')
    vertices= mesh.vertices
    faces= mesh.faces
    N = vertices.shape[0]
    F = faces.shape[0]
    
    edges= mesh.edges_unique
    data= np.ones(len(edges))
    row, col = edges[:, 0], edges[:, 1]
    A_p = sparse.coo_matrix((data, (row, col)), shape=(N, N))
    A_p = A_p + A_p.T + sparse.eye(N)
    A_p = normalize(A_p)
    
    face_adj = mesh.face_adjacency
    data = np.ones(len(face_adj))
    row, col = face_adj[:, 0], face_adj[:, 1]

    A_d = sparse.coo_matrix((data, (row, col)), shape=(F, F))
    A_d = A_d + A_d.T + sparse.eye(F)
    A_d = normalize(A_d)
    
    vf_row = faces.flatten()
    vf_col = np.repeat(np.arange(F), 3)
    
    vf_data = np.ones(len(vf_row))
    A = sparse.coo_matrix((vf_data, (vf_row, vf_col)), shape= (N, F))
    A = normalize(A)
    
    t_verts = torch.from_numpy(vertices).float()
    t_faces = torch.from_numpy(faces).long()
    
    t_Ap = mx_tensor(A_p).to_dense()
    t_Ad = mx_tensor(A_d).to_dense()
    t_A = mx_tensor(A).to_dense()
    
    return t_verts, t_faces, t_Ap, t_Ad, t_A

def preprocess(path, save):
    mesh = trimesh.load(path, force= 'mesh', process= True)
    while len(mesh.faces) < TARGET_FACES:
        mesh = mesh.subdivide()
    
    if len(mesh.faces) > (TARGET_FACES*1.5):
        mesh = mesh.simplify_quadric_decimation(face_count=TARGET_FACES)
    
    components = mesh.split(only_watertight= False)
    if len(components) > 0:
        mesh = max(components, key= lambda m: len(m.faces))
        
    os.makedirs(os.path.dirname(save), exist_ok= True)
    temp = save.replace(".pt", ".off")
    mesh.export(temp)
    verts, faces, A_p, A_d, A = get_matrices(temp)
    
    torch.save({
        "verts": verts,
        "faces": faces,
        "A_p": A_p,
        "A_d": A_d,
        "A": A,
    }, save)
    
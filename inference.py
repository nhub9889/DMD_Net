import torch
import trimesh
import numpy as np
import argparse
import plotly.graph_objects as go
from model import DMD
from plotly.subplots import make_subplots

parser = argparse.ArgumentParser(Description= "Inference one mesh.")
parser.add_argument('test', type= str, help= "Direction to .off or .obj", desc= 'INPUT', default="/content/drive/MyDrive/ModelNet40_Processed/airplane/airplane_0001.off")
parser.add_argument('save', type= str, help= "Direction to save mesh", desc= "OUTPUT", default= "denoised.obj")
parser.add_argument('checkpoint', type= str, help = "Direction to checkpoint", desc= "CHECKPOINT", default="dmd_modelnet40.pth")
parser.add_argument('noise_std', type= float, help= "Noise_std", desc = "NOISE_STD", default= 0.02)
parser.add_argument('device', type= str, default= 'cuda')

INPUT = parser.parse_args(['INPUT'])
OUTPUT = parser.parse_args(['OUTPUT'])
CHECKPOINTS = parser.parse_args(['CHECKPOINTS'])
NOISE_STD = parser.parse_args(['NOISE_STD'])
DEVICE = parser.parse_args(['DEVICE'])

def inference(model, input, output, noise_std= 0.02, device= 'cuda'):
    model.eval()
    print(f"Loading data from: {input}")
    data = torch.load(input)
    verts = data['verts'].to(device)
    noise = torch.randn_like(verts)*noise_std
    noisy_verts =  verts + torch.clamp(noise, -0.05, 0.05)
    
    A_p = data['A_p'].to(device)
    A_d = data['A_d'].to(device)
    A = data['A'].to(device) 
    faces = data['faces'].to(device)
    
    with torch.no_grad():
        denoised = model(noisy_verts, A_p, A_d, A, faces)
        denoised_np = denoised.cpu().numpy()
        faces_np = faces.cpu().numpy()
        mesh = trimesh.Trimesh(vertices= denoised_np, faces= faces_np)
        mesh.export(output)
        
        print(f"Denoised mesh saved to: {output}")
        return mesh
    
def visualize(input, output, noise_std= 0.02):
    def load_mesh(path):
        if path.endswith('.pt'):
            data = torch.load(path)
            verts = data['verts'].cpu().numpy if isinstance(data['verts'], torch.Tensor) else data['verts']
            faces = data['faces'].cpu().numpy if isinstance(data['faces'], torch.Tensor) else data['faces']
            return verts, faces
        elif path.endswith('.off') or path.endswith('.obj'):
            mesh = trimesh.load(path, force= 'mesh', process= False)
            return mesh.vertices, mesh.faces
        
    def plot_mesh(verts, faces, name, color, opacity= 1.0):
        return go.Mesh3d(
            x = verts[:, 0], y = verts[:, 1], z = verts[:, 2],
            i = faces[:, 0], j = faces[:, 1], k = faces[:, 2],
            color = color, opacity = opacity, name = name,
            flatshading= True,
            lighting = dict(ambient= 0.5, diffuse= 0.5)
        )
    
    verts, faces = load_mesh(input)
    denoised_verts, denoised_faces = load_mesh(output)
    noise = np.random.normal(0, noise_std, verts.shape)
    noisy_verts = verts + noise
    
    fig = make_subplots(rows = 1, cols = 3,
                        specs = [[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]],
                        subplot_titles=("1. Ground Truth", "2. Noisy", "3. Denoised"),
                        horizontal_spacing=0.01
                    )
    fig.add_trace(plot_mesh(verts, faces, "Clean", 'lightgreen'), row=1, col=1)
    fig.add_trace(plot_mesh(noisy_verts, faces, "Noisy", 'lightsalmon'), row=1, col=2)
    fig.add_trace(plot_mesh(denoised_verts, denoised_faces, "Denoised", 'lightblue'), row=1, col=3)

    camera = dict(eye=dict(x=0, y=0, z=2.0))
    no_axis = dict(visible=False)
    
    layout_sets = dict(xaxis=no_axis, yaxis=no_axis, zaxis=no_axis, aspectmode='data')
    
    fig.update_layout(
        scene1=layout_sets, scene2=layout_sets, scene3=layout_sets,
        width=1400, height=600,
        title="Denoising Visualizer",
        margin=dict(l=0, r=0, b=0, t=50)
    )

    fig.show()
    
if __name__ == '__main__':
    model = DMD(in_dim=512).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINTS, weights_only = True))

    inference(model, INPUT, OUTPUT, NOISE_STD, DEVICE)
    visualize(INPUT, OUTPUT, NOISE_STD)
    
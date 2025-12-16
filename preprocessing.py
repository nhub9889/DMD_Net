import os
import glob
import argparse
from tqdm import tqdm
from Dataset.utils import preprocess

parser = argparse.ArgumentParser(description= "Preprocessing ModelNet30 for DMD_Net dataset input")
parser.add_argument('modelnet40_dir', type= str, default= "/content/modelnet40-princeton-3d-object-dataset/ModelNet40", dest= "DATASET_ROOT")
parser.add_argument('processed_root', type= str, default= "/content/drive/MyDrive/ModelNet40_Processed", dest= "PROCESSED_ROOT")

DATASET_ROOT = parser.parse_args(['DATASET_ROOT'])
PROCESSED_ROOT = parser.parse_args(['PROCESSED_ROOT'])

train_files = glob.glob(os.path.join(DATASET_ROOT, "*", "train", "*.off"))
print(f"Found {len(train_files)} training models.")
for file_path in tqdm(train_files):
    parts = file_path.split(os.sep)
    category = parts[-3]
    filename = parts[-1].replace(".off", ".pt")

    save_path = os.path.join(PROCESSED_ROOT, category, filename)

    if not os.path.exists(save_path):
        preprocess(file_path, save_path)
        
print('Processing ModelNet40 for DMD_Net done.')
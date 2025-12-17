import os
import sys
import zipfile
import glob
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import torch
# Monkey patch for torch.solve (removed in torch >= 1.9, needed by fastai 1.0)
if not hasattr(torch, 'solve'):
    print("Monkey patching torch.solve for compatibility...")
    def _solve_shim(B, A):
        # torch.solve(B, A) -> torch.linalg.solve(A, B)
        # Old returns (solution, LU), new returns solution. We mock the second return.
        return torch.linalg.solve(A, B), None 
    torch.solve = _solve_shim

from fastai import *
from fastai.vision import *
from fastai.callbacks import *
from sklearn.utils import shuffle

# Filter warnings
warnings.filterwarnings('ignore')

def setup_data(data_path="data"):
    """
    Handles data extraction from zip files if necessary.
    Checks for 'skin-cancer-mnist-ham10000.zip' or 'archive.zip'.
    """
    path = Path(data_path)
    path.mkdir(exist_ok=True, parents=True)
    
    expected_file = path / 'HAM10000_metadata.csv'
    # Flatten logic should run even if data is already extracted to handle nested folders
    
    # Check for zip files if metadata doesn't exist
    if not expected_file.exists():
        zip_files = list(path.glob('*.zip'))
        target_zip = None
        
        # Priority to specific name, then generic archive.zip
        for z in zip_files:
            if z.name == 'skin-cancer-mnist-ham10000.zip':
                target_zip = z
                break
            elif z.name == 'archive.zip':
                target_zip = z
                
        if not target_zip and zip_files:
            target_zip = zip_files[0] # Take first available zip if specific ones not found

        if target_zip:
            print(f"Extracting {target_zip}...")
            with zipfile.ZipFile(target_zip, 'r') as zip_ref:
                zip_ref.extractall(path)
            
            # Check for nested zips (images often in part1/part2 zips within main zip)
            part_zips = list(path.glob('HAM10000_images_part_*.zip'))
            for pz in part_zips:
                print(f"Extracting nested zip {pz.name}...")
                with zipfile.ZipFile(pz, 'r') as zip_ref:
                    zip_ref.extractall(path)
        else:
            # If no zip and no metadata, we can't proceed
            print("No zip file found in data directory. Please upload 'archive.zip' or 'skin-cancer-mnist-ham10000.zip' to the data/ folder.")
            sys.exit(1)
            
    # Flatten directory structure: move images from subfolders to data root
    # Check subfolders like HAM10000_images_part_1 and HAM10000_images_part_2
    print("Checking for nested image directories to flatten...")
    import shutil
    subdirs = list(path.glob('HAM10000_images_part_*'))
    for subdir in subdirs:
        if subdir.is_dir():
            print(f"Moving files from {subdir.name}...")
            for file in subdir.glob('*'):
                try:
                    shutil.move(str(file), str(path / file.name))
                except shutil.Error:
                    pass # File likely exists
            # Optional: remove empty subdir
            try:
                subdir.rmdir()
            except:
                pass
    print("Data setup complete.")
        
    return path

def prepare_dataframe(path):
    """
    Loads metadata, maps lesion types, and performs undersampling.
    """
    metadata_path = path / 'HAM10000_metadata.csv'
    if not metadata_path.exists():
        print(f"Error: {metadata_path} not found.")
        sys.exit(1)
        
    df = pd.read_csv(metadata_path)
    
    lesion_type_dict = {
        'nv': 'Melanocytic nevi',
        'mel': 'Melanoma',
        'bkl': 'Benign keratosis ',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }
    
    df['lesion'] = df.dx.map(lesion_type_dict)
    
    # Undersampling logic from notebook
    num_sample = 200
    
    df_df = df.loc[df['dx'] == "df"][0:115]
    df_vasc = df.loc[df['dx'] == "vasc"][0:142]
    
    # Using safe sampling (replace=False if enough samples, else handled by pandas logic or we limit cap)
    # The notebook code strictly sampled without check, assuming > num_sample for these classes.
    # We will wrap in try/except or checks for robustness, but sticking to notebook logic for fidelity:
    
    classes_to_sample = ['akiec', 'bcc', 'bkl', 'mel', 'nv']
    sampled_dfs = []
    
    for c in classes_to_sample:
        class_df = df.loc[df['dx'] == c]
        if len(class_df) > num_sample:
            # Notebook logic: randomly sample num_sample
            sampled_dfs.append(class_df.sample(num_sample))
        else:
            sampled_dfs.append(class_df)
            
    df_final = pd.concat([df_df, df_vasc] + sampled_dfs)
    df_final = shuffle(df_final)
    
    print("Class distribution after undersampling:")
    print(df_final['lesion'].value_counts())
    
    return df_final

def train():
    data_path = setup_data("data")
    df = prepare_dataframe(data_path)
    
    # Setup DataBunch
    print("Setting up DataBunch...")
    tfms = get_transforms(flip_vert=True)
    # fn_col=1 ('image_id'), label_col=7 ('lesion'), suffix='.jpg'
    # Important: Notebook used 'image_id' which is just ID, need to append suffix
    # The ImageDataBunch.from_df might usually expect extension if not provided? 
    # Notebook source: fn_col=1, suffix='.jpg'
    
    data = ImageDataBunch.from_df(
        path=data_path, 
        df=df, 
        fn_col=1, 
        suffix='.jpg', 
        label_col='lesion',  # Using column name instead of index for robustness
        ds_tfms=tfms, 
        size=224, 
        bs=16
    ).normalize(imagenet_stats)
    
    print(f"Data ready. Classes: {data.classes}")
    
    # Create Learner
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)
    
    print("Creating learner with densenet169...")
    learn = cnn_learner(data, models.densenet169, model_dir=str(model_dir.absolute()))
    
    print("Starting training (3 epochs)...")
    learn.fit_one_cycle(3, 1e-3, callbacks=[SaveModelCallback(learn, every='improvement', monitor='accuracy', name='model_best')])
    
    print("Training complete. Best model saved as 'model_best'.")
    
    # Optional: Save validation results or classification report
    interp = ClassificationInterpretation.from_learner(learn)
    print("Most confused classes:")
    print(interp.most_confused())
    
    # Save the model export for inference (if needed for app)
    learn.export('export.pkl') # FastAI export
    print("Model exported to export.pkl")

if __name__ == "__main__":
    train()

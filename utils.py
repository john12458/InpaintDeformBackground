import torch
import numpy as np
import random
import os
from tqdm.auto import tqdm
import torchvision
import cv2
import matplotlib.pyplot as plt

to_pillow_f = torchvision.transforms.ToPILImage()

def create_guassian_blur_f(guassian_ksize,guassian_sigma):
    return lambda x: cv2.GaussianBlur(x, (guassian_ksize, guassian_ksize), guassian_sigma)[...,np.newaxis]
    
#
def seed_everything(seed):
    # Seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def checkallData(data_dir,image_id_list):
    print("Check all data exist")
    origin_dir = f"{data_dir}/origin/"
    warpped_dir = f"{data_dir}/warpped/"
    mask_dir = f"{data_dir}/mask/"
    mesh_dir = f"{data_dir}/mesh/"
    for select_image_id in tqdm(image_id_list):
        if os.path.exists(f"{origin_dir}/{select_image_id}.jpg")  and os.path.exists(f"{warpped_dir}/{select_image_id}.jpg")             and os.path.exists(f"{mask_dir}/{select_image_id}.npy")             and os.path.exists(f"{mesh_dir}/{select_image_id}.npz") :
                continue
        else:
            raise FileNotFoundError(f"{select_image_id} is broken")

def check_create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def visualize(dicts,img_path):
    length_dict = len(dicts.keys())
    # plot result
    fig, axs = plt.subplots(1, length_dict, figsize=(length_dict*8,8))
    for i, (k,v) in zip(range(length_dict) , dicts.items()):
        axs[i].set_title(k)
        axs[i].imshow(**v)
    fig.savefig(img_path) 
    plt.close(fig)
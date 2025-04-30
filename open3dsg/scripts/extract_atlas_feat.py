import argparse
import os
import json
import numpy as np
import pickle
import open3d as o3d
from torch.utils.tensorboard import SummaryWriter
from open3dsg.config.config import CONF
from open3dsg.models.SGRec3D import SGRec3D
from open3dsg.scripts.sgrec_eval import SGRecEvaluator
import os
import torch
from itertools import accumulate
from tqdm import tqdm
torch.manual_seed(1124)
import torch.multiprocessing as mp
# Add at the beginning of your script
mp.set_start_method('spawn', force=True)
from collections import defaultdict
from AtlasNetLocal.model.model import EncoderDecoder
from AtlasNetLocal.auxiliary import argument_parser as argument_parser

from open3dsg.data.sgrec_dataloader import SGRecDataset, DataLoaderX

def load_model(checkpoint_path, opt):
    model = EncoderDecoder(opt)
    model = torch.nn.DataParallel(model, device_ids=opt.multi_gpu)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cuda:0'))
    return model

def get_encoder_output(model, x):
    with torch.no_grad():
        batch_size, num_objects, num_points, channels = x.shape
        x = x[..., :3]
        x_reshaped = x.reshape(-1, num_points, 3)
        x_permuted = x_reshaped.permute(0, 2, 1)
        features = model.module.encoder(x_permuted)
    return features

def check_atlas_feature(pickle_file_path="/home/ubuntu/SceneGraphEdit/data_out/datasets/OpenSG_3RScan/preprocessed_custom_atlas_feat/7272e161-a01b-20f6-8b5a-0b97efeb6545/data_dict_1.pkl"):
    if not os.path.exists(pickle_file_path):
        print(f"File {pickle_file_path} does not exist.")
        return

    with open(pickle_file_path, "rb") as f:
        data = pickle.load(f)

    if "atlas_feat" in data:
        print("The pickle file contains the 'atlas_feat' key.")
    else:
        print("The pickle file does NOT contain the 'atlas_feat' key.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/ubuntu/SceneGraphEdit/data_out/datasets/OpenSG_3RScan/preprocessed_custom")
    parser.add_argument("--hashes", type=str, default="/home/ubuntu/SceneGraphEdit/data/3RScan/_3DSSG_subset/train_scans.txt")
    parser.add_argument("--output_dir", type=str, default="/home/ubuntu/SceneGraphEdit/data_out/datasets/OpenSG_3RScan/preprocessed_custom_atlas_feat")
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        print(f"Creating output directory {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)
    
    opt = argument_parser.parser()
    torch.cuda.set_device(opt.multi_gpu[0])
    opt['device'] = torch.device('cuda')
    checkpoint_path = '/home/ubuntu/SceneGraphEdit/AtlasNetLocal/checkpoint/network.pth'
    model = load_model(checkpoint_path, opt)
    model.eval()
    sgrec_dataset = SGRecDataset(hashes=args.hashes, data_root_dir=args.data_dir, downsample_points=1500)
    
    for idx in range(len(sgrec_dataset)):
        print(f"Processing {idx} / {len(sgrec_dataset)}")
        data_dict = sgrec_dataset[idx]
        hash, filename = sgrec_dataset.file_mapping[idx]
        print(f"{hash} / {filename}")
        data_dict = sgrec_dataset.collate_fn([data_dict])
        pcl = data_dict['objects_pcl'].to(opt['device'])
        features = get_encoder_output(model, pcl)
        
        # load existing content
        input_path = os.path.join(args.data_dir, hash, filename)
        with open(input_path, 'rb') as f:
            original_content = pickle.load(f)
            
        # update content
        original_content['atlas_feat'] = features.cpu().numpy()
        
        # save content
        hash_dir = os.path.join(args.output_dir, hash)
        if not os.path.exists(hash_dir):
            os.makedirs(hash_dir, exist_ok=True)
        output_path = os.path.join(hash_dir, filename)
        if os.path.exists(output_path):
            print(f"File {output_path} already exists.")
            continue
        with open(output_path, 'wb') as f:
            pickle.dump(original_content, f)
        print(f"Saving new pickle file to {output_path}")
    
    
    
    
    
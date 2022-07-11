from utils.seed import seed_torch
import os

import torch
import argparse
import numpy as np
from pathlib import Path
import imageio
import glob

from tqdm import tqdm
from torch.autograd import Variable
import datetime
from models import make_model
import config

import VoxelUtils as vu
from utils import utils

parser = argparse.ArgumentParser(description='PyTorch SSC Inference')
parser.add_argument('--dataset', type=str, default='nyu', choices=['nyu', 'nyucad', 'debug'],
                    help='dataset name (default: nyu)')
parser.add_argument('--model', type=str, default='palnet', choices=['ddrnet', 'aicnet', 'grfnet', 'palnet', 'palnet_ours'],
                    help='model name (default: palnet)')
parser.add_argument('--files', default="/home/mcheem/data/datasets/large_room/", help='Depth Images')

parser.add_argument('--resume', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--save_completions', type=str, metavar='PATH', default="outputs", help='path to save completions (default: none)')
parser.add_argument('--model_name', default='SSC_debug', type=str, help='name of model to save check points')


global args
args = parser.parse_args()

def load_data_from_depth_image(filename, max_depth=8, cam_k=[[320, 0, 320], [0, 320, 240], [0, 0, 1]]):
    """
    Read depth and pose froms ave npz file and return tsdf voxels.
    """
    rgb = None
    frame_data = np.load(filename[:-4] + ".npz")
    depth_npy = frame_data["depth"]
    cam_pose = frame_data["pose"]

    depth_npy[depth_npy > max_depth] = depth_npy.min()
    vox_origin = utils.get_origin_from_depth_image(depth_npy, cam_k, cam_pose)

    vox_tsdf, depth_mapping_idxs, voxel_occupancy = utils.compute_tsdf(
        depth_npy, vox_origin, cam_k, cam_pose)
    return rgb, torch.as_tensor(depth_npy).unsqueeze(0).unsqueeze(0), torch.as_tensor(vox_tsdf).unsqueeze(0), torch.as_tensor(depth_mapping_idxs).unsqueeze(0).unsqueeze(0), torch.as_tensor(voxel_occupancy.transpose(2, 1, 0)).unsqueeze(0)


def infer():
    """
    Performan Inference on saved depth data and save the results
    to output directory specified in the arguments.
    """
    NUM_CLASSES = 12
    net = make_model(args.model, num_classes=NUM_CLASSES).cuda() 

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            cp_states = torch.load(args.resume, map_location=torch.device('cpu'))
            net.load_state_dict(cp_states['state_dict'], strict=True)
        else:
            raise Exception("=> NO checkpoint found at '{}'".format(args.resume))

     # switch to eval mode
    net.eval() 
    torch.cuda.empty_cache()
    
    # retrive list of saved depth/pose array files
    file_list = glob.glob(str(Path(args.files) / "*.npz"))
    
    for step, depth_file in enumerate(file_list):
        rgb, depth, tsdf, position, occupancy_grid = load_data_from_depth_image(depth_file)
        x_depth = Variable(depth.float()).cuda() 
        position = position.long().cuda() 

        if args.model == 'palnet':
            x_tsdf = Variable(tsdf.float()).cuda() 
            y_pred = net(x_depth=x_depth, x_tsdf=x_tsdf, p=position)
        else:
            x_rgb = Variable(rgb.float())
            y_pred = net(x_depth=x_depth, x_rgb=x_rgb, p=position)

        # calculate per voxel class
        scores = torch.nn.Softmax(dim=0)(y_pred.squeeze())
        scores[0] += 0.3 #Increase offset of empty class to weed out low prob predictions
        preds = torch.argmax(scores, dim=0).cpu().numpy()
        
        # save completions
        if args.save_completions:
            utils.labeled_voxel2ply(preds,"{}/{}_preds.ply".format(args.save_completions, Path(depth_file).stem))
            occupancy_grid_downsampled = utils.downsample_voxel(occupancy_grid.squeeze().numpy())
            utils.labeled_voxel2ply(occupancy_grid_downsampled,"{}/{}_scan.ply".format(args.save_completions, Path(depth_file).stem))


def main():
    # ---- Check CUDA
    if torch.cuda.is_available():
        print("CUDA device found!".format(torch.cuda.device_count()))
    else:
        print("Using CPU!")

    infer()


if __name__ == '__main__':
    main()

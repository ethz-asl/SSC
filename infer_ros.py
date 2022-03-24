#!/usr/bin/env python3
from utils.seed import seed_torch
import os

# Network dependencies
import torch
import argparse
import numpy as np
from torch.autograd import Variable

# ROS dependencies
import rospy
from ssc_msgs.msg import SSCInput
from cv_bridge import CvBridge

# local imports
from models import make_model
from utils import utils
from ssc_msgs.msg import SSCGrid


class ROSInfer:
    def __init__(self):
        self._load_arguments()
        self.net = make_model(self.args.model, num_classes=12)
        self.input_topic = self.args.input_topic
        self.ssc_pub = rospy.Publisher('ssc', SSCGrid, queue_size=10)
        self.bridge = CvBridge()

    def start(self):
        """
        Loads SSC Network model and start listening to depth images.
        """
        # load pretrained model
        self.load_network()
        self.depth_img_subscriber = rospy.Subscriber(
            self.input_topic, SSCInput, self.callback)
        print("SSC inference is setup.")

    def callback(self, ssc_input):
        """
        Receive a Depth image from the simulation, voxelize the depthmap as TSDF, 2D to 3D mapping
        and perform inference using 3D CNN. Publish the results as SSCGrid Message.
        """

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # parse depth image
        cv_image = self.bridge.imgmsg_to_cv2(
            ssc_input.image, desired_encoding='passthrough')

        # prepare pose matrix
        pose_matrix = np.array(ssc_input.pose)
        pose_matrix = pose_matrix.reshape([4,4])
        print("Loading depth data")
        vox_origin, rgb, depth, tsdf, position, occupancy = self._load_data_from_depth_image(
            cv_image, pose_matrix)

        print("Loaded depth data")
        print(np.shape(tsdf))
        x_depth = Variable(depth.float()).to(self.device)
        position = position.long().to(self.device)

        if self.args.model == 'palnet':
            x_tsdf = Variable(tsdf.float()).to(self.device)
            y_pred = self.net(x_depth=x_depth, x_tsdf=x_tsdf, p=position)
        else:
            x_rgb = Variable(rgb.float())
            y_pred = self.net(x_depth=x_depth, x_rgb=x_rgb, p=position)

        scores = torch.nn.Softmax(dim=0)(y_pred.squeeze())
        preds = torch.argmax(scores, dim=0).cpu().numpy()

        #setup message
        msg = SSCGrid()
        msg.data = preds.reshape(-1).astype(np.float32).tolist()

        msg.origin_x = vox_origin[0]
        msg.origin_y = vox_origin[1]
        msg.origin_z = vox_origin[2]
        msg.frame = 'odom'

        msg.width = preds.shape[0]
        msg.height = preds.shape[1]
        msg.depth = preds.shape[2]

        # publish message
        self.ssc_pub.publish(msg)

    def _load_data_from_depth_image(self, depth, cam_pose, max_depth=8, cam_k=[[320, 0, 320], [0, 320, 240], [0, 0, 1]]):
        """
        Takes a depth map, pose as input and outputs the 3D voxeloccupancy, 2D to 3D mapping and TSDF grid.
        """
        rgb = None
        depth_npy = np.array(depth)

        # discard inf points
        depth_npy[depth_npy > max_depth] = depth_npy.min()

        # get voxel grid origin
        vox_origin = utils.get_origin_from_depth_image(
            depth_npy, cam_k, cam_pose)

        # compute tsdf for the voxel grid from depth camera
        vox_tsdf, depth_mapping_idxs, voxel_occupancy = utils.compute_tsdf(
            depth_npy, vox_origin, cam_k, cam_pose)

        return vox_origin, rgb, torch.as_tensor(depth_npy).unsqueeze(0).unsqueeze(0), torch.as_tensor(vox_tsdf).unsqueeze(0), torch.as_tensor(depth_mapping_idxs).unsqueeze(0).unsqueeze(0), torch.as_tensor(voxel_occupancy.transpose(2, 1, 0)).unsqueeze(0)

    def load_network(self):
        """
        Loads a pretrained model for inference
        """
        if os.path.isfile(self.args.resume):
            print("=> loading checkpoint '{}'".format(self.args.resume))
            cp_states = torch.load(self.args.resume, map_location=torch.device('cpu'))
            self.net.load_state_dict(cp_states['state_dict'], strict=True)

        else:
            raise Exception("=> NO checkpoint found at '{}'".format(self.args.resume))

        if torch.cuda.is_available():
            print("CUDA device found!".format(torch.cuda.device_count()))
            self.device = torch.device('cuda')
        else:
            print("Using CPU!")
            self.device = torch.device('cpu')

        self.net = self.net.to(self.device)

        # switch to test mode
        self.net.eval()

    def _load_arguments(self):
        parser = argparse.ArgumentParser(description='PyTorch SSC Inference')
        parser.add_argument('--model', type=str, default='palnet', choices=['ddrnet', 'palnet'],
                            help='model name (default: palnet)')
        parser.add_argument('--resume', type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
        parser.add_argument('--input_topic', type=str, default='ssc_input',    help='Name of the input ros topic (default: /ssc_input)')
        args = parser.parse_args()

        # use argparse arguments as default and override with ros params
        args.model = rospy.get_param('~model', args.model)
        args.resume = rospy.get_param('~resume', args.resume)
        args.input_topic = rospy.get_param('~input_topic', args.input_topic)
        self.args = args


if __name__ == '__main__':
    rospy.init_node("scene_completion")
    ri = ROSInfer()
    ri.start()
    rospy.spin()

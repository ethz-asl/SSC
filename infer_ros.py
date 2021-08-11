import os

# Network Model dependencies
import torch
import argparse
import numpy as np
from torch.autograd import Variable

# ROS dependencies
import rospy
from sensor_msgs.msg import Image
import tf.transformations as tr
import tf
from cv_bridge import CvBridge

# Local imports
from models import make_model
from utils import utils
from ssc_msgs.msg import SSCGrid

class RosInference(object):
    """
    A class to listen for depth image over a ros topic an dpublish
    compelted 3D semantic occupancy grid voer ros
    """
    def __init__(self, publisher_topic='ssc', model='palnet',  frame='/odom', sensor='/airsim_drone/Depth_cam'):
        self._listener = tf.TransformListener()
        self._ssc_publisher =  rospy.Publisher(publisher_topic, SSCGrid, queue_size=10)
        self._cv_bridge = CvBridge()
        self._frame = frame
        self.sensor = sensor
        self._model = model
        self.net = make_model(model, num_classes=12).cuda()

    def load_network(self, model_path):
        """
        Loads a pretrained model for inference
        """
        if os.path.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            cp_states = torch.load(model_path, map_location=torch.device('cpu'))
            self.net.load_state_dict(cp_states['state_dict'], strict=True)
        else:
            raise Exception("=> NO checkpoint found at '{}'".format(model_path))

        # switch to test mode 
        self.net.eval()

    def start_inference(self):
        """
        Subscribe for depth images and start listening
        """
        rospy.Subscriber(self.sensor, Image, self._depth_image_callback) 

    def _inference(self, depth, rgb,  position, tsdf):
        x_depth = Variable(depth.float()).cuda() 
        position = position.long().cuda() 

        if self._model == 'palnet':
            x_tsdf = Variable(tsdf.float()).cuda() 
            y_pred = self.net(x_depth=x_depth, x_tsdf=x_tsdf, p=position)
        else:
            x_rgb = Variable(rgb.float())
            y_pred = self.net(x_depth=x_depth, x_rgb=x_rgb, p=position)
        return y_pred 
    
    def _get_sensor_pose_at_timestamp(self, timestamp):
        """
        Get sensor pose wrt world frame. 
        Returns 4x4 pose matrix [R|T]
        """
        position, orientation = self._listener.lookupTransform(self._frame, self.sensor, timestamp)
        
        # prepare pose matrix
        pose_matrix = tr.quaternion_matrix(orientation)
        pose_matrix[0:3, -1] = position
        return pose_matrix

    def _depth_image_callback(self, depth_image):
        """
        Receives a Depth image from the simulation, voxelize the depthmap as TSDF, 2D to 3D mapping
        and perform inference using 3D CNN. Publish the results as SSCGrid Message.
        """
        torch.cuda.empty_cache()

        # parse depth image
        cv_image = self._cv_bridge.imgmsg_to_cv2(depth_image, desired_encoding='passthrough')
        
        # get depth camera pose wrt odom
        pose_matrix = self._get_sensor_pose_at_timestamp(depth_image.header.stamp)

        # load voxelized data from depth image
        vox_origin, rgb, depth, tsdf, position, occupancy_grid = self._load_data_from_depth_image(cv_image, pose_matrix)
        
        # perform inference using cnn model
        scores = self._inference(depth, rgb, position, tsdf)
        preds = self._labeled_voxels_from_predictions(scores)

        # publish completions over ros 
        self.publish_ssc(preds, vox_origin)

    def _labeled_voxels_from_predictions(self, scores):
        """
        Calculate labeled voxels from perclass voxel scores
        """
        scores = torch.nn.Softmax(dim=0)(scores.squeeze())
        scores[0] += 0.3 #Increase offset of empty class to weed out low prob predictions
        preds = torch.argmax(scores, dim=0).cpu().numpy()
        return preds

    def publish_ssc(self, preds, grid_origin):
        """
        Create a SSCGrid ROS message and publish the completed
        predictions
        """
        msg = SSCGrid()
        msg.data  = preds.reshape(-1).astype(np.float32).tolist()
        
        msg.origin_x = grid_origin[0]
        msg.origin_y = grid_origin[1]
        msg.origin_z = grid_origin[2]
        msg.frame = self._frame

        msg.width = preds.shape[0]
        msg.height= preds.shape[1]
        msg.depth = preds.shape[2]
        
        # publish message
        self._ssc_publisher.publish(msg)

    def _load_data_from_depth_image(self, depth, cam_pose, max_depth=8, cam_k=[[320, 0, 320], [0, 320, 240], [0, 0, 1]]):
        """
        Takes a depth map, pose as input and outputs the 3D voxeloccupancy, 2D to 3D mapping and TSDF grid.
        """
        rgb = None
        depth_npy = np.array(depth)

        # discard inf points
        depth_npy[depth_npy > max_depth] = depth_npy.min()

        # get voxel grid origin
        vox_origin = utils.get_origin_from_depth_image(depth_npy, cam_k, cam_pose)

        # compute tsdf for the voxel grid from depth camera
        vox_tsdf, depth_mapping_idxs, voxel_occupancy = utils.compute_tsdf(
            depth_npy, vox_origin, cam_k, cam_pose)
            
        return vox_origin, rgb, torch.as_tensor(depth_npy).unsqueeze(0).unsqueeze(0), torch.as_tensor(vox_tsdf).unsqueeze(0), torch.as_tensor(depth_mapping_idxs).unsqueeze(0).unsqueeze(0), torch.as_tensor(voxel_occupancy.transpose(2, 1, 0)).unsqueeze(0)

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch SSC Inference')
    parser.add_argument('--model', type=str, default='palnet', choices=['ddrnet', 'aicnet', 'grfnet', 'palnet'],
                        help='model name (default: palnet)')
    parser.add_argument('--sensor', default="/airsim_drone/Depth_cam", help='Depth Images publisher')
    parser.add_argument('--publisher', default="/ssc", help='Publish completed voxels')
    parser.add_argument('--resume', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    return parser.parse_args()

def main():
    rospy.init_node("scene_completion")

    # parse arguments
    args = parse_args()

    # setup inference object
    ros_inference = RosInference(model=args.model, sensor = args.sensor, publisher_topic=args.publisher)

    # load network weights and start inference
    ros_inference.load_network(args.resume)
    ros_inference.start_inference()

    rospy.spin()


if __name__ == '__main__':
    main()

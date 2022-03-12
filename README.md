# Semantic Scene Completion
**A 3D Convolutional Neural Network for semantic scene completions from depth maps**
![3d_palnet_cnn](https://user-images.githubusercontent.com/10983181/148416145-ecc6f019-f7a2-47c1-9c30-6b0261cd4d89.png)

## Table of Contents
0. [Installation](#installation)
0. [Data Preparation](#Data-Preparation)
0. [Train and Test](#Train-and-Test)
0. [Inference (ROS)](#Inference)
0. [Credits](#Credits)

## Installation
### Requirements:
- [pytorch](https://pytorch.org/)≥1.4.0
- [torch_scatter](https://github.com/rusty1s/pytorch_scatter)
- imageio
- scipy
- scikit-learn
- tqdm

You can install the requirements by running `pip install -r requirements.txt`.

If you use other versions of PyTorch or CUDA, be sure to select the corresponding version of torch_scatter.


## Data Preparation
### Download dataset

The raw data can be found in [SSCNet](https://github.com/shurans/sscnet).

The repackaged data can be downloaded via 
[Google Drive](https://drive.google.com/drive/folders/15vFzZQL2eLu6AKSAcCbIyaA9n1cQi3PO?usp=sharing)
or
[BaiduYun(Access code:lpmk)](https://pan.baidu.com/s/1mtdAEdHYTwS4j8QjptISBg).

The repackaged data includes:
```python
rgb_tensor   = npz_file['rgb']		# pytorch tensor of color image
depth_tensor = npz_file['depth']	# pytorch tensor of depth 
tsdf_hr      = npz_file['tsdf_hr']  	# flipped TSDF, (240, 144, 240)
tsdf_lr      = npz_file['tsdf_lr']  	# flipped TSDF, ( 60,  36,  60)
target_hr    = npz_file['target_hr']	# ground truth, (240, 144, 240)
target_lr    = npz_file['target_lr']	# ground truth, ( 60,  36,  60)
position     = npz_file['position']	# 2D-3D projection mapping index
```

### 

## Train and Test

### Configure the data path in [config.py](https://github.com/waterljwant/SSC/blob/master/config.py#L9)

```
'train': '/path/to/your/training/data'

'val': '/path/to/your/testing/data'
```

### Train
Edit the training script [run_SSC_train.sh](https://github.com/waterljwant/SSC/blob/master/run_SSC_train.sh#L4), then run
```
bash run_SSC_train.sh
```

### Test
Edit the testing script [run_SSC_test.sh](https://github.com/waterljwant/SSC/blob/master/run_SSC_test.sh#L3), then run
```
bash run_SSC_test.sh
```

## Inference
The SSC Network is deployed as ROS node for scene completions from depth topics. Please follow the follow instructuon for setting up ROS scene completion node.
### Pre-Requisites
* [**ROS**](http://wiki.ros.org/ROS/Installation)
* **VoxelUtils** (https://github.com/mansoorcheema/VoxelUtils)

   A python library providing optimized C++ (optionally CUDA accelerated) backend implementations for:
     - Fixed size TSDF volume computation from a single depth image
     - Fixed size 3D Volumetric grid computation by probabilistically fusing pointcloud (for SCFusion)
     - 3D projection indices from a 2D depth image 
> **_NOTE:_**  A CUDA only depracated version is available in `voxel_utils` directory.

Install the python extension VoxelUtils package for inference on depth images from ROS topics:
```
cd voxel_utils
make
pyhton setup.py install
```

### Launching Scene Completion ROS node
```
python infer_ros.py --model palnet --resume trained_model.pth
```
A pretreined model can be download from [here.](https://github.com/ethz-asl/SSC/blob/experiment008/pretrained_models/weights/Experiment008/cpBest_SSC_PALNet.pth.tar)

> **_NOTE:_** Make sure to activate catkin workspace before starting  inference. 



## Credits

The Semantic Scene Completion Networks are adapted from PALNet and DDRNet. Please cite the respective papers: 

    @InProceedings{Li2019ddr,
        author    = {Li, Jie and Liu, Yu and Gong, Dong and Shi, Qinfeng and Yuan, Xia and Zhao, Chunxia and Reid, Ian},
        title     = {RGBD Based Dimensional Decomposition Residual Network for 3D Semantic Scene Completion},
        booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
        month     = {June},
        pages     = {7693--7702},
        year      = {2019}
    }
    
    @article{li2019palnet,
	  title={Depth Based Semantic Scene Completion With Position Importance Aware Loss},
	  author={Li, Jie and Liu, Yu and Yuan, Xia and Zhao, Chunxia and Siegwart, Roland and Reid, Ian and Cadena, Cesar},
	  journal={IEEE Robotics and Automation Letters},
	  volume={5},
	  number={1},
	  pages={219--226},
	  year={2019},
	  publisher={IEEE}
}

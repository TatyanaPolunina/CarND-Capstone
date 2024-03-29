This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. For more information about the project, see the project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).

Please use **one** of the two installation options, either native **or** docker installation.

### Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases).

### Docker Installation
[Install Docker](https://docs.docker.com/engine/installation/)

Build the docker container
```bash
docker build . -t capstone
```

Run the docker file
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```

### Port Forwarding
To set up port forwarding, please refer to the "uWebSocketIO Starter Guide" found in the classroom (see Extended Kalman Filter Project lesson).

### Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

### Real world testing
1. Download [training bag](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic_light_bag_file.zip) that was recorded on the Udacity self-driving car.
2. Unzip the file
```bash
unzip traffic_light_bag_file.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_file/traffic_light_training.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images

### Other library/driver information
Outside of `requirements.txt`, here is information on other driver/library versions used in the simulator and Carla:

Specific to these libraries, the simulator grader and Carla use the following:

|        | Simulator | Carla  |
| :-----------: |:-------------:| :-----:|
| Nvidia driver | 384.130 | 384.130 |
| CUDA | 8.0.61 | 8.0.61 |
| cuDNN | 6.0.21 | 6.0.21 |
| TensorRT | N/A | N/A |
| OpenCV | 3.2.0-dev | 2.4.8 |
| OpenMP | N/A | N/A |

We are working on a fix to line up the OpenCV versions between the two.

### Project overview

The following is a system architecture diagram showing the ROS nodes and topics used in the project. The ROS nodes and topics shown in the diagram are described briefly in the Code Structure section below

![architecture](imgs/final-project-ros-graph-v2.png)

### Code structure

Below is a brief overview of the repo structure, along with descriptions of the ROS nodes.

#### **ros/src/tl_detector/tl_detector**

This package contains the traffic light detection node: ```tl_detector.py```. This node takes in data from the ```/image_color```, ```/current_pose```, and ```/base_waypoints``` topics and publishes the locations to stop for red traffic lights to the /traffic_waypoint topic.

The ```/current_pose``` topic provides the vehicle's current position, and ```/base_waypoints``` provides a complete list of waypoints the car will be following.

To detect the lights ```tl_detector/light_classification_model/tl_classfier.py``` is used

#### **/ros/src/waypoint_updater**
This package contains the waypoint updater node: ```waypoint_updater.py```. The purpose of this node is to update the target velocity property of each waypoint based on traffic light detection data. This node will subscribe to the ```/base_waypoints```, ```/current_pose``` and ```/traffic_waypoint topics```, and publish a list of waypoints ahead of the car with target velocities to the ```/final_waypoints``` topic.

#### **/ros/src/twist_controller**

Carla is equipped with a drive-by-wire (dbw) system, meaning the *throttle*, *brake*, and *steering* have electronic control. This package contains the files that are responsible for control of the vehicle: the node ```dbw_node.py``` and the file ```twist_controller.py```. The ```dbw_node``` subscribes to the ```/current_velocity``` topic along with the ```/twist_cmd``` topic to receive target linear and angular velocities. Additionally, this node subscribes to ```/vehicle/dbw_enabled```, which indicates if the car is under dbw or driver control. This node will publish *throttle*, *brake*, and *steering* commands to the ```/vehicle/throttle_cmd```, ```/vehicle/brake_cmd```, and ```/vehicle/steering_cmd``` topics.

### Traffic lights detection

The traffic light detection consists of two main steps:
1. Identify if we have traffic light behind waypoint interval taking into account current vehicle and traffic light coordinates
2. In case traffic light is presented - identify the color of it.

The first step is presented inside ```tl_detector.py``` in *process_traffic_lights* method. If the traffic light is detected in the nearest interval, ```tl_classfier``` is used to identify the light

### Traffic light classification

```TLClassifier``` do the classification staff with two main steps:

1. Define the traffic light bounding box
2. Classify the color of the traffic light using classification model

For the first step pretrained object detection TF model is used based on [COCO dataset](http://cocodataset.org/). The fastest pretrained model ```ssd_mobilenet_v1_coco``` was chosen from the [model zoo](https://github.com/tensorflow/models/blob/477ed41e7e4e8a8443bc633846eb01e2182dc68a/object_detection/g3doc/detection_model_zoo.md). 

From the [label map](https://github.com/tensorflow/models/blob/477ed41e7e4e8a8443bc633846eb01e2182dc68a/object_detection/data/mscoco_label_map.pbtxt) we can find that traffic light object should be defined with tag "10". So current model is used to define traffic light bounding box without any modifications.

It's  completely enough for simulator data to use this model with *detection score = 0.3*. But for real data lower detection score may be needed.

If bounding box is defined the image is extracted from this box and resized to the image with size (48,96) because images with current size are needed as input to precompiled classification model. There is two pretrained models presented inside:
* ```sim_model.h5``` for the simulator data
* ```real_model.h5``` for the real data

#### Traffic lights classifier model

To train both of the models the data gathered by [Vatsal Srivastava](https://github.com/coldKnight/TrafficLight_Detection-TensorFlowAPI#get-the-models) was used.

The presented model is implemented by keras and uses transer learning concept with VGG pretrained model inside, so has such architecture:

* convolution layer
* pretrained VGG model
* two fully connected layers
* softmax activation function from three outputs 

I understand that as we have only three output classes maybe current model is overcomplicated and in case of performance issues need to be reworked to some simple LeNet style model. But presented model show quite good results now.


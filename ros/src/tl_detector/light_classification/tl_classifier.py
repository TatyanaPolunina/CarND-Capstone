#from styx_msgs.msg import TrafficLight
import rospy
import tensorflow as tf
import os
import cv2
import numpy as np

from keras.models import load_model

#pretrained tf model for object generation

object_detection_graph_model = "/frozen_inference_graph.pb"
TRAFFIC_LIGHT_LABEL = 10

class TLClassifier(object):
    def __init__(self, model_name):
    
        current_path = os.path.dirname(os.path.realpath(__file__))
        self.model = load_model(current_path + '/' + model_name, compile=False);
        self.object_detection_graph = tf.Graph()
        self.class_graph =  tf.get_default_graph();
       
        # load  pretrained object generation model
        with self.object_detection_graph.as_default():
            gdef = tf.GraphDef()
            with open(current_path + object_detection_graph_model, 'rb') as f:
                gdef.ParseFromString( f.read() )
                tf.import_graph_def( gdef, name="" )

            #define the names of model nodes. 
            self.session = tf.Session(graph=self.object_detection_graph )
            self.image_tensor = self.object_detection_graph.get_tensor_by_name('image_tensor:0')
            self.boxes =  self.object_detection_graph.get_tensor_by_name('detection_boxes:0')
            self.scores = self.object_detection_graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.object_detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections    = self.object_detection_graph.get_tensor_by_name('num_detections:0')


    #classify the bounding box
    def get_classification(self, image, path_to_save = None):
        #extract traffic light image from the initial image
        traffic_light_image = self.get_traffic_light_image(image)
        if (traffic_light_image is None):
            return -1;
        if (path_to_save is not None):
            cv2.imwrite(path_to_save, traffic_light_image)
        #resize to model input
        traffic_light_image = cv2.resize(traffic_light_image, (48,96))
        img_resize = np.expand_dims(traffic_light_image, axis=0)
        #predict the color of traffic light
        with self.class_graph.as_default():
            prediction = np.argmax(self.model.predict(img_resize, batch_size=1));
        #rospy.logwarn(prediction)
        return prediction
        
    #extract traffic light image from initial image
    def get_traffic_light_image(self, image):
        
        with self.object_detection_graph.as_default():
            #switch from BGR to RGB. Important otherwise detection won't work
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            tf_image_input = np.expand_dims(image,axis=0)
            #run detection model
            (detection_boxes, detection_scores, detection_classes, num_detections) = self.session.run(
                    [self.boxes, self.scores, self.classes, self.num_detections],
                    feed_dict={self.image_tensor: tf_image_input})

            detection_boxes = np.squeeze(detection_boxes)
            detection_classes = np.squeeze(detection_classes)
            detection_scores = np.squeeze(detection_scores)

            tf_image = None
            #for sim images threshold could be bigger for real images even 0.1
            detection_threshold = 0.1

            # Find first detection of signal. It's labeled with number 10
            idx = -1
            for i, cl in enumerate(detection_classes.tolist()):
                if cl == TRAFFIC_LIGHT_LABEL:
                    idx = i;
                    break;

            if idx == -1 or detection_scores[idx] < detection_threshold:
                return None # traffic ligth wasn't found
            else:
                img_shape = image.shape[0:2]
                #boxes recieved in [0,1] interval, need the image coordinates
                img_box = to_image_coords(detection_boxes[idx], img_shape[0], img_shape[1]);
                light_image = image[img_box[0]:img_box[2], img_box[1]:img_box[3]]
                return light_image;
                
def to_image_coords(box, height, width):
    """
    The original box coordinate output is normalized, i.e [0, 1].
    
    This converts it back to the original coordinate based on the image
    size.
    """
    box_coords = np.zeros_like(box)
    box_coords[0] = np.int(box[0] * height)
    box_coords[1] = np.int(box[1] * width)
    box_coords[2] = np.int(box[2] * height)
    box_coords[3] = np.int(box[3] * width)
    
    return box_coords.astype(int)

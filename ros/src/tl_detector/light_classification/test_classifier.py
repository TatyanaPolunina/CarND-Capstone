from tl_classifier import TLClassifier
import cv2
import sys
import os



classifier = TLClassifier("sim_model.h5")
print(sys.argv[1])
image = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(classifier.get_classification(image, "out.jpg"))

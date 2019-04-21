# set up session:
import os
import os.path
import re
import cv2
import pickle
import numpy as np

class Dataset:
    # filename parsing pattern:
    FILENAME_PATTERN = re.compile('(\d+)--(before|after)-(\d+)--(\d+)==(\d)-([a-z]+).jpg')

    def __init__(
        self,
        path = './traffic_light_images'
    ):
        # init dataset:
        self.images = []
        self.labels = []

        # parse dataset:
        for dirpath, _, filenames in os.walk("."):
            for filename in filenames:
                parsed = Dataset.FILENAME_PATTERN.match(filename)
                if parsed:
                    # parse image: 
                    image = cv2.imread(os.path.join(dirpath, filename))
                    self.images.append(image)
                    # parse label:
                    label = int(parsed.group(5))
                    # change unknown code to 3 for deep learning modeling:
                    if label == 4:
                        label = 3
                    self.labels.append(label)
                else:
                    continue

        # format as numpy array
        self.images = np.asarray(self.images)
        self.labels = np.asarray(self.labels)

    def __str__(self):
        pass
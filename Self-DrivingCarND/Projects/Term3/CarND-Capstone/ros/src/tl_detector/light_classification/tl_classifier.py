import os
import os.path
import time
import timeit
import re
from collections import namedtuple
from utils.dataset import Dataset
# image processing:
import numpy as np
import cv2
# keras:
import tensorflow as tf
from keras import backend as K
from keras.layers import Input
from keras.layers import SeparableConv2D, Conv2D, BatchNormalization, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import sparse_categorical_accuracy
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
# ROS:
from styx_msgs.msg import TrafficLight

class TLClassifier(object):
    # region of interest:
    ROI = namedtuple('ROI', ['top', 'left', 'bottom', 'right'], verbose=False)
    # image size:
    ImageSize = namedtuple('ImageSize', ['height', 'width'], verbose=False)
    # traffic light encoding:
    ENCODING = {
        0: TrafficLight.RED,
        1: TrafficLight.YELLOW,
        2: TrafficLight.GREEN,
        3: TrafficLight.UNKNOWN
    }

    def __init__(
        self,
        # ROI(top, left, bottom, right):
        ROI = (30, 0, 530, 800),
        downscale_ratio = 5.0
    ):
        # init ROI:
        (top, left, bottom, right) = ROI
        self.ROI = TLClassifier.ROI(
            top = top, left = left, 
            bottom = bottom, right = right
        )
        # init input image size:
        (height, width) = (self.ROI.bottom - self.ROI.top, self.ROI.right - self.ROI.left)
        self.input_size = TLClassifier.ImageSize(
            height = int(height / downscale_ratio),
            width = int(width / downscale_ratio)
        )
        # format for OpenCV:
        self.input_size_ = tuple(
            (self.input_size.width, self.input_size.height)
        )

        # init model:
        self.graph = tf.get_default_graph()
        with self.graph.as_default():
            self.model = self.__build_classifier()

    def __build_classifier(self):
        """ define classifier for traffic light classification

        Args:
        
        Returns:

        """
        # input:
        input = Input(
            shape=(self.input_size.height, self.input_size.width, 3)
        )

        # separable conv 1:
        x = SeparableConv2D(
            filters = 16,
            kernel_size = (3, 3),
            padding = 'same',
            activation = 'relu',
            depthwise_initializer = 'he_normal',
            pointwise_initializer = 'he_normal'
        )(input)
        x = BatchNormalization()(x)

        # separable conv 2:
        x = SeparableConv2D(
            filters = 16,
            kernel_size = (3, 3),
            padding = 'same',
            activation = 'relu',
            depthwise_initializer = 'he_normal',
            pointwise_initializer = 'he_normal'
        )(x)
        x = BatchNormalization()(x)

        # max pooling 1:
        x = MaxPooling2D()(x)

        # separable conv 3:
        x = SeparableConv2D(
            filters = 32,
            kernel_size = (3, 3),
            padding = 'same',
            activation = 'relu',
            depthwise_initializer = 'he_normal',
            pointwise_initializer = 'he_normal'
        )(x)
        x = BatchNormalization()(x)

        # separable conv 4:
        x = SeparableConv2D(
            filters = 32,
            kernel_size = (3, 3),
            padding = 'same',
            activation = 'relu',
            depthwise_initializer = 'he_normal',
            pointwise_initializer = 'he_normal'
        )(x)
        x = BatchNormalization()(x)

        # max pooling 2:
        x = MaxPooling2D()(x)

        # 1-by-1 conv 4:
        x = Conv2D(
            filters = 4,
            kernel_size = (1, 1),
            padding = 'same',
            activation = 'relu',
            kernel_initializer = 'he_normal'
        )(x)

        # flatten:
        x = Flatten()(x)

        # droput:
        x = Dropout(
            rate = 0.25
        )(x)

        # dense 1:
        x = Dense(
            units = 64,
            activation='relu',
            kernel_initializer='he_normal'
        )(x)

        # prediction:
        prediction = Dense(
            units = 4, 
            activation='softmax'
        )(x)

        # model handler:
        model = Model(inputs=input, outputs=prediction)

        # loss and optimization:
        model.compile(
            optimizer=Adam(lr=5e-4),
            loss='sparse_categorical_crossentropy',
            metrics=[sparse_categorical_accuracy]
        )

        return model        

    def preprocess(self, image):
        """ pre-process camera image for traffic light classification

        Args:
            image (cv2.Image): raw image input
        Returns:
            processed image for classifcation
        """
        # step 1 -- crop to ROI:
        ROI = image[self.ROI.top:self.ROI.bottom, self.ROI.left:self.ROI.right, :]

        # step 2 -- resize:
        resized = cv2.resize(
            ROI,
            self.input_size_,
            interpolation = cv2.INTER_AREA
        )

        # step 3 -- histogram equalization:
        YUV = cv2.cvtColor(resized, cv2.COLOR_BGR2YUV)

        Y = cv2.split(YUV)[0]
        Y_equalized = cv2.equalizeHist(Y)

        YUV[:, :, 0] = Y_equalized

        return cv2.cvtColor(YUV, cv2.COLOR_YUV2BGR)

    def train(self, training, validation, num_epochs, batch_size):
        """ train classifier:

        Args:
            training (Dataset): training dataset
            validation (Dataset): validation dataset
            num_epochs (int): number of epochs
            batch_size (int): mini-batch size
        """
        with self.graph.as_default():
            # training image generator:
            training_generator = ImageDataGenerator(horizontal_flip = True)
            validation_generator = ImageDataGenerator(horizontal_flip = True)

            # number of batches per epoch:
            N_training = training.images.shape[0]
            training_num_batches = (N_training // batch_size) + 1
            N_validation = validation.images.shape[0]
            validation_num_batches = (N_validation // batch_size) + 1

            # callbacks:
            callback_tensorboard = TensorBoard(
                log_dir='./logs', 
                write_graph=True
            )
            callback_reduce_lr_on_plateau = ReduceLROnPlateau(
                factor = 0.5,
                epsilon = 0.0002,
                patience = 2,
                min_lr = 1e-6 
            )
            callback_model_checkpoint = ModelCheckpoint(
                filepath = "models/weights.{epoch:02d}-{val_loss:.2f}.hdf5",
                save_best_only = True,
                save_weights_only = True
            )
            callback_early_stopping = EarlyStopping(
                min_delta = 0.0004,
                patience = 4
            )

            # fit model:
            self.model.fit_generator(
                generator = training_generator.flow(
                    training.images, training.labels, 
                    batch_size = batch_size
                ),
                steps_per_epoch = training_num_batches, 
                epochs = num_epochs,
                validation_data = validation_generator.flow(
                    validation.images, validation.labels,
                    batch_size = batch_size
                ),
                validation_steps = validation_num_batches,
                callbacks=[
                    callback_tensorboard,
                    callback_reduce_lr_on_plateau,
                    callback_model_checkpoint,
                    callback_early_stopping        
                ]
            )

    def predict(self, image):
        """ determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        with self.graph.as_default():
            result = np.argmax(
                self.model.predict(image)
            )

        return TLClassifier.ENCODING[result]

    def save(self, filename):
        """ save model parameters
        """
        with self.graph.as_default():
            self.model.save_weights(filename)

    def load(self, filename):
        """ load model parameters
        """
        with self.graph.as_default():
            self.model.load_weights(filename)

if __name__ == '__main__':
    # init classifer:
    tl_classifier = TLClassifier()

    # load dataset:
    training_dataset = Dataset('./traffic_light_images/training')
    validation_dataset = Dataset('./traffic_light_images/validation')

    # identify existing models:
    filenames = os.listdir('./models')
    if not filenames:
        # train:
        tl_classifier.train(
            training = training_dataset,
            validation = validation_dataset,
            num_epochs = 64,
            batch_size = 512
        )

        # save model:
        timestamp = int(round(time.time()))
        tl_classifier.save(
            "models/{}-model-params.h5".format(timestamp)
        )
    else:
        FILENAME_PATTERN = re.compile('(\d+)-model-params.h5')

        # parse model timestamps:
        timestamps = [int(FILENAME_PATTERN.match(filename).group(1)) for filename in filenames]

        # identify latest model:
        _, latest_model_filename = max(zip(timestamps, filenames), key = lambda t: t[0])
        
        # load latest model:
        tl_classifier.load(os.path.join('./models', latest_model_filename))

        # get inference time statistics:    
        start = time.time()

        N = 1000
        for _ in range(N):
            image = dataset.images[0][np.newaxis]
            label = tl_classifier.predict(image)
            
        end = time.time()

        # mean inference time:
        mean_inference_time = float(end - start) / N

        print "[Mean Inference Time per Frame]: {}--{}".format(mean_inference_time, label)
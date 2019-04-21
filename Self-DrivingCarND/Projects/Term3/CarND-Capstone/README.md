# Udacity Self-Driving Car Engineer Capstone -- System Integration
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
---

[Project Repository](https://github.com/udacity/CarND-Capstonet)

[Project Rubrics](https://review.udacity.com/#!/rubrics/1140/view)


<img src="doc/demo.gif" width="100%" alt="System Integration" />

The goals of this project are the following:

* Launch correctly using the launch files provided in the capstone repo. Please note that we will not be able to accommodate special launch instructions or run additional scripts from your submission to download files. The launch/styx.launch and launch/site.launch files will be used to test code in the simulator and on the vehicle respectively. The submission size limit for this project has been increased to 2GB.
* Smoothly follow waypoints in the simulator.
* Respect the target top speed set for the waypoints' twist.twist.linear.x in waypoint_loader.py. Be sure to check that this is working by testing with different values for kph velocity parameter in /ros/src/waypoint_loader/launch/waypoint_loader.launch. If your vehicle adheres to the kph target top speed set here, then you have satisfied this requirement.
* Stop at traffic lights when needed.
* Stop and restart PID controllers depending on the state of /vehicle/dbw_enabled.
Publish throttle, steering, and brake commands at 50hz.

---

## Notes to Reviewer

Hi there! We are **Team Lambda**! 

|      Name     |        E-Mail        |     Role    |        Responsibility        |
|:-------------:|:--------------------:|:-----------:|:----------------------------:|
|     Ge Yao    | alexgecontrol@qq.com | Team Leader | Traffic Light Classification |
|   Jiho Choi   |  89jan25th@gmail.com | Team Member |       Waypoint Updator       |
| Xu Kuangzheng | kuang.work@gmail.com | Team Member |       Waypoint Updator       |
|   Sun Fengyi  |    85143781@qq.com   | Team Member |        DBW Controller        |

---

## Environment Setup

### 1. Clone This Repo

Clone the project repository
```bash
git clone https://github.com/AlexGeControl/Auto-Car-05-System-Integration-01-Udacity-Capstone.git
```

### 2. Set Up Docker Environment

1. [Install Docker](https://docs.docker.com/engine/installation/)

2. Change to project directory
    ```bash
    cd [YOUR_PROJECT_DIRECTORY]
    ```

3. Build the docker container for project
    ```bash
    docker build . -t capstone
    ```

4. Run the docker file
    ```bash
    docker run -it -v $PWD:/capstone -v /tmp/log:/root/.ros/ -p 4567:4567 capstone
    ```

### 3. Launch ROS
After entering docker, in the default entrypoint directory, execute the following commands:
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```

### 4. Run Simulator

1. Download [Udacity Capstone Project Simulator](https://github.com/udacity/CarND-Capstone/releases) 

2. Extract the file and make the programs runnable
```bash
unzip linux_sys_int.zip
chmod +x sys_int.x86
chmod +x sys_int.x86_64
```
3. Run the simulator

---

## Real World Testing

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

---

## Performance Evaluation

### 1. Smoothly Follow Waypoints in the Simulator

The ego vehicle follows waypoints given smoothly in simulator

<img src="doc/performance-evaluation/01-waypoints.png" width="100%" alt="Waypoints Following Demo" />

### 2. Respect the Target Top Speed Set for the Waypoints

<img src="doc/performance-evaluation/02-speed-limit.png" width="100%" alt="Speed Limit Demo" />

### 3. Stop at Traffic Lights When Needed

<img src="doc/performance-evaluation/03-red-light-stop.png" width="100%" alt="Red Light Stop Demo" />

### 4. Stop and Restart PID Controllers Depending on the State of /vehicle/dbw_enabled

This can be verified with <a href="ros/src/twist_controller/dbw_node.py">DBW node source code</a>. Only when DBW node is enabled will autonomous control commands be sent to ego vehicle to actuate it.

```python
class DBWNode(object):
    """ generate drive-by-wire(DBW) command for autonomous driving
   
        @subscribed /vehicle/dbw_enabled:  the indicator for whether the car is under dbw or driver control
        @subscribed /current_velocity:     the vehicle's target linear velocities
        @subscribed /twist_cmd:            the vehicle's target angular velocities

        @published  /vehicle/brake_cmd:    the final brake for electronic control   
        @published  /vehicle/throttle_cmd: the final throttle for electronic control  
        @published  /vehicle/steering_cmd: the final steering for electronic control      
    """
    DBW_UPDATE_FREQ = 50 # Waypoint update frequency

    ...

    def loop(self):
        """ 
            The DBW system on Carla expects messages at 50Hz
            It will disengage (reverting control back to the driver) if control messages are published at less than 10hz
        """
        rate = rospy.Rate(DBWNode.DBW_UPDATE_FREQ) # at least 50Hz
        while not rospy.is_shutdown():
            # Get predicted throttle, brake, and steering using `twist_controller`
            throttle, brake, steer = self.controller.control(
                is_dbw_enabled = self.is_dbw_enabled,
                actual_longitudinal_velocity = self.actual_longitudinal_velocity,
                target_longitudinal_velocity = self.target_longitudinal_velocity,
                target_angular_velocity = self.target_angular_velocity
            )

            # Only publish the control commands if dbw is enabled
            if self.is_dbw_enabled:
                self.publish(throttle, brake, steer)
            rate.sleep()
```

### 5. Publish Throttle, Steering, and Brake Commands at 50hz

This can be verified with <a href="ros/src/twist_controller/dbw_node.py">DBW node source code</a>. The control command publish frequency is set as 50Hz.

```python
class DBWNode(object):
    """ generate drive-by-wire(DBW) command for autonomous driving
   
        @subscribed /vehicle/dbw_enabled:  the indicator for whether the car is under dbw or driver control
        @subscribed /current_velocity:     the vehicle's target linear velocities
        @subscribed /twist_cmd:            the vehicle's target angular velocities

        @published  /vehicle/brake_cmd:    the final brake for electronic control   
        @published  /vehicle/throttle_cmd: the final throttle for electronic control  
        @published  /vehicle/steering_cmd: the final steering for electronic control      
    """
    DBW_UPDATE_FREQ = 50 # Waypoint update frequency

    ...

    def loop(self):
        """ 
            The DBW system on Carla expects messages at 50Hz
            It will disengage (reverting control back to the driver) if control messages are published at less than 10hz
        """
        rate = rospy.Rate(DBWNode.DBW_UPDATE_FREQ) # at least 50Hz
        while not rospy.is_shutdown():
            # Get predicted throttle, brake, and steering using `twist_controller`
            throttle, brake, steer = self.controller.control(
                is_dbw_enabled = self.is_dbw_enabled,
                actual_longitudinal_velocity = self.actual_longitudinal_velocity,
                target_longitudinal_velocity = self.target_longitudinal_velocity,
                target_angular_velocity = self.target_angular_velocity
            )

            # Only publish the control commands if dbw is enabled
            if self.is_dbw_enabled:
                self.publish(throttle, brake, steer)
            rate.sleep()
```

---

## Solution Architecture

System architecture is shown in the following graph.

<img src="doc/architecture/architecture.png" width="100%" alt="System Architecture" />

### 1. Waypoint Update

<img src="doc/architecture/waypoint-update-node.png" width="100%" alt="Waypoint Update" />

<a href="ros/src/waypoint_updater/waypoint_updater.py">Waypoint Update Node</a>

### 2. DBW Control

<img src="doc/architecture/twist-control-node.png" width="100%" alt="DBW Control" />

<a href="ros/src/twist_controller/dbw_node.py">DBW Control Node</a> 

### 3. Traffic Light Detection

<img src="doc/architecture/traffic-light-detection-and-classification-node.png" width="100%" alt="Traffic Light Detection" />

<a href="ros/src/tl_detector/tl_detector.py">Traffic Light Detection Node</a> detects and classifies incoming traffic light in order to stop ego vehicle correctly at stop line when there is a red light ahead.

#### Image Preprocessing

In order to strike a good balance between scene representation and inference speed, the following pre-processings are used:

1. Region of Interest(ROI) is defined as (30, 0, 530, 800)(top, left, bottom, right) according to scene image statistics.
2. After crop to ROI, the image is further down-sampled to (100, 160)(width, height) to reduce processing time of scene inference.
3. Finally, histogram equalization is applied to scene to reduce variance in illumination.

Corresponding Python implementation is demonstrated below:

```python
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
```

#### Traffic Light Classification

Here a deep neural network is built to classify camera image from ego vehicle and deduce traffic light status.

The network architecture is shown in the following table:

|         Layer         |                                          Description                                          |
|:---------------------:|:---------------------------------------------------------------------------------------------:|
|         input         |                                  placeholder for input image                                  |
|    separable conv 1   | kernel: 3x3x16, stride: 1x1, padding: 'SAME', activation: 'ReLU', initialization: 'He normal' |
| batch normalization 1 |                                      batch normalization                                      |
|    separable conv 2   | kernel: 3x3x16, stride: 1x1, padding: 'SAME', activation: 'ReLU', initialization: 'He normal' |
| batch normalization 2 |                                      batch normalization                                      |
|     max pooling 1     |                                    kernel: 2x2, stride: 2x2                                   |
|    separable conv 3   | kernel: 3x3x32, stride: 1x1, padding: 'SAME', activation: 'ReLU', initialization: 'He normal' |
| batch normalization 3 |                                      batch normalization                                      |
|    separable conv 4   | kernel: 3x3x32, stride: 1x1, padding: 'SAME', activation: 'ReLU', initialization: 'He normal' |
| batch normalization 4 |                                      batch normalization                                      |
|     max pooling 1     |                                    kernel: 2x2, stride: 2x2                                   |
|       1x1 conv 1      |  kernel: 1x1x4, stride: 1x1, padding: 'SAME', activation: 'ReLU', initialization: 'He normal' |
|        flatten        |                                            flatten                                            |
|        dropout        |                                           rate: 0.25                                          |
|        dense 1        |             kernel: (40x25x4)x64, activation: 'ReLU', initialization: 'He normal'             |
|        dense 2        |             kernel: 64x4, activation: 'softmax', initialization: 'glorot uniform'             |

Corresponding Keras implementation is as follows:

```python
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
```

#### Training

The network is trained using training set consists of 9218 images and validation set consists of 2912 images.

During the training process, after each epoch the model will be evaluated using validation set and only the best model would be kept using Keras ModelCheckpoint callback.

Keras ReduceLROnPlateau callback is used to tune the model in finer grain when performance plateau is reached. 

Finally, Keras EarlyStopping callback is used to prevent over-fitting.

```python
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
```
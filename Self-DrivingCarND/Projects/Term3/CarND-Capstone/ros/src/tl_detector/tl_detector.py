#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier

import os
import re
import numpy as np
from scipy.spatial import KDTree
import tf
import cv2
import yaml

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    """ detect and classify traffic light

        @subscribed /base_waypoints:         the complete list of waypoints the car will be following
        @subscribed /current_pose:           the vehicle's current position
        @subscribed /image_color:            the image stream from the car's camera
        @subscribed /vehicle/traffic_lights: the exact location and status of all traffic lights in simulator
        
        @published  /traffic_waypoint:       the index of the waypoint for nearest upcoming red light's stop line
    """
    CAMERA_IMAGE_CLASSIFICATION_WPS = 53
    CAMERA_IMAGE_COLLECTION_AFTER_LINE_COUNT = 27

    def __init__(self):
        rospy.init_node('tl_detector')

        # load config params:
        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        # state variables:
        self.pose = None
        self.waypoints = None
        self._waypoints_location = None
        self._waypoints_size = None
        self._waypoints_index = None
        self.camera_image = None
        self.lights = []

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        # image collector:
        self.after_stop_line_count = TLDetector.CAMERA_IMAGE_COLLECTION_AFTER_LINE_COUNT

        # classifier--subscriber:
        self.listener = tf.TransformListener()
        # classifier--format convertor:
        self.bridge = CvBridge()
        # classifier--pre-trained model:
        filenames = os.listdir('./light_classification/models')
        if not filenames:
            pass
        else:
            # model name pattern:
            FILENAME_PATTERN = re.compile('(\d+)-model-params.h5')

            # parse model timestamps:
            timestamps = [int(FILENAME_PATTERN.match(filename).group(1)) for filename in filenames]

            # identify latest model:
            _, latest_model_filename = max(zip(timestamps, filenames), key = lambda t: t[0])

            # load latest model:
            self.light_classifier = TLClassifier()
            self.light_classifier.load(
                os.path.join('./light_classification/models', latest_model_filename)
            )

        # subscribe:
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/image_color', Image, self.image_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)

        # publish:
        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        rospy.spin()

    def waypoints_cb(self, waypoints):
        """ load base waypoints from system 

        Args:
            waypoints (list of Waypoint): reference trajectory as waypoints
        """
        if not self.waypoints:
            # load waypoints:
            self.waypoints = waypoints
            # build index upon waypoints:
            self._waypoints_location = np.array(
                [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            )
            self._waypoints_size, _ = self._waypoints_location.shape
            self._waypoints_index = KDTree(
                self._waypoints_location
            )

    def pose_cb(self, msg):
        """ parse ego vehicle pose

        Args:
            msg (PoseStamped): ego vehicle pose
        """
        self.pose = msg

    def image_cb(self, msg):
        """ identify red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera
        """
        # parse input:
        self.has_image = True
        self.camera_image = msg
        # process traffic lights:
        stop_line_waypoint_index, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            stop_line_waypoint_index = stop_line_waypoint_index if (state == TrafficLight.RED or state == TrafficLight.YELLOW) else -1
            self.last_wp = stop_line_waypoint_index
            self.upcoming_red_light_pub.publish(Int32(stop_line_waypoint_index))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def traffic_cb(self, msg):
        """ parse traffic light status from telegram

        Args:
            msg (TrafficLightArray): list of all traffic light telegrams
        """
        self.lights = msg.lights

    def get_next_waypoint_index(self):
        """ get next waypoint index for ego vehicle
        """
        # ego vehicle location:
        ego_vehicle_location = np.array(
            [self.pose.pose.position.x, self.pose.pose.position.y]
        )

        # closest waypoint
        _, closest_waypoint_index = self._waypoints_index.query(ego_vehicle_location)

        closest_waypoint_location = self._waypoints_location[closest_waypoint_index]
        previous_waypoint_location = self._waypoints_location[closest_waypoint_index - 1]

        # whether the closest waypoint is the next waypoint:
        is_next_waypoint = (
            np.dot(
                closest_waypoint_location - ego_vehicle_location,
                previous_waypoint_location - ego_vehicle_location
            ) < 0.0
        )

        # next waypoint index:
        next_waypoint_index = closest_waypoint_index
        if not is_next_waypoint:
            next_waypoint_index = (closest_waypoint_index + 1) % self._waypoints_size

        return next_waypoint_index

    def get_closest_waypoint(self, position):
        """ get closest waypoint index for stop line

        Args:
            position (Pose): ego vehicle pose
        """
        location = np.array(position)
        _, index = self._waypoints_index.query(location)

        return index

    def get_light_state_from_camera(self):
        """ Determines the closest traffic light state from image analysis
        """
        if (not self.has_image):
            self.prev_light_loc = None
            return TrafficLight.UNKNOWN

        # format as OpenCV:
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        # preprocess:
        preprocessed_image = self.light_classifier.preprocess(cv_image)

        # predict:
        return self.light_classifier.predict(preprocessed_image[np.newaxis])

    def save_traffic_light_image(self, index, order, distance, state):
        """ Save traffic light image for offline training

        Args:
            index (Int): traffic light index
            order (str): 'before' or 'after'
            distance (Int): distance to incoming stop line
            state (TrafficLight.state): traffic light state
        """
        # format image:
        traffic_light_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        preprocessed = self.light_classifier.preprocess(traffic_light_image)
        filename = "light_classification/traffic_light_images/{}--{}-{}--{}=={}-preprocessed.jpg".format(
            rospy.Time.now().to_nsec(),
            order,
            index, 
            distance,
            state
        )
        cv2.imwrite(filename, preprocessed)

    def get_light_state_from_telegram(self, light):
        """ Determines the closest traffic light state from telegram broadcast

        Args:
            light (TrafficLight): traffic light status
        """
        return light.state

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        closest_distance = self._waypoints_size
        closest_stop_line_index = None
        closest_stop_line_waypoint_index = None

        # list of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if (self.pose) and self.waypoints:
            # ego vehicle position:
            ego_vehicle_waypoint_index = self.get_next_waypoint_index()

            # identify closest stop line:
            for i, stop_line_position in enumerate(stop_line_positions):
                stop_line_waypoint_index = self.get_closest_waypoint(stop_line_position)

                distance = stop_line_waypoint_index - ego_vehicle_waypoint_index
                if distance > 0 and distance < closest_distance:
                    closest_distance = distance
                    closest_stop_line_index = i
                    closest_stop_line_waypoint_index = stop_line_waypoint_index

        # if there is incoming stop line:
        if (
            (closest_stop_line_waypoint_index) and 
            (closest_distance <= TLDetector.CAMERA_IMAGE_CLASSIFICATION_WPS or self.after_stop_line_count > 0)
        ):
            '''
            # code for hard negative mining in simulator

            order = "before"
            # ego vehicle just passed stop line:
            if closest_distance > TLDetector.CAMERA_IMAGE_CLASSIFICATION_WPS:
                if self.after_stop_line_count > 0:
                    self.after_stop_line_count -= 1
                    order = "after"
                    closest_stop_line_index -= 1
                    closest_distance = self.after_stop_line_count
            # ego vehicle is about to cross stop line:
            elif closest_distance <= 3 and self.after_stop_line_count <= 0:
                self.after_stop_line_count = TLDetector.CAMERA_IMAGE_COLLECTION_AFTER_LINE_COUNT

            # method 01: telegram:
            state_telegram = self.get_light_state_from_telegram(self.lights[closest_stop_line_index])
            # method 02: image analysis
            state_camera = self.get_light_state_from_camera()

            # image collection:
            if state_telegram != state_camera and state_camera != TrafficLight.UNKNOWN:
                # save for hard negative mining:
                self.save_traffic_light_image(
                    closest_stop_line_index, order, closest_distance, state_telegram
                )
                # prompt:
                rospy.logwarn(
                    "[Discrepancy between Telegram and Camera]: %d--%d @ %d, Camera Image Saved",
                    state_telegram, state_camera,
                    closest_stop_line_index
                )
            '''
            state_camera = self.get_light_state_from_camera()

            return closest_stop_line_waypoint_index, state_camera
            
        '''
        # code for image collection from test site rosbag:
        
        self.save_traffic_light_image(
            0, "unknown", 0, 0
        )
        rospy.logwarn("test site image saved.")
        '''

        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')

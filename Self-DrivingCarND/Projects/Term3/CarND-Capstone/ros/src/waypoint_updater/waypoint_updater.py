#!/usr/bin/env python

import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree

import copy
import math
import numpy as np

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

class WaypointUpdater(object):
    """ update the target velocity of each waypoint based on traffic light and obstacle detection data

        @subscribed /current_pose:      the vehicle's current position
        @subscribed /base_waypoints:    the complete list of waypoints the car will be following
        @subscribed /obstacle_waypoint: the locations to stop for obstacles    
        @subscribed /traffic_waypoint:  the locations to stop for red traffic lights

        @published  /final_waypoints:   the list of waypoints ahead of the car with final target velocities   
    """
    WAYPOINT_UPDATE_FREQ = 10 # Waypoint update frequency
    LOOKAHEAD_WPS = 50       # Number of waypoints we will publish. You can change this number
    MAX_DECELERATION = 2.5     # Max deceleration

    def __init__(self):
        rospy.init_node('waypoint_updater')

        # ego vehicle pose:
        self.pose = None 
        self.base_waypoints = None
        self._base_waypoints_size = None
        self._base_waypoints_location = None
        self._base_waypoints_index = None

        # stopline waypoint:
        self.stop_line_waypoint_index = -1

        # subscribe:
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        # publish:
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.spin()

    def pose_cb(self, pose):
        """ update ego vehicle pose
        """
        self.pose = pose

    def waypoints_cb(self, waypoints):
        """ load base waypoints from system 
        """
        if not self.base_waypoints:
            # load waypoints:
            self.base_waypoints = waypoints
            # build index upon waypoints:
            self._base_waypoints_location = np.array(
                [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            )
            self._base_waypoints_size, _ = self._base_waypoints_location.shape
            self._base_waypoints_index = KDTree(
                self._base_waypoints_location
            )

    def traffic_cb(self, msg):
        # update stop line waypoint index
        self.stop_line_waypoint_index = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def get_next_waypoint_index(self):
        """ get next waypoint index 
        """
        # ego vehicle location:
        ego_vehicle_location = np.array(
            [self.pose.pose.position.x, self.pose.pose.position.y]
        )

        # closest waypoint
        _, closest_waypoint_index = self._base_waypoints_index.query(ego_vehicle_location)

        closest_waypoint_location = self._base_waypoints_location[closest_waypoint_index]
        previous_waypoint_location = self._base_waypoints_location[closest_waypoint_index - 1]

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
            next_waypoint_index = (closest_waypoint_index + 1) % self._base_waypoints_size

        return next_waypoint_index

    def generate_decelerated_waypoints(self, next_waypoints, next_waypoint_index_start):
        """ generate decelerated waypoints
        """
        ego_vehicle_stop_waypoint_index = max(self.stop_line_waypoint_index - next_waypoint_index_start - 3, 0)

        decelerated_waypoints = copy.deepcopy(next_waypoints[:ego_vehicle_stop_waypoint_index + 1])

        for i in range(ego_vehicle_stop_waypoint_index + 1):
            # ego vehicle's distance to target stop waypoint:
            distance_to_stop_line = self.distance(decelerated_waypoints, i, ego_vehicle_stop_waypoint_index)
            # in order to stop at that waypoint, velocity should be velocity ** 2 / (2 * MAX_DECELERATION) = distance_to_stop_line
            decelerated_velocity = np.sqrt(2 * WaypointUpdater.MAX_DECELERATION * distance_to_stop_line)
            
            if decelerated_velocity < 1.0:
                decelerated_velocity = 0.0
            
            decelerated_velocity = min(
                self.get_waypoint_velocity(decelerated_waypoints[i]),
                decelerated_velocity
            )

            self.set_waypoint_velocity(
                decelerated_waypoints, 
                i, 
                decelerated_velocity
            )

        return decelerated_waypoints

    def generate_next_waypoints(self, next_waypoint_index_start):
        # init:
        next_lane = Lane()

        # set header:
        next_lane.header = self.base_waypoints.header
        next_lane.header.stamp = rospy.Time(0)
        next_lane.header.frame_id = '/World'
        # set waypoints:
        next_waypoint_index_end = next_waypoint_index_start + WaypointUpdater.LOOKAHEAD_WPS

        # rospy.logwarn("[Next Waypoints]: [%d, %d]--%d",next_waypoint_index_start, next_waypoint_index_end, self.stop_line_waypoint_index)
        if self.stop_line_waypoint_index == -1 or self.stop_line_waypoint_index >= next_waypoint_index_end:
            next_lane.waypoints = self.base_waypoints.waypoints[
                next_waypoint_index_start: next_waypoint_index_end
            ]
        else:
            next_lane.waypoints = self.generate_decelerated_waypoints(
                self.base_waypoints.waypoints[
                    next_waypoint_index_start: next_waypoint_index_end
                ], 
                next_waypoint_index_start
            )

        return next_lane

    def publish_next_waypoints(self, next_waypoint_index):
        """ publish next waypoints
        """
        # generate:
        next_lane = self.generate_next_waypoints(next_waypoint_index)
        # publish:
        self.final_waypoints_pub.publish(next_lane)

    def spin(self):
        """ do spin
        """
        # update frequency:
        rate=rospy.Rate(WaypointUpdater.WAYPOINT_UPDATE_FREQ)
        # do spin:
        while not rospy.is_shutdown():
            # if both base waypoints and ego vehicle pose present
            if self.base_waypoints and self.pose:
                next_waypoint_index = self.get_next_waypoint_index()
                self.publish_next_waypoints(next_waypoint_index)
            rate.sleep()

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')

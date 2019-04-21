#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import TwistStamped
import math

from twist_controller import Controller

'''
You can build this node only after you have built (or partially built) the `waypoint_updater` node.

You will subscribe to `/twist_cmd` message which provides the proposed linear and angular velocities.
You can subscribe to any other message that you find important or refer to the document for list
of messages subscribed to by the reference implementation of this node.

One thing to keep in mind while building this node and the `twist_controller` class is the status
of `dbw_enabled`. While in the simulator, its enabled all the time, in the real car, that will
not be the case. This may cause your PID controller to accumulate error because the car could
temporarily be driven by a human instead of your controller.

We have provided two launch files with this node. Vehicle specific values (like vehicle_mass,
wheel_base) etc should not be altered in these files.

We have also provided some reference implementations for PID controller and other utility classes.
You are free to use them or build your own.

Once you have the proposed throttle, brake, and steer values, publish it on the various publishers
that we have created in the `__init__` function.

'''

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

    def __init__(self):
        rospy.init_node('dbw_node')

        # load ego vehicle params:
        vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35)
        fuel_capacity = rospy.get_param('~fuel_capacity', 13.5)
        brake_deadband = rospy.get_param('~brake_deadband', .1)
        decel_limit = rospy.get_param('~decel_limit', -5)
        accel_limit = rospy.get_param('~accel_limit', 1.)
        wheel_radius = rospy.get_param('~wheel_radius', 0.2413)
        wheel_base = rospy.get_param('~wheel_base', 2.8498)
        steer_ratio = rospy.get_param('~steer_ratio', 14.8)
        max_lat_accel = rospy.get_param('~max_lat_accel', 3.)
        max_steer_angle = rospy.get_param('~max_steer_angle', 8.)

        # state variables:
        self.is_dbw_enabled = None

        self.actual_longitudinal_velocity = None
        
        self.target_longitudinal_velocity = None
        self.target_angular_velocity = None

        # controller object
        self.controller = Controller(
            vehicle_mass = vehicle_mass,
            fuel_capacity = fuel_capacity,
            brake_deadband = brake_deadband,
            decel_limit = decel_limit,
            accel_limit = accel_limit,
            wheel_radius = wheel_radius,
            wheel_base = wheel_base,
            steer_ratio = steer_ratio,
            max_lat_accel = max_lat_accel,
            max_steer_angle = max_steer_angle
        )

        # subscribe:
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.dbw_enabled_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_cb)
        rospy.Subscriber('/twist_cmd', TwistStamped, self.twist_cmd_cb)

        # publish:
        self.steer_pub = rospy.Publisher(
            '/vehicle/steering_cmd',
            SteeringCmd, queue_size=1
        )
        self.throttle_pub = rospy.Publisher(
            '/vehicle/throttle_cmd',
             ThrottleCmd, queue_size=1
        )
        self.brake_pub = rospy.Publisher(
            '/vehicle/brake_cmd',
            BrakeCmd, queue_size=1
        )

        self.loop()

    def dbw_enabled_cb(self, is_dbw_enabled):
        """ update DBW activation status
        """
        self.is_dbw_enabled = is_dbw_enabled

    def current_velocity_cb(self, current_velocity):
        """ update waypoint velocities
        """
        self.actual_longitudinal_velocity = current_velocity.twist.linear.x

    def twist_cmd_cb(self, target_velocity):
        """ update twist command
        """
        self.target_longitudinal_velocity = target_velocity.twist.linear.x
        self.target_angular_velocity = target_velocity.twist.angular.z

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

    def publish(self, throttle, brake, steer):
        """ publish drive-by-wire(DBW) control command for autonomous driving
        """
        # throttle:
        tcmd = ThrottleCmd()
        tcmd.enable = True
        tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
        tcmd.pedal_cmd = throttle
        self.throttle_pub.publish(tcmd)

        # steering:
        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer
        self.steer_pub.publish(scmd)

        # brake:
        bcmd = BrakeCmd()
        bcmd.enable = True
        bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
        bcmd.pedal_cmd = brake
        self.brake_pub.publish(bcmd)


if __name__ == '__main__':
    DBWNode()

import rospy

from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter


class Controller(object):
    """ generate target angular velocities for ego vehicle
    """
    GAS_DENSITY = 2.858
    ONE_MPH = 0.44704
    MAX_BRAKE = 400.0

    def __init__(self, *args, **kwargs):
        # velocity lowpass filter:
        (tau, ts) = (0.5, 0.02)
        self.velocity_filter = LowPassFilter(tau, ts)

        # throttle controller:
        (kp, ki, kd) = (0.3, 0.1, 0.0)
        (v_min, v_max) = (0.0, 0.2)
        self.throttle_controller = PID(
            kp = kp, ki = ki, kd = kd,
            mn = v_min, mx = v_max
        )
        self.last_timestamp = rospy.get_time()

        # brake controller:
        self.decel_limit = kwargs['decel_limit']
        self.vehicle_mass = kwargs['vehicle_mass']
        self.wheel_radius = kwargs['wheel_radius']

        # yaw controller:
        self.yaw_controller = YawController(
            wheel_base = kwargs['wheel_base'],
            steer_ratio = kwargs['steer_ratio'],
            min_speed = 0.1,
            max_lat_accel = kwargs['max_lat_accel'],
            max_steer_angle = kwargs['max_steer_angle']
        )

    def control(self, *args, **kwargs):
        if not kwargs['is_dbw_enabled']:
            self.throttle_controller.reset()
            return 0.0, 0.0, 0.0

        # Parse params:
        actual_longitudinal_velocity = self.velocity_filter.filt(kwargs['actual_longitudinal_velocity'])
        target_longitudinal_velocity = kwargs['target_longitudinal_velocity']
        target_angular_velocity = kwargs['target_angular_velocity']

        # Longitudinal velocity error:
        longitudinal_velocity_error = target_longitudinal_velocity - actual_longitudinal_velocity

        # Time elapsed:
        current_timestamp = rospy.get_time()
        sample_time = current_timestamp - self.last_timestamp
        self.last_timestamp = current_timestamp

        # Throttle:
        throttle = self.throttle_controller.step(
            error = longitudinal_velocity_error, 
            sample_time = sample_time
        )

        # Brake:
        brake = 0.0
        # case 1: plan to stop, brake hard
        if target_longitudinal_velocity == 0.0 and actual_longitudinal_velocity < 0.1:
            throttle = 0.0
            brake = Controller.MAX_BRAKE
        # case 2: faster than target longitudinal velocity, brake gently
        elif throttle < 0.1 and longitudinal_velocity_error < 0.0:
            throttle = 0.0
            deceleration = max(self.decel_limit, longitudinal_velocity_error)
            brake = abs(deceleration) * self.vehicle_mass * self.wheel_radius

        # Steer:
        steer = self.yaw_controller.get_steering(
            current_velocity = actual_longitudinal_velocity,
            linear_velocity = target_longitudinal_velocity,
            angular_velocity = target_angular_velocity
        )

        # Return throttle, brake, steer
        return throttle, brake, steer

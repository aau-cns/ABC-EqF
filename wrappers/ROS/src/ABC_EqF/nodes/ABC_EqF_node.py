#!/usr/bin/env python3

"""Generic ROS1 Wrapper"""

__author__ = "Alessandro Fornasier"
__copyright__ = "Copyright (C) 2022 Alessandro Fornasier"
__credits__ = ["Alessandro Fornasier"]
__license__ = "GPL-3.0"
__version__ = "1.0.0"
__maintainer__ = "Alessandro Fornasier"
__email__ = "alessandro.fornasier@ieee.org"
__status__ = "Academic research"

#
# External libraries:
# - pylie : https://github.com/pvangoor/pylie
#

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)).split('/wrappers')[0])

import rospy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3Stamped, QuaternionStamped

from filter.EqF import EqF
from system.system import Input, Measurement
from utils.utils import *


class GyroBuffer:
    """Implementation of a Circular gyro buffer (FIFO)"""

    def __init__(self):

        self.__buffer = []

    def push(self, timestamp: float, u: Input):
        if not isinstance(timestamp, float):
            raise TypeError("GyroBuffer timestamp should be provided as float")
        if not isinstance(u, Input):
            raise TypeError("GyroBuffer u should be provided as Input")

        self.__buffer.append({"t": timestamp, "u": u})

        if len(self.__buffer) > 2:
            self.__buffer.pop(0)

    def ready(self):
        return len(self.__buffer) == 2

    def get(self, idx: int):
        if idx < 0 and idx > 2:
            raise ValueError("GyroBuffer idx out of bounds")
        return self.__buffer[idx]


class ABCEqFROSWrapper:

    def __init__(self):

        #####################
        # Input measurement #
        #####################

        gyro_topic = rospy.get_param("~gyro_topic")
        gyro_meas_std = rospy.get_param("~gyro_measurement_standard_deviation")
        gyro_bias_std = rospy.get_param("~gyro_bias_standard_deviation")
        self.__input_Sigma = blockDiag((gyro_meas_std ** 2) * np.eye(3), (gyro_bias_std ** 2) * np.eye(3))

        rospy.Subscriber(gyro_topic, Imu, self.callback_gyro)

        #######################
        # Output Measurements #
        #######################

        # Sensors' information
        # A list of dictionaries each including the following key-value
        # "topic": "topicname"
        # "type": "calibrated" or "uncalibrated"
        # "dir": [dir_x, dir_y, dir_Z]
        # "std": [std_X, std_y, std_z]
        # "cal": [11, 12, 13, 21, 22, 23, 31, 32 33] or []
        # "id" : id
        sensors = rospy.get_param("~sensors")

        n_cal = 0
        n_uncal = 0

        for sensor in sensors:
            if sensor["type"] == "calibrated":
                args = {k : sensor[k] for k in ("topic", "dir", "std", "cal") if k in sensor}
                rospy.Subscriber(sensor["topic"], Vector3Stamped, self.callback_calibrated_sensor, args)
                n_cal += 1
            elif sensor["type"] == "uncalibrated":
                args = {k : sensor[k] for k in ("topic" ,"dir", "std", "id") if k in sensor}
                rospy.Subscriber(sensor["topic"], Vector3Stamped, self.callback_uncalibrated_sensor, args)
                n_uncal += 1
            else:
                raise ValueError("Wrong type for sensor")

        n_sensor = n_cal + n_uncal

        delim = "--------------------------------------------------------------------------------"
        for sensor in sensors:
            print('\n' + delim)
            for key in sensor.keys():
                print(f"{key} : {sensor[key]}")
            print(delim)
        print("")


        #######
        # EqF #
        #######

        # Standard deviation for core state and calibration state
        init_stds = rospy.get_param("~initial_standard_deviations", {"core": 1.0, "cal": 1.0})

        # Define initial covariance
        S_core = blockDiag((init_stds["core"] ** 2) * np.eye(3), 0.1 * np.eye(3))
        S_cal = repBlock((init_stds["cal"] ** 2) * np.eye(3), n_uncal)
        Sigma = blockDiag(S_core, S_cal)

        # EqF
        self.__eqf = EqF(Sigma, n_uncal, n_sensor)

        ########
        # Data #
        ########

        # Gyro measurement buffer
        self.__gyro_buffer = GyroBuffer()

        #############
        # Publisher #
        #############

        self.__attitude_publisher = rospy.Publisher("attitude", QuaternionStamped, queue_size=1)


    def callback_gyro(self, msg):

        # Parse data
        t = msg.header.stamp.to_sec()
        w = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]).reshape((3, 1))
        u = Input(w, self.__input_Sigma)

        # Fill buffer
        self.__gyro_buffer.push(t, u)

        # Propagate
        if self.__gyro_buffer.ready():
            dt = self.__gyro_buffer.get(1)["t"] - self.__gyro_buffer.get(0)["t"]
            try:
                self.__eqf.propagation(self.__gyro_buffer.get(0)["u"], dt)
                self.__publish_attitude__(t)
            except:
                print('EqF propagation Error\n')


    def callback_calibrated_sensor(self, msg, args):

        # Get measurement
        t = msg.header.stamp.to_sec()
        y = np.array([msg.vector.x, msg.vector.y, msg.vector.z]).reshape((3, 1))

        # Get direction
        d = np.asarray(args["dir"]).reshape((3, 1))

        # Apply known calibrations
        S = np.asarray(args["cal"]).reshape((3, 3))
        y = S @ y

        # Get covariance
        R = np.diag([x ** 2 for x in args["std"]])

        # Update
        try:
            self.__eqf.update(Measurement(y, d, R, -1))
            self.__publish_attitude__(t)
        except:
            print('EqF update Error\n')


    def callback_uncalibrated_sensor(self, msg, args):

        # Get measurement
        t = msg.header.stamp.to_sec()
        y = np.array([msg.vector.x, msg.vector.y, msg.vector.z]).reshape((3, 1))

        # Get direction
        d = np.asarray(args["dir"]).reshape((3, 1))

        # Get covariance
        R = np.diag([x ** 2 for x in args["std"]])

        # Update
        try:
            self.__eqf.update(Measurement(y, d, R, args["id"]))
            self.__publish_attitude__(t)
        except:
            print('EqF update Error\n')


    def __publish_attitude__(self, t):
        q = self.__eqf.stateEstimate().R.as_quaternion()

        msg = QuaternionStamped()
        msg.header.stamp.from_sec(t)
        msg.quaternion.x = q[0]
        msg.quaternion.y = q[1]
        msg.quaternion.z = q[2]
        msg.quaternion.w = q[3]

        self.__attitude_publisher.publish(msg)


if __name__ == "__main__":
    rospy.init_node("ABC_EqF")
    wrapper = ABCEqFROSWrapper()
    rospy.spin()

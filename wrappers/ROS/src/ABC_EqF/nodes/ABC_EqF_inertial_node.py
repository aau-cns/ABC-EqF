#!/usr/bin/env python3

"""ROS1 Wrapper for ABC EqF that uses only an inertial measurement unit"""

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

        gyro_meas_std = rospy.get_param("~gyro_measurement_standard_deviation")
        gyro_bias_std = rospy.get_param("~gyro_bias_standard_deviation")
        self.__input_Sigma = blockDiag((gyro_meas_std ** 2) * np.eye(3), (gyro_bias_std ** 2) * np.eye(3))

        #######################
        # Output Measurements #
        #######################

        acc_meas_std = rospy.get_param("~acc_measurement_standard_deviation")
        self.__norm_th = rospy.get_param("~norm_threshold")
        self.__output_Sigma = (acc_meas_std ** 2) * np.eye(3)

        ##########
        # Topics #
        ##########

        gyro_topic = rospy.get_param("~gyro_topic")
        acc_topic = rospy.get_param("~acc_topic")

        if gyro_topic == acc_topic:
            rospy.Subscriber(gyro_topic, Imu, self.callback_imu)
        else:
            rospy.Subscriber(gyro_topic, Imu, self.callback_gyro)
            rospy.Subscriber(acc_topic, Imu, self.callback_acc)

        #######
        # EqF #
        #######

        # Standard deviation for core states
        init_std = rospy.get_param("~initial_standard_deviation")

        # Define initial covariance
        Sigma = blockDiag((init_std ** 2) * np.eye(3), 0.01 * np.eye(3))

        # EqF
        self.__eqf = EqF(Sigma, 0, 2)

        ########
        # Data #
        ########

        # Gyro measurement buffer
        self.__gyro_buffer = GyroBuffer()

        #############
        # Publisher #
        #############

        self.__attitude_publisher = rospy.Publisher("attitude", QuaternionStamped, queue_size=1)

        #############
        # Constants #
        #############

        self.__g = 9.80665


    def process_gyro(self, msg):

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
                # self.__publish_attitude__(t)
            except:
                print('EqF propagation Error\n')


    def process_acc(self, msg):

        # Get measurement
        t = msg.header.stamp.to_sec()
        y = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]).reshape((3, 1))

        if abs(np.linalg.norm(y) - self.__g) < self.__norm_th:

            # Normalize measurement
            y = y / np.linalg.norm(y)

            # Get direction
            d = np.asarray([0, 0, 1]).reshape((3, 1))

            # Update
            try:
                self.__eqf.update(Measurement(y, d, self.__output_Sigma, -1))
                self.__publish_attitude__(t)
            except:
                print('EqF update Error\n')

    def callback_gyro(self, msg):
        self.process_gyro(msg)

    def callback_acc(self, msg):
        self.process_acc(msg)

    def callback_imu(self, msg):
        self.process_gyro(msg)
        self.process_acc(msg)

    def __publish_attitude__(self, t):

        # Get estimated quaternion
        q = self.__eqf.stateEstimate().R.as_quaternion()

        msg = QuaternionStamped()
        msg.header.stamp = rospy.Time.from_sec(t)
        msg.quaternion.x = q[0]
        msg.quaternion.y = q[1]
        msg.quaternion.z = q[2]
        msg.quaternion.w = q[3]

        # Publish
        self.__attitude_publisher.publish(msg)


if __name__ == "__main__":
    rospy.init_node("ABC_EqF")
    wrapper = ABCEqFROSWrapper()
    rospy.spin()

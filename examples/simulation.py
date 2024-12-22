#!/usr/bin/env python3

"""Run simulation"""

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
import math
from dataclasses import dataclass
import argparse
from pylie import SO3
import progressbar
import pandas as pd
import matplotlib.pyplot as plt

# Update path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)).split('/examples')[0])

from filter.EqF import EqF
from system.system import State, Input, Measurement
from scipy.spatial.transform import Rotation
from utils.utils import *


# Argument parser
parser = argparse.ArgumentParser("Load dataset(s) for Attitude Bias and Calibration EqF.")
parser.add_argument("data_path", metavar='m', help="The dataset file name or the folder name.")
parser.add_argument("--num-threads", type=int, default=1, help="number of threads for NumPy to use")
args = parser.parse_args()

@dataclass
class Data:
    """Define ground-truth, input and output data"""

    # Ground-truth state
    xi: State
    n_cal: int

    # Input measurements
    u: Input

    # Output measurements as a list of Measurement
    y: list
    n_meas: int

    # Time
    t: float
    dt: float


def readCSV(pname):
    """Read data from csv file formatted as follows:

    | -------------------------------------------------------------------------------------------- |
    | t: time                                                                                      |
    | -------------------------------------------------------------------------------------------- |
    | q_x, q_y, q_z, q_w: Quaternion representing the attitude                                     |
    | -------------------------------------------------------------------------------------------- |
    | b_x, b_y, b_z: Gyro bias                                                                     |
    | -------------------------------------------------------------------------------------------- |
    | cq_x_0, cq_yv, cq_z_0, cq_w_0: Quaternion representing the first calibration                 |
    | ...                                                                                          |
    | cq_x_n, cq_y_n, cq_z_n, cq_w_n: Quaternion representing the last calibration                 |
    | -------------------------------------------------------------------------------------------- |
    | w_x, w_y, w_z: Gyro measurements                                                             |
    | -------------------------------------------------------------------------------------------- |
    | std_w_x, std_w_x, std_w_z: Gyro measurements noise standard deviation                        |
    | -------------------------------------------------------------------------------------------- |
    | std_b_x, std_b_x, std_b_z: Gyro bias random walk standard deviation                          |
    | -------------------------------------------------------------------------------------------- |
    | y_x_0, y_y_0, y_z_0: Direction measurement in sensor frame associated to calibration state 0 |
    | ...                                                                                          |
    | y_x_n, y_y_n, y_z_n: Direction measurement in sensor frame associated to calibration state n |
    | y_x_n+1, y_y_n+1, y_z_n+1: Direction measurement in sensor frame from calibrated sensor      |
    | ...                                                                                          |
    | y_x_m, y_y_m, y_z_m: Direction measurement in sensor frame from calibrated sensor            |
    | -------------------------------------------------------------------------------------------- |
    | std_y_x_0, std_y_y_0, std_y_z_0: Standard deviation of direction measurement y_0             |
    | ...                                                                                          |
    | std_y_x_m, std_y_y_m, std_y_z_m: Standard deviation of direction measurement y_m             |
    | -------------------------------------------------------------------------------------------- |
    | d_x_0, d_y_0, d_z_0: Known direction in global frame associated to direction measurement 0   |
    | ...                                                                                          |
    | d_x_m, d_y_m, d_z_m: Known direction in global frame associated to direction measurement m   |
    | -------------------------------------------------------------------------------------------- |

    NaN cell means that value is not present at that time

    Max allowd n = 5
    Max allowd m = 10

    :param pname: path name
    """

    # read .csv file into pandas dataframe
    df = pd.read_csv(pname)
    df = df.reset_index()

    # Define data_list as list
    data_list = []
    last_timestamp = df.t[0]

    # Check for existence of bias ground-truth into loaded data
    bias_exist = False
    if {'b_x', 'b_y', 'b_z'}.issubset(df.columns):
        bias_exist = True

    # Check for existence of calibration ground-truth (yaw, pitch, roll angles) into loaded data
    cal_exist = False
    n_cal = 0
    for i in range(6):
        if {'cq_x_' + str(i), 'cq_y_' + str(i), 'cq_z_' + str(i), 'cq_w_' + str(i)}.issubset(df.columns):
            cal_exist = True
            n_cal = i+1

    # Check for existence of direction measurements
    n_meas = 0
    for i in range(11):
        if {'y_x_' + str(i), 'y_y_' + str(i), 'y_z_' + str(i)}.issubset(df.columns):
            n_meas = i + 1

    for index, row in df.iterrows():

        # Load timestamps and record dt
        t = float(row['t'])
        dt = t - last_timestamp

        # Skip data_list if dt is smaller than a micro second
        if dt < 1e-6:
            continue

        last_timestamp = t

        # Load groundtruth values
        quat = np.array([float(row['q_x']), float(row['q_y']), float(row['q_z']), float(row['q_w'])])
        R = SO3.from_matrix(Rotation.from_quat(quat).as_matrix())

        # Load Gyro biases
        if bias_exist:
            b = np.array([float(row['b_x']), float(row['b_y']), float(row['b_z'])]).reshape((3, 1))
        else:
            b = np.zeros(3)

        # Load GNSS calibration
        S = []
        if cal_exist:
            for i in range(n_cal):
                cal = np.array([float(row['cq_x_' + str(i)]), float(row['cq_y_' + str(i)]), float(row['cq_z_' + str(i)]), float(row['cq_w_' + str(i)])])
                S.append(SO3.from_matrix(Rotation.from_quat(cal).as_matrix()))

        # Load Gyro inputs
        w = np.array([float(row['w_x']), float(row['w_y']), float(row['w_z'])]).reshape((3, 1))
        std_w = np.array([float(row['std_w_x']), float(row['std_w_y']), float(row['std_w_z'])]).reshape((3, 1))
        std_b = np.array([float(row['std_b_x']), float(row['std_b_y']), float(row['std_b_z'])]).reshape((3, 1))
        Sigma_wb = blockDiag(np.eye(3) * (std_w ** 2), np.eye(3) * (std_b ** 2))

        # Load measurements
        meas = []
        for i in range(n_meas):
            y = np.array([float(row['y_x_' + str(i)]), float(row['y_y_' + str(i)]), float(row['y_z_' + str(i)])]).reshape((3, 1))
            d = np.array([float(row['d_x_' + str(i)]), float(row['d_y_' + str(i)]), float(row['d_z_' + str(i)])]).reshape((3, 1))
            std_y = np.array([float(row['std_y_x_' + str(i)]), float(row['std_y_y_' + str(i)]), float(row['std_y_z_' + str(i)])]).reshape((3, 1))
            if i < n_cal:
                meas.append(Measurement(y, d, np.eye(3) * (std_y ** 2), i))
            else:
                meas.append(Measurement(y, d, np.eye(3) * (std_y ** 2), -1))

        # Append to data_list
        d = Data(State(R, b, S), n_cal, Input(w, Sigma_wb), meas, n_meas, t, dt)
        data_list.append(d)

    return data_list


def sim(filter_args, data):

    # Define progressbar
    p = progressbar.ProgressBar()

    # EqF
    filter = EqF(*filter_args)

    # Allocate variables
    t = []
    est = []

    # Filter loop
    for d in p(data):

        t.append(d.t)

        # Run filter
        try:
            filter.propagation(d.u, d.dt)
        except:
            print('Filter.propagation Error\n')
        for y in d.y:
            if not (np.isnan(np.linalg.norm(y.y.d)) or np.isnan(np.linalg.norm(y.d.d))):
                try:
                    filter.update(y)
                except:
                    print('Filter.update Error\n')
        est.append(filter.stateEstimate())

    # Plot Attitude
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
    ax = [ax0, ax1, ax2]
    for i in range(3):
        ax[i].plot(t, [d.xi.R.as_euler()[i] * 180 / math.pi for d in data], color="C0")
        ax[i].plot(t, [xi.R.as_euler()[i] * 180 / math.pi for xi in est], color="C1")
        ax[i].set_xlabel("t")
    ax0.set_title("Attitude: Roll")
    ax1.set_title("Attitude: Pitch")
    ax2.set_title("Attitude: Yaw")
    plt.show()

    # Plot bias
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
    ax = [ax0, ax1, ax2]
    for i in range(3):
        ax[i].plot(t, [d.xi.b[i] for d in data], color="C0")
        ax[i].plot(t, [xi.b[i] for xi in est], color="C1")
        ax[i].set_xlabel("t")
    ax0.set_title("Bias: x")
    ax1.set_title("Bias: y")
    ax2.set_title("Bias: z")
    plt.show()

    # Plot calibration states
    for j in range(data[0].n_cal):
        fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
        ax = [ax0, ax1, ax2]
        for i in range(3):
            ax[i].plot(t, [d.xi.S[j].as_euler()[i] * 180 / math.pi for d in data], color="C0")
            ax[i].plot(t, [xi.S[j].as_euler()[i] * 180 / math.pi for xi in est], color="C1")
            ax[i].set_xlabel("t")
        ax0.set_title("Calibration: Roll")
        ax1.set_title("Calibration: Pitch")
        ax2.set_title("Calibration: Yaw")
        plt.show()


if __name__ == '__main__':

    # Seed
    np.random.seed(0)

    # Load dataset
    print(f"Loading dataset: {args.data_path}\n")
    data = readCSV(args.data_path)

    # Define initial covariance
    Score = blockDiag(np.eye(3), 0.1 * np.eye(3))
    Scal = repBlock(np.eye(3), data[0].n_cal)
    Sigma = blockDiag(Score, Scal)

    # Run filter
    filter_args = [Sigma, data[0].n_cal, data[0].n_meas]
    sim(filter_args, data)

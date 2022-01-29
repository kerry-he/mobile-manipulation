#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import traceback

import swift
import spatialgeometry as sg
import roboticstoolbox as rtb
import spatialmath as sm
import numpy as np
import qpsolvers as qp
from itertools import product
import csv, timeit
from baseController import BaseController

pos_p = 2
pos_w = 10000

rot_p = 6
rot_w = 100

col_p = 6
col_w = 10000

considerCollisions = True



class Slack(BaseController):
    def step(self, panda, Tep, NUM_OBJECTS, n, collisions):
        global considerCollisions

        # The pose of the Panda's end-effector
        start = timeit.default_timer()

        Te = panda.fkine(panda.q)

        # Transform from the end-effector to desired pose
        eTep = Te.inv() * Tep

        # Spatial error
        e = np.sum(np.abs(np.r_[eTep.t, eTep.rpy() * np.pi / 180]))
        et = np.sum(np.abs(eTep.A[:3, -1]))

        # Calulate the required end-effector spatial velocity for the robot
        # to approach the goal. Gain is set to 1.0
        v, arrived = rtb.p_servo(Te, Tep, 1, 0.01)

        # Gain term (lambda) for control minimisation
        Y = 0.01

        # Quadratic component of objective function
        # joint angles + (position + rotation) + number of objects
        Q = np.eye(n + 6 + NUM_OBJECTS)

        # Joint velocity component of Q
        Q[:n, :n] *= Y

        # Slack component of Q
        Q[n : n + 3, n : n + 3] = pos_w * (1 / np.power(et, pos_p)) * np.eye(3)

        # dealing with rotation
        Q[n + 3 : n + 6, n + 3 : n + 6] = rot_w * (1 / (np.power(et, rot_p))) * np.eye(3)

        for j in range(2, NUM_OBJECTS):
            Q[-j, -j] = col_w * np.power(et, col_p)

        # extra weighting for the target object
        Q[-1, -1] = 100 * col_w * np.power(et, 2 + col_p)

        # The equality contraints
        Aeq = np.c_[panda.jacobe(panda.q), np.eye(6), np.zeros((6, NUM_OBJECTS))]
        beq = v.reshape((6,))

        # The inequality constraints for joint limit avoidance
        Ain = np.zeros((n + 6 + NUM_OBJECTS, n + 6 + NUM_OBJECTS))
        bin = np.zeros(n + 6 + NUM_OBJECTS)

        # The minimum angle (in radians) in which the joint is allowed to approach
        # to its limit
        ps = 0.05

        # The influence angle (in radians) in which the velocity damper
        # becomes active
        pi = 0.9
        
        # Form the joint limit velocity damper
        Ain[:n, :n], bin[:n] = panda.joint_velocity_damper(ps, pi, n)


        occluded, Ain, bin = self.calcVelocityDamper(panda, collisions, NUM_OBJECTS, n, Ain, bin)

        # Linear component of objective function: the manipulability Jacobian
        c = np.r_[-panda.jacobm(panda.q).reshape((n,)), np.zeros(6 + NUM_OBJECTS)]

        # The lower and upper bounds on the joint velocity and slack variable
        lb = -np.r_[panda.qdlim[:n], 10 * np.ones(6), np.zeros(NUM_OBJECTS)]
        ub = np.r_[panda.qdlim[:n], 10 * np.ones(6 + NUM_OBJECTS)]

        # s = timeit.default_timer()
        qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub)
        # e = timeit.default_timer()

        panda.qd[:n] = qd[:n]
        # print((setup_time - start) * 1000, (velocity_damper_time - setup_time)* 1000, (solve_time - velocity_damper_time)* 1000, (step_time - solve_time)* 1000)
        return qd, arrived, occluded

    def updateVelDamper(self, c_Ain, c_bin, Ain, bin, NUM_OBJECTS, index):
        slack_matrix = np.zeros((c_Ain.shape[0], NUM_OBJECTS))
        slack_matrix[:, index] = -np.ones((c_Ain.shape[0]))

        c_Ain = np.c_[
        c_Ain, np.zeros((c_Ain.shape[0], 6)), slack_matrix
        ]

        # Stack the inequality constraints
        Ain = np.r_[Ain, c_Ain]
        bin = np.r_[bin, c_bin]

        return Ain, bin

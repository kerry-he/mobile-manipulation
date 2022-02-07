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

pos_p = 4
pos_w = 50000

rot_p = 1
rot_w = 1

col_p = 1
col_w = 100000


PROGRAM_TIME = 0
considerCollisions = True

class Ours(BaseController):

    def changeParameters(self, _pos_p, _pos_w, _rot_p, _rot_w, _col_p, _col_w):
        global pos_p
        global pos_w
        global rot_p
        global rot_w
        global col_p
        global col_w

        pos_p = _pos_p
        pos_w = _pos_w
        rot_p = _rot_p
        rot_w = _rot_w
        col_p = _col_p
        col_w = _col_w

    def step(self, panda, Tep, NUM_OBJECTS, n, collisions, table, index):
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

        # The inequality constraints for joint limit avoidance
        N_SLACK_TERMS = 3 + 1 + NUM_OBJECTS

        # Quadratic component of objective function
        # joint angles + (position ) + number of objects + gripper constraint
        Q = np.eye(n + N_SLACK_TERMS)

        # Joint velocity component of Q
        Q[:n, :n] *= Y

        # Slack component of Q
        Q[n : n + 3, n : n + 3] = pos_w * (1 / np.power(et, pos_p)) * np.eye(3)

        # 4th is equality
        Q[n + 3 : n + 4, n + 3 : n + 4] = rot_w * (1 / (np.power(et, rot_p))) * np.eye(1)

        # make the collisions a soft constraint
        # by introducing a slack term for each of the objects
        for j in range(1,NUM_OBJECTS+1):
            Q[-j, -j] = 10000000
   
        # 3 objects, 3rd index -> -1
                    # 2nd index -> 2
                    # 1st index -> -3
        Q[-(NUM_OBJECTS - index), -(NUM_OBJECTS - index)] = col_w * np.power(et, col_p)    

        # Q[-1, -1] = et * et * et * et * et * 100

        Aeq = np.c_[panda.jacobe(panda.q)[:3], np.eye(3), np.zeros((3, 1 + NUM_OBJECTS))][:3]
        beq = v[:3].reshape((3,))

        Ain = np.zeros((n + N_SLACK_TERMS, n + N_SLACK_TERMS))
        bin = np.zeros(n + N_SLACK_TERMS)

        # The minimum angle (in radians) in which the joint is allowed to approach
        # to its limit
        ps = 0.05

        # The influence angle (in radians) in which the velocity damper
        # becomes active
        pi = 0.9
        
        # Form the joint limit velocity damper
        Ain[:n, :n], bin[:n] = panda.joint_velocity_damper(ps, pi, n)

        # Get robot gripper z axis
        gripper_angle_limit = np.deg2rad(30)

        gripper_z = panda.fkine(panda.q).A[:3, 2]
        z_axis = np.array([0, 0, 1])

        beta = np.arccos(np.dot(-gripper_z, z_axis) / (np.linalg.norm(gripper_z) * np.linalg.norm(z_axis)))

        u = np.cross(gripper_z, z_axis)

        J_cone = u.T @ panda.jacob0(panda.q)[3:]
        J_cone = J_cone.reshape((1, 7))

        damper = 1.0 * (np.cos(beta) - np.cos(gripper_angle_limit)) / (1 - np.cos(gripper_angle_limit))

        c_Ain = np.c_[J_cone, np.zeros((1, N_SLACK_TERMS))]

        Ain = np.r_[Ain, c_Ain]
        bin = np.r_[bin, damper]
        # print(beta * 180 / 3.14)

        gripper_x = panda.fkine(panda.q).A[:3, 0]
        gripper_y = panda.fkine(panda.q).A[:3, 1]
        u2 = np.cross(z_axis, gripper_y)
        gamma = np.arccos(np.dot(gripper_y, z_axis) / (np.linalg.norm(gripper_y) * np.linalg.norm(z_axis)))

        # print(gamma)

        J_face = u2.T @ panda.jacob0(panda.q)[3:]
        J_face = J_face.reshape((1, 7))

        c_Aeq = np.c_[J_face, np.zeros((1, N_SLACK_TERMS))]

        # 4th slack term
        c_Aeq[0, 7+3] = 1

        Aeq = np.r_[Aeq, c_Aeq]
        beq = np.r_[beq, np.cos(gamma)]

        occluded, Ain, bin = self.calcVelocityDamper(panda, collisions, NUM_OBJECTS, n, Ain, bin)

        c_Ain, c_bin, d_in = panda.link_collision_damper(
            table,
            panda.q[:n],
            0.25,
            0.0,
            1.0,
            start=panda.link_dict["panda_link2"], # Base is already colliding with the table
            end=panda.link_dict["panda_hand"],
        )        

        if c_Ain is not None and c_bin is not None:
            c_Ain = np.c_[c_Ain, np.zeros((c_Ain.shape[0], 4 + NUM_OBJECTS))]

            # Stack the inequality constraints
            Ain = np.r_[Ain, c_Ain]
            bin = np.r_[bin, c_bin]  

        # Linear component of objective function: the manipulability Jacobian
        c = np.r_[-panda.jacobm(panda.q).reshape((n,)), np.zeros(N_SLACK_TERMS)]

        # The lower and upper bounds on the joint velocity and slack variable
        lb = -np.r_[panda.qdlim[:n]*1/5, 10 * np.ones(3), 10 * np.ones(1), np.zeros(N_SLACK_TERMS -4)]
        # lb = -np.r_[panda.qdlim[:n], 10 * np.ones(N_SLACK_TERMS)]
        ub = np.r_[panda.qdlim[:n]*1/5, 10 * np.ones(N_SLACK_TERMS)]

        # s = timeit.default_timer()
        qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub)
        # e = timeit.default_timer()

        return qd, et < 0.02, occluded


    def updateVelDamper(self, c_Ain, c_bin, Ain, bin, NUM_OBJECTS, index):
        slack_matrix = np.zeros((c_Ain.shape[0], NUM_OBJECTS))
        slack_matrix[:, index] = -np.ones((c_Ain.shape[0]))

        c_Ain = np.c_[
            c_Ain, np.zeros((c_Ain.shape[0], 4)), slack_matrix
        ]

        # Stack the inequality constraints
        Ain = np.r_[Ain, c_Ain]
        bin = np.r_[bin, c_bin]
        return Ain, bin
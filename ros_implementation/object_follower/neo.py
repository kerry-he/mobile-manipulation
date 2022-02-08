#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import swift
import spatialgeometry as sg
import roboticstoolbox as rtb
import spatialmath as sm
import numpy as np
import qpsolvers as qp
import timeit
from baseController import BaseController


class NEO(BaseController):

    def step(self, panda, Tep, NUM_OBJECTS, n, collisions, table, index, wTcamp, wTtp):
        # The pose of the Panda's end-effector
        Te = panda.fkine(panda.q)

        # Transform from the end-effector to desired pose
        eTep = Te.inv() * Tep

        # Spatial error
        e = np.sum(np.abs(np.r_[eTep.t, eTep.rpy() * np.pi / 180]))

        # Calulate the required end-effector spatial velocity for the robot
        # to approach the goal. Gain is set to 1.0
        v, arrived = rtb.p_servo(Te, Tep, 1, 0.02)

        # Gain term (lambda) for control minimisation
        Y = 0.01

        # Quadratic component of objective function
        Q = np.eye(n + 6)

        # Joint velocity component of Q
        Q[:n, :n] *= Y

        # Slack component of Q
        Q[n:, n:] = (1 / e) * np.eye(6)

        # The equality contraints
        Aeq = np.c_[panda.jacobe(panda.q), np.eye(6)]
        beq = v.reshape((6,))

        # The inequality constraints for joint limit avoidance
        Ain = np.zeros((n + 6, n + 6))
        bin = np.zeros(n + 6)

        # The minimum angle (in radians) in which the joint is allowed to approach
        # to its limit
        ps = 0.3

        # The influence angle (in radians) in which the velocity damper
        # becomes active
        pi = 0.9

        # Form the joint limit velocity damper
        Ain[:n, :n], bin[:n] = panda.joint_velocity_damper(ps, pi, n)

        occluded, _, _ = self.calcVelocityDamper(panda, collisions, NUM_OBJECTS, n)


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
            c_Ain = np.c_[c_Ain, np.zeros((c_Ain.shape[0], 6))]

            # Stack the inequality constraints
            Ain = np.r_[Ain, c_Ain]
            bin = np.r_[bin, c_bin]  

        # Linear component of objective function: the manipulability Jacobian
        c = np.r_[-panda.jacobm(panda.q).reshape((n,)), np.zeros(6)]

        # The lower and upper bounds on the joint velocity and slack variable
        lb = -np.r_[panda.qdlim[:n]*1/5, 10 * np.ones(6)]
        ub = np.r_[panda.qdlim[:n]*1/5, 10 * np.ones(6)]

        # Solve for the joint velocities dq
        qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub)

        return qd, arrived, occluded


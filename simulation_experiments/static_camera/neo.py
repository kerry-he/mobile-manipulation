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



def step(panda, Tep, NUM_OBJECTS, n, collisions):
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
    v, arrived = rtb.p_servo(Te, Tep, 10, 0.01)

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

    # make the collisions a soft constraint
    # by introducing a slack term for each of the objects
    for j in range(NUM_OBJECTS):
        Q[-j, -j] = col_w * np.power(et, col_p)


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
    setup_time = timeit.default_timer()
    
    # Form the joint limit velocity damper
    Ain[:n, :n], bin[:n] = panda.joint_velocity_damper(ps, pi, n)

    occluded = [False] * NUM_OBJECTS
    # For each collision in the scene
    for (index, collision) in enumerate(collisions):

        # Form the velocity damper inequality contraint for each collision
        # object on the robot to the collision in the scene
        # c_start = timeit.default_timer()
        c_Ain, c_bin, d_in = panda.link_collision_damper(
            collision,
            panda.q[:n],
            0.3,
            0.2,
            1.0,
            start=panda.link_dict["panda_link1"],
            end=panda.link_dict["panda_hand"],
        )
        # c_end = timeit.default_timer()
        

        # If there are any parts of the robot within the influence distance
        # to the collision in the scene
        if c_Ain is not None and c_bin is not None and considerCollisions:
            slack_matrix = np.zeros((c_Ain.shape[0], NUM_OBJECTS))
            slack_matrix[:, index] = -np.ones((c_Ain.shape[0]))
            # print(slack_matrix)
            # input()
            c_Ain = np.c_[
                c_Ain, np.zeros((c_Ain.shape[0], 6)), slack_matrix
            ]

            # Stack the inequality constraints
            Ain = np.r_[Ain, c_Ain]
            bin = np.r_[bin, c_bin]
        # matrix_end = timeit.default_timer()

        # print((c_end - c_start) * 1000, (matrix_end - c_end)*1000)

        # print(d_in)
        if isinstance(d_in, float):
            occluded[index] = d_in < 0
        elif d_in is None:
            occluded[index] = False
        else:
            occluded[index] = min(d_in) < 0
    
    velocity_damper_time = timeit.default_timer()

    # Linear component of objective function: the manipulability Jacobian
    c = np.r_[-panda.jacobm(panda.q).reshape((n,)), np.zeros(6 + NUM_OBJECTS)]

    # The lower and upper bounds on the joint velocity and slack variable
    lb = -np.r_[panda.qdlim[:n], 10 * np.ones(6), np.zeros(NUM_OBJECTS)]
    ub = np.r_[panda.qdlim[:n], 10 * np.ones(6 + NUM_OBJECTS)]

    # s = timeit.default_timer()
    qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub)
    # e = timeit.default_timer()

    panda.qd[:n] = qd[:n]

    solve_time = timeit.default_timer()

    env.step(dt)


    step_time = timeit.default_timer()

    # print((setup_time - start) * 1000, (velocity_damper_time - setup_time)* 1000, (solve_time - velocity_damper_time)* 1000, (step_time - solve_time)* 1000)
    return arrived, occluded

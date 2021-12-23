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



def transform_between_vectors(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    angle = np.arccos(np.dot(a, b))
    axis = np.cross(a, b)

    return sm.SE3.AngleAxis(angle, axis), angle, axis


# Launch the simulator Swift
env = swift.Swift()
env.launch(realtime=False, headless=False)

# Create a Panda robot object
panda = rtb.models.Panda()

# Set joint angles to ready configuration
panda.q = panda.qr

_total = []
_totalSeen = []

# Number of joint in the panda which we are controlling
n = 7

dt = 0.01

# Make two obstacles with velocities
camera_pos = np.array([0.5, 0.5, 1.0])

np.random.seed(12345)
success = 0

considerCollisions = True

NUM_OBJECTS = 8

collisions = []
spheres = []



for i in range(NUM_OBJECTS):
    angle = np.random.uniform() * np.pi * 2
    radius = np.random.uniform() / 4 + 0.25
    target_pos = np.array([radius * np.cos(angle), radius * np.sin(angle), 0])

    middle = (camera_pos + target_pos) / 2
    R, _, _ = transform_between_vectors(
        np.array([0.0, 0.0, 1.0]), camera_pos - target_pos
    )
    # print(sm.SE3(middle) * R)

    # line of sight between camera and object we want to avoid
    s0 = sg.Cylinder(
        radius=0.001,
        length=2, #np.linalg.norm(camera_pos - target_pos),
        base=sm.SE3(middle) * R,
    )
    collisions.append(s0)

    # Make a target
    target = sg.Sphere(
        radius=0.02, base=sm.SE3(*target_pos)
    )
    spheres.append(target)

    env.add(s0)
    env.add(target)

env.add(panda)

for i in range(30):
    panda.q = panda.qr

    for k in range(NUM_OBJECTS):
        angle = np.random.uniform() * np.pi * 2
        radius = np.random.uniform() / 4 + 0.25
        target_pos = np.array([radius * np.cos(angle), radius * np.sin(angle), 0])

        middle = (camera_pos + target_pos) / 2
        R, _, _ = transform_between_vectors(
            np.array([0.0, 0.0, 1.0]), camera_pos - target_pos
        )

        collisions[k]._base = (sm.SE3(middle) * R).A
        spheres[k]._base = sm.SE3(*target_pos).A

        # collisions[i].length = np.linalg.norm(camera_pos - target_pos)

    # Add the Panda and shapes to the simulator
    # env.add(panda)    
    # env.add(target)

    # Set the desired end-effector pose to the location of target
    Tep = panda.fkine(panda.q)
    Tep.A[:3, 3] = target.base.t
    # Tep.A[2, 3] += 0.1

    total_seen = 0
    total = 0
    env.step()

    def step():
        # The pose of the Panda's end-effector
        Te = panda.fkine(panda.q)

        # Transform from the end-effector to desired pose
        eTep = Te.inv() * Tep

        # Spatial error
        e = np.sum(np.abs(np.r_[eTep.t, eTep.rpy() * np.pi / 180]))
        et = np.sum(np.abs(eTep.A[:3, -1]))

        # Calulate the required end-effector spatial velocity for the robot
        # to approach the goal. Gain is set to 1.0
        v, arrived = rtb.p_servo(Te, Tep, 5, 0.01)

        # Gain term (lambda) for control minimisation
        Y = 0.01

        # Quadratic component of objective function
        # joint angles + (position + rotation) + number of objects
        Q = np.eye(n + 6 + NUM_OBJECTS)

        # Joint velocity component of Q
        Q[:n, :n] *= Y

        # Slack component of Q
        Q[n : n + 3, n : n + 3] = 10000 * (1 / np.power(et, 3)) * np.eye(3)

        # dealing with rotation
        Q[n + 3 : n + 6, n + 3 : n + 6] = (1 / (np.power(et, 6))) * np.eye(3)

        # make the collisions a soft constraint
        # by introducing a slack term for each of the objects
        for j in range(NUM_OBJECTS):
            Q[-j, -j] = 100000 * np.power(et, 3)


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

        occluded = [False] * NUM_OBJECTS
        # For each collision in the scene
        for (index, collision) in enumerate(collisions):

            # Form the velocity damper inequality contraint for each collision
            # object on the robot to the collision in the scene
            c_Ain, c_bin, d_in = panda.link_collision_damper(
                collision,
                panda.q[:n],
                0.3,
                0.2,
                1.0,
                start=panda.link_dict["panda_link1"],
                end=panda.link_dict["panda_hand"],
            )

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

            # print(d_in)
            if isinstance(d_in, float):
                occluded[index] = d_in < 0
            elif d_in is None:
                occluded[index] = False
            else:
                occluded[index] = min(d_in) < 0

        # Linear component of objective function: the manipulability Jacobian
        c = np.r_[-panda.jacobm(panda.q).reshape((n,)), np.zeros(6 + NUM_OBJECTS)]

        # The lower and upper bounds on the joint velocity and slack variable
        lb = -np.r_[panda.qdlim[:n], 10 * np.ones(6), np.zeros(NUM_OBJECTS)]
        ub = np.r_[panda.qdlim[:n], 10 * np.ones(6 + NUM_OBJECTS)]

        # print(Q.shape)
        # print(c.shape)
        # print(Ain.shape)
        # print(Ain.shape)
        # print(bin.shape)
        # print(Aeq.shape)
        # print(beq.shape)

        # print(lb.shape)
        # print(ub.shape)

        # Solve for the joint velocities dq
        qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub)
        # print(qd)
        # if qd is None:
        #     print(qd, ":(")
        #     input()
        # input
        # Apply the joint velocities to the Panda
        panda.qd[:n] = qd[:n]

        # Step the simulator by 50 ms
        env.step(dt)

        return arrived, occluded

    time = 0
    arrived = False
    number_objects_seen = 0
    while not arrived:
        try:
            arrived, occluded = step()
            total_seen += NUM_OBJECTS - sum(occluded)
            # print(occluded)
            # input()

            # total_seen += not occluded
            total += NUM_OBJECTS
            time += dt

            if time > 6:
                print("sim timed out")
                break
        except Exception as e:
            # input()
            print(traceback.format_exc())

            # print("qp could not solve", e)
            break
    # input("arrived")
    # try:
    # env.restart()
    # except:
    #     pass

    success += arrived
    try:
        print(
            f"success: {success/(i+1)}", f"{NUM_OBJECTS * total_seen / total} number objects seen on average", f"{time:.2f}s"
        )
    except:
        pass
    
    _total += [total]
    _totalSeen += [total_seen]

# print(f"average: {100 * _totalSeen / _total}%")
print(np.mean(np.divide(_totalSeen, _total)) * NUM_OBJECTS)
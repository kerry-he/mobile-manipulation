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


def transform_between_vectors(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    angle = np.arccos(np.dot(a, b))
    axis = np.cross(a, b)

    return sm.SE3.AngleAxis(angle, axis), angle, axis


# Launch the simulator Swift
env = swift.Swift()
env.launch(realtime=False, headless=True)

# Create a Panda robot object
panda = rtb.models.Panda()

# Set joint angles to ready configuration
panda.q = panda.qr

_total = 0
_totalSeen = 0

# Number of joint in the panda which we are controlling
n = 7

dt = 0.01

# Make two obstacles with velocities
camera_pos = np.array([0.5, 0.5, 1.0])

np.random.seed(12345)
success = 0

considerCollisions = True

for i in range(100):
    panda.q = panda.qr

    angle = np.random.uniform() * np.pi * 2
    radius = 0.5  # np.random.uniform() / 2 + 0.5
    target_pos = np.array([radius * np.cos(angle), radius * np.sin(angle), 0])

    middle = (camera_pos + target_pos) / 2
    R, _, _ = transform_between_vectors(
        np.array([0.0, 0.0, 1.0]), camera_pos - target_pos
    )
    # print(sm.SE3(middle) * R)

    # line of sight between camera and object we want to avoid
    s0 = sg.Cylinder(
        radius=0.001,
        length=np.linalg.norm(camera_pos - target_pos),
        base=sm.SE3(middle) * R,
    )

    collisions = [s0]

    # Make a target
    target = sg.Sphere(
        radius=0.02, base=sm.SE3(target_pos[0], target_pos[1], target_pos[2])
    )

    # Add the Panda and shapes to the simulator
    env.add(panda)
    env.add(s0)
    env.add(target)

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
        Q = np.eye(n + 7)

        # Joint velocity component of Q
        Q[:n, :n] *= Y

        # Slack component of Q
        Q[n : n + 3, n : n + 3] = 10000 * (1 / np.power(et, 3)) * np.eye(3)

        # dealing with rotation
        Q[n + 3 : -1, n + 3 : -1] = (1 / (np.power(et, 6))) * np.eye(3)

        # make the collisions a soft constraint
        Q[-1, -1] = 100000 * np.power(et, 3)

        # The equality contraints
        Aeq = np.c_[panda.jacobe(panda.q), np.eye(6), np.zeros((6, 1))]
        beq = v.reshape((6,))

        # The inequality constraints for joint limit avoidance
        Ain = np.zeros((n + 7, n + 7))
        bin = np.zeros(n + 7)

        # The minimum angle (in radians) in which the joint is allowed to approach
        # to its limit
        ps = 0.05

        # The influence angle (in radians) in which the velocity damper
        # becomes active
        pi = 0.9

        # Form the joint limit velocity damper
        Ain[:n, :n], bin[:n] = panda.joint_velocity_damper(ps, pi, n)

        occluded = False
        # For each collision in the scene
        for collision in collisions:

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
                c_Ain = np.c_[
                    c_Ain, np.zeros((c_Ain.shape[0], 6)), -np.ones((c_Ain.shape[0], 1))
                ]

                # Stack the inequality constraints
                Ain = np.r_[Ain, c_Ain]
                bin = np.r_[bin, c_bin]

            # print(d_in)
            if isinstance(d_in, float):
                occluded = occluded or d_in < 0
            elif d_in is None:
                occluded = True
            else:
                occluded = occluded or min(d_in) < 0

        # Linear component of objective function: the manipulability Jacobian
        c = np.r_[-panda.jacobm(panda.q).reshape((n,)), np.zeros(7)]

        # The lower and upper bounds on the joint velocity and slack variable
        lb = -np.r_[panda.qdlim[:n], 10 * np.ones(7)]
        ub = np.r_[panda.qdlim[:n], 10 * np.ones(7)]

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
    while not arrived:
        try:
            arrived, occluded = step()
            total_seen += not occluded
            total += 1
            time += dt

            if time > 4:
                print("sim timed out")
                break
        except Exception as e:
            # input()
            print(traceback.format_exc())

            # print("qp could not solve", e)
            break
    env.restart()

    success += arrived
    try:
        print(
            f"success: {success/(i+1)}", f"{total_seen / total * 100}%", f"{time:.2f}s"
        )
        _total += total
        _totalSeen += total_seen
    except:
        pass

print(f"average: {100 * _totalSeen / _total}%")

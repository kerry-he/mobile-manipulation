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
import cProfile

def transform_between_vectors(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    angle = np.arccos(np.dot(a, b))
    axis = np.cross(a, b)

    return sm.SE3.AngleAxis(angle, axis), angle, axis


# Launch the simulator Swift
env = swift.Swift()
env.launch()

# Create a Panda robot object
panda = rtb.models.Panda()

# Set joint angles to ready configuration
panda.q = panda.qr
panda.q[-1] = 1.5

# Number of joint in the panda which we are controlling
n = 7

# Make two obstacles with velocities
camera_pos = np.array([0.0, 0.0, 1.0])
target_pos = np.array([0.3, -0.3, 0.35])
middle = (camera_pos + target_pos) / 2
R, _, _ = transform_between_vectors(np.array([0., 0., 1.]), camera_pos - target_pos)

s0 = sg.Cylinder(
    radius=0.001, 
    length=np.linalg.norm(camera_pos - target_pos), 
    base=sm.SE3(middle) * R
)

collisions = [s0]

# Make a target
target = sg.Sphere(radius=0.02, base=sm.SE3(target_pos[0], target_pos[1], target_pos[2]))

# Add the Panda and shapes to the simulator
env.add(panda)
env.add(s0)
env.add(target)

# Set the desired end-effector pose to the location of target
Tep = panda.fkine(panda.q)
Tep.A[:3, 3] = target.base.t
# Tep.A[2, 3] += 0.1

ax_goal = sg.Axes(0.1)
env.add(ax_goal)
ax_goal._base = Tep.A


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
    v, arrived = rtb.p_servo(Te, Tep, 1.0, 0.01)

    # Gain term (lambda) for control minimisation
    Y = 0.01

    # Quadratic component of objective function
    Q = np.eye(n + 7)

    # Joint velocity component of Q
    Q[:n, :n] *= Y

    # Slack component of Q
    Q[n:n+3, n:n+3] = (1 / e) * np.eye(3)
    Q[n+3:-1, n+3:-1] = (1 / (et * et * et * et * 10000)) * np.eye(3)
    Q[-1, -1] = et * et * et * et * et * 100

    # The equality contraints
    Aeq = np.c_[panda.jacobe(panda.q), np.eye(6), np.zeros((6, 1))][:3]
    beq = v[:3].reshape((3,))

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

    # For each collision in the scene
    for collision in collisions:

        # Form the velocity damper inequality contraint for each collision
        # object on the robot to the collision in the scene
        c_Ain, c_bin = panda.link_collision_damper(
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
        if c_Ain is not None and c_bin is not None:
            c_Ain = np.c_[c_Ain, np.zeros((c_Ain.shape[0], 6)), -np.ones((c_Ain.shape[0], 1))]

            # Stack the inequality constraints
            Ain = np.r_[Ain, c_Ain]
            bin = np.r_[bin, c_bin]


    # Get robot gripper z axis
    gripper_angle_limit = np.deg2rad(45)

    gripper_z = panda.fkine(panda.q).A[:3, 2]
    z_axis = np.array([0, 0, 1])

    beta = np.arccos(np.dot(-gripper_z, z_axis) / (np.linalg.norm(gripper_z) * np.linalg.norm(z_axis)))

    u = np.cross(gripper_z, z_axis)

    J_cone = u.T @ panda.jacob0(panda.q)[3:]
    J_cone = J_cone.reshape((1, 7))

    damper = 1.0 * (np.cos(beta) - np.cos(gripper_angle_limit)) / (1 - np.cos(gripper_angle_limit))

    c_Ain = np.c_[J_cone, np.zeros((1, 7))]
    Ain = np.r_[Ain, c_Ain]
    bin = np.r_[bin, damper]
    # print(beta * 180 / 3.14)


    gripper_x = panda.fkine(panda.q).A[:3, 0]
    gripper_y = panda.fkine(panda.q).A[:3, 1]
    gamma = np.arccos(np.dot(gripper_y, z_axis) / (np.linalg.norm(gripper_y) * np.linalg.norm(z_axis)))

    print(gamma)

    J_face = gripper_x.T @ panda.jacob0(panda.q)[3:]
    J_face = J_face.reshape((1, 7))

    c_Aeq = np.c_[J_face, np.zeros((1, 7))]
    Aeq = np.r_[Aeq, c_Aeq]
    beq = np.r_[beq, np.cos(gamma)]


    # Linear component of objective function: the manipulability Jacobian
    c = np.r_[-panda.jacobm(panda.q).reshape((n,)), np.zeros(7)]

    # The lower and upper bounds on the joint velocity and slack variable
    lb = -np.r_[panda.qdlim[:n], 10 * np.ones(7)]
    ub = np.r_[panda.qdlim[:n], 10 * np.ones(7)]

    # Solve for the joint velocities dq
    qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub)

    # Apply the joint velocities to the Panda
    panda.qd[:n] = qd[:n]

    # Step the simulator by 50 ms
    env.step(0.01)

    return arrived

arrived = False
while not arrived:
    arrived = step()


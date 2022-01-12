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
import csv



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
panda.q[-1] = 1.5

_total = []
_totalSeen = []
_time = []

# Number of joint in the panda which we are controlling
n = 7

dt = 0.01

# Make two obstacles with velocities
<<<<<<< HEAD
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
=======
camera_pos = np.array([0.5, 0.5, 1.0])


considerCollisions = True

NUM_OBJECTS = 4

collisions = []
spheres = []
all_times = []

import timeit


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

>>>>>>> d4bd07bc845b48767797c57e8e4a3a2803a67fd8
env.add(panda)


pos_power = [2,4,6]
pos_weight = [1, 100, 10000]

rot_power = [2,4,6]
rot_weight = [1,100,10000]

col_power = [2,4,6]
col_weight = [1, 100, 10000, 100000]

PROGRAM_TIME = 0


ax_goal = sg.Axes(0.1)
env.add(ax_goal)
ax_goal._base = Tep.A


for (index_so_far, (pos_p, pos_w, rot_p, rot_w, col_p, col_w)) in enumerate(product(pos_power, pos_weight, rot_power, rot_weight, col_power, col_weight)):
    print("starting")
    success = 0

    START_RUN = timeit.default_timer()
    np.random.seed(12345)

    for i in range(10):
        panda.q = panda.qr

<<<<<<< HEAD
    # Calulate the required end-effector spatial velocity for the robot
    # to approach the goal. Gain is set to 1.0
    v, arrived = rtb.p_servo(Te, Tep, 1.0, 0.01)
=======
        for k in range(NUM_OBJECTS):
            angle = np.random.uniform() * np.pi * 2
            radius = np.random.uniform() / 4 + 0.25
            target_pos = np.array([radius * np.cos(angle), radius * np.sin(angle), np.random.uniform() / 2])
>>>>>>> d4bd07bc845b48767797c57e8e4a3a2803a67fd8

            middle = (camera_pos + target_pos) / 2
            R, _, _ = transform_between_vectors(
                np.array([0.0, 0.0, 1.0]), camera_pos - target_pos
            )

            collisions[k]._base = (sm.SE3(middle) * R).A
            spheres[k]._base = sm.SE3(*target_pos).A

            # collisions[i].length = np.linalg.norm(camera_pos - target_pos)
        # print([bb._base for bb in collisions])
        # Add the Panda and shapes to the simulator
        # env.add(panda)    
        # env.add(target)

<<<<<<< HEAD
    # Slack component of Q
    Q[n:n+3, n:n+3] = (1 / e) * np.eye(3)
    Q[n+3:-1, n+3:-1] = (1 / (et * et * et * et * 10000)) * np.eye(3)
    Q[-1, -1] = et * et * et * et * et * 100

    # The equality contraints
    Aeq = np.c_[panda.jacobe(panda.q), np.eye(6), np.zeros((6, 1))][:3]
    beq = v[:3].reshape((3,))
=======
        # Set the desired end-effector pose to the location of target
        Tep = panda.fkine(panda.q)
        Tep.A[:3, 3] = target.base.t
        # Tep.A[2, 3] += 0.1

        total_seen = 0
        total = 0
        env.step()
>>>>>>> d4bd07bc845b48767797c57e8e4a3a2803a67fd8

        def step():
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

<<<<<<< HEAD

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
=======
            # dealing with rotation
            Q[n + 3 : n + 6, n + 3 : n + 6] = rot_w * (1 / (np.power(et, rot_p))) * np.eye(3)
>>>>>>> d4bd07bc845b48767797c57e8e4a3a2803a67fd8

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

        time = 0
        arrived = False
        number_objects_seen = 0
        s = timeit.default_timer()
        while not arrived:
            try:

                _s = timeit.default_timer()
                arrived, occluded = step()
                _e = timeit.default_timer()
                # print(f"Step time {1000 * (_e - _s)}ms")
                # input()
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
        e = timeit.default_timer()
        # print(e - s)

        _total += [total]
        _totalSeen += [total_seen]
        _time += [time]

        success += arrived
        # try:
        #     print(
        #         f"success: {success/(i+1)}", f"{NUM_OBJECTS * total_seen / total} number objects seen on average", f"{time:.2f}s"
        #     )
        # except:
        #     pass

    with open('avg_data.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        vision = np.divide(_totalSeen, _total)
        average_vision = np.mean(vision)
        writer.writerow([pos_p, pos_w, rot_p, rot_w, col_p, col_w, average_vision*NUM_OBJECTS, np.max(vision)*NUM_OBJECTS, np.min(vision)*NUM_OBJECTS, sum(_time), success])

    END_RUN = timeit.default_timer()

    all_times.append(END_RUN - START_RUN)

    average_time = np.mean(all_times)

    num_runs_left = 972 - (index_so_far + 1)

    print(f"Percentage Complete: {100 * (index_so_far + 1) / 972}")
    print(f"This run took: {END_RUN - START_RUN}, average time so far: {average_time}, Time left: {num_runs_left * average_time}s or {num_runs_left * average_time / 3600} hours")

#!/usr/bin/env python
"""
@author Rhys Newbury
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
import sys
import timeit
from enum import Enum
import copy
from trajectories import *
from scipy.stats import special_ortho_group


class Alg(Enum):
    Ours = 1
    Slack = 2
    NEO = 3
    PBVS = 4
    MoveIt = 5


CURRENT_ALG = Alg.Ours

if CURRENT_ALG == Alg.Ours:
    from ours import Ours as Controller
elif CURRENT_ALG == Alg.NEO:
    from neo_original import NEO as Controller
elif CURRENT_ALG == Alg.PBVS:
    from PBVS import PBVS as Controller


collisions = []
spheres = []
all_times = []


SPEEDS = [10, 20]
TRAJECTORIES = [
    ("Circle", []),
    ("Flower", []),
    ("DoubleInfinity", []),
    ("DiagonalDoubleInifinity", []),
    ("InscribedCircle", []),
    ("RadiusedRandomWalk", [100, 50.0/140, 10.0/140, 0]),
    ("RadiusedRandomWalk", [100, 50.0/140, 10.0/140, 1]),
    ("Rectangle", [200.0, 200.0]),
    ("Rectangle", [50.0, 200.0]),
    ("Rectangle", [200.0, 50.0])
]
# TRAJECTORIES.reverse()

# Launch the simulator Swift
env = swift.Swift()
env.launch(realtime=False, headless=True)

# Create a Panda robot object
panda = rtb.models.Panda()

# Set joint angles to ready configuration
panda.q = panda.qr


# Number of joint in the panda which we are controlling
n = 7

dt = 0.01

# Make two obstacles with velocities

np.random.seed(1337)

controller = Controller()

camera_pos = None


angle = None
radius = None
angular_velocity = 0.005


def move_object(pos, ee_position, collisions, spheres):

    prev_position = spheres[0]._base[:3, 3]
    (x, y, z) = prev_position
    distance = np.linalg.norm(np.subtract([x, y], ee_position[:2]))

    target_pos = [pos[0], pos[1], z]
    collisions[0].v[:3] = (target_pos - prev_position) / 0.01

    spheres[0]._base[:3, 3] = target_pos
    middle = (camera_pos + target_pos) / 2
    R, _, _ = transform_between_vectors(
        np.array([0.0, 0.0, 1.0]), camera_pos - target_pos
    )
    collisions[0]._base = (sm.SE3(middle) * R).A

    # print(radius, spheres[0]._base, prev_position)
    return spheres[0], distance
    # input()


def spawn_object(addToEnv=False):
    global camera_pos

    camera_pos = np.array(
        [
            np.random.uniform(-1, 1),
            np.random.uniform(-1, 1),
            np.random.uniform(0.5, 1.5),
        ]
    )

    # spawn objects.
    k = 0
    iterations = 0
    while k < NUM_OBJECTS:
        # for k in range(NUM_OBJECTS):
        iterations += 1
        if iterations > 100:
            camera_pos = np.array(
                [
                    np.random.uniform(-1, 1),
                    np.random.uniform(-1, 1),
                    np.random.uniform(0.5, 1.5),
                ]
            )
            k = 0
            iterations = 0

        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0.2, 0.3)
        height = np.random.uniform(0, 0.5)
        target_pos = np.array(
            [radius * np.cos(angle), radius * np.sin(angle), height])

        middle = (camera_pos + target_pos) / 2
        R, _, _ = transform_between_vectors(
            np.array([0.0, 0.0, 1.0]), camera_pos - target_pos
        )

        if addToEnv:

            # line of sight between camera and object we want to avoid
            s0 = sg.Cylinder(
                radius=0.001,
                length=3,  # np.linalg.norm(camera_pos - target_pos),
                base=sm.SE3(middle) * R,
            )
            collisions.append(s0)
            if k == NUM_OBJECTS - 1:
                color = [255, 0, 0]
            else:
                color = [0, 255, 0]
            # Make a target
            target = sg.Sphere(radius=0.02, base=sm.SE3(
                *target_pos), color=color)
            spheres.append(target)
            env.add(s0)
            env.add(target)
        else:
            collisions[k]._base = (sm.SE3(middle) * R).A
            spheres[k]._base = sm.SE3(*target_pos).A

        env.step()

        # print(k, "in collision: ", controller.isInCollision(panda, collisions[k], n))
        # input()

        if addToEnv or not controller.isInCollision(panda, collisions[k], n):
            k += 1

    return spheres[-1]


def transform_between_vectors(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    angle = np.arccos(np.dot(a, b))
    axis = np.cross(a, b)

    return sm.SE3.AngleAxis(angle, axis), angle, axis


NUM_OBJECTS = 1
print(f"************ NUM OBJECTS = {NUM_OBJECTS} *************")

env.add(panda)

spawn_object(addToEnv=True)


success = 0

_total = []
_totalSeen = []
_time = []
_manipulability = []


START_RUN = timeit.default_timer()


for i in range(len(SPEEDS) * len(TRAJECTORIES)):
    for j in range(50):
        traj_name, args = TRAJECTORIES[int(i / len(SPEEDS))]
        speed = SPEEDS[i % len(SPEEDS)]

        class_ = getattr(sys.modules[__name__], traj_name)
        if len(args) == 0:
            shape = class_()
        else:
            shape = class_(*args)

        traj = interpolate_trajectory(shape.generate(), max_distance=speed*dt)
        traj = scale_trajectory(traj, np.random.uniform(
            0.1, 0.5), np.random.uniform(0.1, 0.5))
        random_matrix = np.eye(3)
        random_matrix[:2, :2] = special_ortho_group.rvs(2)

        random_x = np.random.uniform(
            0.1, 0.3) * (-1 if np.random.uniform() > 0.5 else 1)
        random_y = np.random.uniform(
            0.1, 0.3) * (-1 if np.random.uniform() > 0.5 else 1)

        random_matrix[:2, 2] = [
            random_x, random_y]
        traj = transform_trajectory(traj, random_matrix)
        shear_matrix = np.eye(3)
        shear_matrix[0, 1] = np.random.uniform(-0.5, 0.5)
        shear_matrix[1, 0] = np.random.uniform(-0.5, 0.5)
        traj = transform_trajectory(traj, shear_matrix)

        mean_manip = []
        panda.q = panda.qr

        target = spawn_object(addToEnv=False)
        env.step()

        # Set the desired end-effector pose to the location of target
        current_pose = panda.fkine(panda.q)
        Tep = copy.deepcopy(current_pose)
        Tep.A[:3, 3] = target.base.t
        Tep.A[2, 3] = 0.6
        # Tep.A[2, 3] += 0.1

        gripper_x_desired = target.base.t - current_pose.A[:3, 3]
        gripper_x_desired = gripper_x_desired / \
            np.linalg.norm(gripper_x_desired)
        gripper_y_desired = np.cross(gripper_x_desired, [0, 0, 1])
        gripper_z_desired = [0, 0, -1]

        gripper_x_desired = np.cross(gripper_y_desired, gripper_z_desired)

        if CURRENT_ALG != Alg.Ours:
            Tep.A[:3, 0] = gripper_x_desired
            Tep.A[:3, 1] = gripper_y_desired
            Tep.A[:3, 2] = gripper_z_desired

        planned = controller.init(spheres, camera_pos, panda, Tep)

        if not planned:
            _total += [-1]
            _totalSeen += [-1]
            _time += [-1]
            continue

        total_seen = 0
        total = 0
        env.step()

        time = 0
        arrived = False
        number_objects_seen = 0

        s = timeit.default_timer()

        time_blocking = [0] * NUM_OBJECTS

        xy_distances = []
        angular_velocity = np.random.uniform(0.005, 0.005*2)

        for x, y in traj:
            try:

                _s = timeit.default_timer()

                current_pose = panda.fkine(panda.q)

                target, distance = move_object(
                    (x, y), current_pose.A[:3, 3], collisions, spheres)

                qd, arrived, occluded = controller.step(
                    panda, Tep, NUM_OBJECTS, n, collisions, camera_pos, target.base.t
                )

                Tep.A[:3, 3] = target.base.t
                Tep.A[2, 3] = 0.6
                # print("xy distance", distance)
                xy_distances.append(distance)
                panda.qd[:n] = qd[:n]

                current_dt = dt if CURRENT_ALG != Alg.MoveIt else controller.prev_timestep
                collisions[0].v[:3] = 0
                env.step(current_dt)

                _e = timeit.default_timer()

                total_seen += NUM_OBJECTS - sum(occluded)

                total += NUM_OBJECTS
                time += current_dt
                time_blocking = np.add(
                    current_dt * np.array(occluded), time_blocking)
                mean_manip.append(panda.manipulability(panda.q))

                if time > 60:
                    break
            except Exception as e:
                print(traceback.format_exc())
                break

        controller.cleanup(NUM_OBJECTS)
        _total += [total]
        _totalSeen += [total_seen]
        _time += [time]
        _manipulability.append(np.average(mean_manip))

        print("manipulability", _manipulability[-1])
        print("xy distance", np.average(xy_distances))
        print(time, time_blocking)
        print(f"Completed {100*i+j}/1000")
        # input()

        FILE = open(f"{CURRENT_ALG}_{NUM_OBJECTS}", "a")
        if CURRENT_ALG != Alg.MoveIt:
            FILE.write(
                f"{_manipulability[-1]}, {time}, {','.join([str(x) for x in time_blocking])}\n"
            )
        else:
            FILE.write(
                f"{_manipulability[-1]}, {time}, {controller.planningTime}, {','.join([str(x) for x in time_blocking])}, {xy_distances}\n"
            )
            print(
                f"{_manipulability[-1]}, {time}, {controller.planningTime}, {','.join([str(x) for x in time_blocking])}, {xy_distances} \n"
            )
        FILE.close()

        success += arrived
        panda.qd = [0] * n


vision = np.divide(_totalSeen, _total)
average_vision = np.mean(vision)

FILE = open(f"{CURRENT_ALG}_{NUM_OBJECTS}", "a")
FILE.write(f"{sum(_time)}\n")
FILE.write(f"{average_vision*NUM_OBJECTS}\n")
FILE.write(f"{np.min(vision) * NUM_OBJECTS}\n")
FILE.write(f"{np.max(vision) * NUM_OBJECTS}\n")
FILE.write(f"{success}\n")


FILE.close()

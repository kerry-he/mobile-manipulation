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
import sys
import timeit
from enum import Enum

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
elif CURRENT_ALG == Alg.Slack:
    from slack import Slack as Controller
elif CURRENT_ALG == Alg.PBVS:
    from PBVS import PBVS as Controller
elif CURRENT_ALG == Alg.MoveIt:
    from MoveIt import kerry_moveit as Controller

collisions = []
spheres = []
all_times = []


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

np.random.seed(12345)

controller = Controller()

camera_pos = None



def spawn_objects(addToEnv=False):
    global camera_pos

    camera_pos = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(0.5, 1.5)])

    # spawn objects.
    k = 0
    while k < NUM_OBJECTS:
    # for k in range(NUM_OBJECTS):

        angle = np.random.uniform() * np.pi * 2
        radius = np.random.uniform() / 4 + 0.25
        target_pos = np.array([radius * np.cos(angle), radius * np.sin(angle), 0])

        middle = (camera_pos + target_pos) / 2
        R, _, _ = transform_between_vectors(
            np.array([0.0, 0.0, 1.0]), camera_pos - target_pos
        )

        if addToEnv:

            # line of sight between camera and object we want to avoid
            s0 = sg.Cylinder(
                radius=0.001,
                length=2, #np.linalg.norm(camera_pos - target_pos),
                base=sm.SE3(middle) * R,
            )
            collisions.append(s0)
            if  k == NUM_OBJECTS - 1:
                color = [255,0,0]
            else:
                color = [0,255,0]
            # Make a target
            target = sg.Sphere(
                radius=0.02, base=sm.SE3(*target_pos), color=color
            )
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


NUM_OBJECTS = int(sys.argv[1])
print(f"************ NUM OBJECTS = {NUM_OBJECTS} *************")

env.add(panda)

spawn_objects(addToEnv=True)


success = 0

START_RUN = timeit.default_timer()

pos_power = [2,4,6]
pos_weight = [1, 100, 10000]

rot_power = [2,4,6]
rot_weight = [1,100,10000]

col_power = [2,4,6]
col_weight = [1, 100, 10000, 100000]

PROGRAM_TIME = 0

ALL_COMBOS = product(pos_power, pos_weight, rot_power, rot_weight, col_power, col_weight)
total_length = np.prod([len(l) for l in [pos_power, pos_weight, rot_power, rot_weight, col_power, col_weight]])

for (index_so_far, (pos_p, pos_w, rot_p, rot_w, col_p, col_w)) in enumerate(product(pos_power, pos_weight, rot_power, rot_weight, col_power, col_weight)):

    print("\% complete", 100 * index_so_far / total_length)
    success = 0
    _total = []
    _totalSeen = []
    _time = []
    _timeBlocking = [0] * NUM_OBJECTS
    np.random.seed(12345)
    controller.changeParameters(pos_p, pos_w, rot_p, rot_w, col_p, col_w)

    for i in range(10):
        panda.q = panda.qr

        target = spawn_objects(addToEnv=False)
        env.step()
        
        # Set the desired end-effector pose to the location of target
        Tep = panda.fkine(panda.q)
        Tep.A[:3, 3] = target.base.t
        # Tep.A[2, 3] += 0.1

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

        while not arrived:
            try:

                _s = timeit.default_timer()
                qd, arrived, occluded = controller.step(panda, Tep, NUM_OBJECTS, n, collisions)

                panda.qd[:n] = qd[:n]

                current_dt = dt if CURRENT_ALG != Alg.MoveIt else controller.prev_timestep
                env.step(current_dt)

                _e = timeit.default_timer()

                total_seen += NUM_OBJECTS - sum(occluded)

                total += NUM_OBJECTS
                time += current_dt
                time_blocking = np.add(current_dt * np.array(occluded), time_blocking)

                if time > 10:
                    break
            except Exception as e:
                print(traceback.format_exc())
                break
            
        controller.cleanup(NUM_OBJECTS)
        _total += [total]
        _totalSeen += [total_seen]
        _time += [time]
        _timeBlocking = np.add(_timeBlocking, time_blocking)
        success += arrived

    with open('avg_data.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        vision = np.divide(_totalSeen, _total)
        average_vision = np.mean(vision)
        writer.writerow([pos_p, pos_w, rot_p, rot_w, col_p, col_w, average_vision*NUM_OBJECTS, np.max(vision)*NUM_OBJECTS, np.min(vision)*NUM_OBJECTS, sum(_time), success, _timeBlocking])
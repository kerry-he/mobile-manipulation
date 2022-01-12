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

CURRENT_ALG = Alg.MoveIt

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
env.launch(realtime=True, headless=False)

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

_total = []
_totalSeen = []
_time = []


START_RUN = timeit.default_timer()


for i in range(1000):
    panda.q = panda.qr

    target = spawn_objects(addToEnv=False)
    env.step()
    
    print(target)

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

            if time > 60:
                break
        except Exception as e:
            print(traceback.format_exc())
            break
        
    controller.cleanup(NUM_OBJECTS)
    _total += [total]
    _totalSeen += [total_seen]
    _time += [time]
 
    print(time, time_blocking)
    print(f"Completed {i}/1000")
 
    FILE = open(f"{CURRENT_ALG}_{NUM_OBJECTS}", 'a')
    if CURRENT_ALG != Alg.MoveIt:
        FILE.write(f"{time}, {','.join([str(x) for x in time_blocking])}\n")
    else:
        FILE.write(f"{time}, {controller.planningTime}, {','.join([str(x) for x in time_blocking])}\n")
        print(f"{time}, {controller.planningTime}, {','.join([str(x) for x in time_blocking])}\n")
    FILE.close()
 
    success += arrived
 

vision = np.divide(_totalSeen, _total)
average_vision = np.mean(vision)

FILE = open(f"{CURRENT_ALG}_{NUM_OBJECTS}", 'a')
FILE.write(f"{sum(_time)}\n")
FILE.write(f"{average_vision*NUM_OBJECTS}\n")
FILE.write(f"{np.min(vision) * NUM_OBJECTS}\n")
FILE.write(f"{np.max(vision) * NUM_OBJECTS}\n")
FILE.write(f"{success}\n")


FILE.close()

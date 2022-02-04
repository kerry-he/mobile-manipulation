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
from baseController import BaseController

class PBVS(BaseController):
    def step(self, panda, Tep, NUM_OBJECTS, n, collisions):

        # Work out the required end-effector velocity to go towards the goal
        v, arrived = rtb.p_servo(panda.fkine(panda.q), Tep, 1, 0.01)

        # Set the Panda's joint velocities
        qd = np.linalg.pinv(panda.jacobe(panda.q)) @ v

        occluded, _, _ = self.calcVelocityDamper(panda, collisions, NUM_OBJECTS, n)

        return qd, arrived, occluded

"""
@author Rhys Newbury
"""
import numpy as np


class BaseController():
    def __init__(self):
        pass

    def init(self, spheres, camera_pose, panda, Tep):
        return True

    def step(self, panda, Tep, NUM_OBJECTS, n, collisions):
        pass

    def cleanup(self, NUM_OBJECTS):
        pass

    def isInCollision(self, panda, collision, n):
        c_Ain, c_bin, d_in = panda.link_collision_damper(
            collision,
            panda.q[:n],
            0.3,
            0.2,
            1.0,
            start=panda.link_dict["panda_link1"],
            end=panda.link_dict["panda_hand"],
        )
        if isinstance(d_in, float):
            return d_in < 0
        elif d_in is None:
            return False
        else:
            return min(d_in) < 0

    def calcVelocityDamper(self, panda, collisions, NUM_OBJECTS, n, Ain=None, bin=None):

        updateDamper = Ain is not None and bin is not None

        occluded = [False] * NUM_OBJECTS
        for (index, collision) in enumerate(collisions):

            # Form the velocity damper inequality contraint for each collision
            # object on the robot to the collision in the scene
            # c_start = timeit.default_timer()
            c_Ain, c_bin, d_in = panda.link_collision_damper(
                collision,
                panda.q[:n],
                0.3,
                0.05,
                1.0,
                start=panda.link_dict["panda_link1"],
                end=panda.link_dict["panda_hand"],
            )
            # c_end = timeit.default_timer()

            if updateDamper and c_Ain is not None and c_bin is not None:
                Ain, bin = self.updateVelDamper(
                    c_Ain, c_bin, Ain, bin, NUM_OBJECTS, index)

            # print(d_in)
            if isinstance(d_in, float):
                occluded[index] = d_in < 0
            elif d_in is None:
                occluded[index] = False
            else:
                occluded[index] = min(d_in) < 0
        return occluded, Ain, bin

    def updateVelDamper(self, c_Ain, c_bin, Ain, bin, NUM_OBJECTS, index):
        return

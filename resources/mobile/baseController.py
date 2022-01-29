import numpy as np
import spatialmath as sm
from typing import Tuple, list

class BaseController():
    def step_robot(self, r, r_cam, Tep, line_of_sight) -> Tuple[bool, list[float], list[float]]:
        pass
    
    @staticmethod
    def transform_between_vectors(a, b):
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)

        angle = np.arccos(np.dot(a, b))
        axis = np.cross(a, b)

        return sm.SE3.AngleAxis(angle, axis), angle, axis
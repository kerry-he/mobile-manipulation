
"""
@author Kerry He, Rhys Newbury
Base on: Jesse Haviland
"""
from baseController import BaseController
import numpy as np
import math
import roboticstoolbox as rtb
import spatialgeometry as sg
import spatialmath as sm
import qpsolvers as qp


class PCamera(BaseController):
    def step(self, r, r_cam, Tep, centroid_sight):

        wTe = r.fkine(r.q, fast=True)

        eTep = np.linalg.inv(wTe) @ Tep

        # Spatial error
        et = np.sum(np.abs(eTep[:3, -1]))

        # Gain term (lambda) for control minimisation
        Y = 0.01

        # Quadratic component of objective function
        Q = np.eye(r.n + 6)

        # Joint velocity component of Q
        Q[: r.n, : r.n] *= Y
        Q[:2, :2] *= 1.0 / et
        # Q[-4, -4] *= 1000
        Q[-6: -3, -6: -3] = (1000.0 / np.power(et, 2)) * np.eye(3)        # Slack manipulator
        Q[-3:, -3:] = (Y / np.power(et, 5)) * np.eye(3)   # Slack manipulator        

        # Slack component of Q
        Q[r.n:, r.n:] = (1.0 / et) * np.eye(6)

        v, _ = rtb.p_servo(wTe, Tep, 1.5)

        v[3:] *= 1.3

        # The equality contraints
        Aeq = np.c_[r.jacobe(r.q, fast=True), np.eye(6)]
        beq = v.reshape((6,))

        # The inequality constraints for joint limit avoidance
        Ain = np.zeros((r.n + 6, r.n + 6))
        bin = np.zeros(r.n + 6)

        # The minimum angle (in radians) in which the joint is allowed to approach
        # to its limit
        ps = 0.1

        # The influence angle (in radians) in which the velocity damper
        # becomes active
        pi = 0.9

        # Form the joint limit velocity damper
        Ain[: r.n, : r.n], bin[: r.n] = r.joint_velocity_damper(ps, pi, r.n)

        Ain_torso, bin_torso = r.joint_velocity_damper(0.0, 0.05, r.n)
        Ain[2, 2] = Ain_torso[2, 2]
        bin[2] = bin_torso[2]

        # Linear component of objective function: the manipulability Jacobian
        c = np.concatenate(
            (np.zeros(2), -
             r.jacobm(start=r.links[3]).reshape((r.n - 2,)), np.zeros(6))
        )

        # Get base to face end-effector
        k?? = 0.5
        bTe = r.fkine(r.q, include_base=False, fast=True)
        ???? = math.atan2(bTe[1, -1], bTe[0, -1])
        ?? = k?? * ????
        c[0] = -??

        # The lower and upper bounds on the joint velocity and slack variable
        lb = -np.r_[r.qdlim[: r.n], 10 * np.ones(6)]
        ub = np.r_[r.qdlim[: r.n], 10 * np.ones(6)]

        # Solve for the joint velocities dq
        qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub)
        qd = qd[: r.n]

        # Simple camera PID
        wTc = r_cam.fkine(r_cam.q, fast=True)
        cTep = np.linalg.inv(wTc) @ Tep

        # Spatial error
        head_rotation, head_angle, _ = BaseController.transform_between_vectors(
            np.array([1, 0, 0]), cTep[:3, 3])

        yaw = max(min(head_rotation.rpy()[
                  2] * 50, r_cam.qdlim[3]), -r_cam.qdlim[3])
        pitch = max(
            min(head_rotation.rpy()[1] * 50, r_cam.qdlim[4]), -r_cam.qdlim[4])

        qd_cam = np.r_[qd[:3], yaw, pitch]

        if et > 0.5:
            qd *= 0.7 / et
            qd_cam *= 0.7 / et
        else:
            qd *= 1.4
            qd_cam *= 1.4

        if et < 0.02:
            return True, qd, qd_cam
        else:
            return False, qd, qd_cam

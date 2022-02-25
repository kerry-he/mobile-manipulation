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


class Proposed(BaseController):
    def step(self, r, r_cam, Tep, line_of_sight):

        wTe = r.fkine(r.q, fast=True)
        wTc = r_cam.fkine(r_cam.q, fast=True)

        eTep = np.linalg.inv(wTe) @ Tep
        cTep = np.linalg.inv(wTc) @ Tep

        # Spatial error
        et = np.sum(np.abs(eTep[:3, -1]))

        head_rotation, head_angle, _ = BaseController.transform_between_vectors(np.array([1, 0, 0]), cTep[:3, 3])

        # Gain term (lambda) for control minimisation
        Y = 0.01

        # Quadratic component of objective function
        Q = np.eye(r.n + 2 + 10)

        Q[: r.n, : r.n] *= Y                                        # Robotic manipulator
        Q[:2, :2] *= 1.0 / et                                       # Mobile base
        Q[r.n : r.n + 2, r.n : r.n + 2] *= Y                        # Camera
        Q[r.n + 2 : -7, r.n + 2 : -7] = (1000.0 / np.power(et, 2)) * np.eye(3)      # Slack manipulator
        Q[r.n + 5 : -4, r.n + 5 : -4] = (Y / np.power(et, 5)) * np.eye(3)      # Slack manipulator
        Q[-4:-1, -4:-1] = 100 * np.eye(3)                                # Slack camera
        Q[-1, -1] = 1000.0 * np.power(et, 3)
 

        v_manip, _ = rtb.p_servo(wTe, Tep, 1.5)
        v_manip[3:] *= 1.3

        v_camera, _ = rtb.p_servo(sm.SE3(), head_rotation, 20)
        v_camera *= 1.3

        # The equality contraints
        Aeq = np.c_[r.jacobe(r.q, fast=True), np.zeros((6, 2)), np.eye(6), np.zeros((6, 4))]
        beq = v_manip.reshape((6,))

        jacobe_cam = r_cam.jacobe(r_cam.q, fast=True)[3:, :]
        Aeq = np.r_[Aeq, np.c_[jacobe_cam[:, :3], np.zeros((3, 7)), jacobe_cam[:, 3:], np.zeros((3, 6)), np.eye(3), np.zeros((3, 1)),]]
        beq = np.r_[beq, v_camera[3:].reshape((3,))]

        # The inequality constraints for joint limit avoidance
        Ain = np.zeros((r.n + 2 + 10, r.n + 2 + 10))
        bin = np.zeros(r.n + 2 + 10)

        # The minimum angle (in radians) in which the joint is allowed to approach
        # to its limit
        ps = 0.1

        # The influence angle (in radians) in which the velocity damper
        # becomes active
        pi = 0.9

        # Form the joint limit velocity damper
        Ain[: r.n, : r.n], bin[: r.n] = r.joint_velocity_damper(ps, pi, r.n)

        Ain_torso, bin_torso = r_cam.joint_velocity_damper(0.0, 0.05, r_cam.n)
        Ain[2, 2] = Ain_torso[2, 2]
        bin[2] = bin_torso[2]

        Ain_cam, bin_cam = r_cam.joint_velocity_damper(ps, pi, r_cam.n)
        Ain[r.n : r.n + 2, r.n : r.n + 2] = Ain_cam[3:, 3:]
        bin[r.n : r.n + 2] = bin_cam[3:]


        # Draw line of sight between camera and object
        camera_pos = wTc[:3, 3]
        target_pos = Tep[:3, 3]
        middle = (camera_pos + target_pos) / 2
        R, _, _ = BaseController.transform_between_vectors(np.array([0., 0., 1.]), camera_pos - target_pos)

        los = sg.Cylinder(radius=0.001, 
                        length=np.linalg.norm(camera_pos - target_pos), 
                        base=(sm.SE3(middle) * R)
        )

        c_Ain, c_bin, _, _ = r.new_vision_collision_damper(
            los,
            r.q[:r.n],
            0.3,
            0.2,
            1.0,
            start=r.link_dict["shoulder_pan_link"],
            end=r.link_dict["gripper_link"],
            camera=r_cam,
            obj=Tep[:3, 3]
        )


        if c_Ain is not None and c_bin is not None:
            Ain = np.r_[Ain, c_Ain]
            bin = np.r_[bin, c_bin]

        # Linear component of objective function: the manipulability Jacobian
        c = np.concatenate(
            (np.zeros(2), -r.jacobm(start=r.links[3]).reshape((r.n - 2,)), np.zeros(2), np.zeros(10))
        )

        # Get base to face end-effector
        kε = 0.5
        bTe = r.fkine(r.q, include_base=False, fast=True)
        θε = math.atan2(bTe[1, -1], bTe[0, -1])
        ε = kε * θε
        c[0] = -ε

        # The lower and upper bounds on the joint velocity and slack variable
        lb = -np.r_[r.qdlim[: r.n], r_cam.qdlim[3:r_cam.n], 100 * np.ones(9), 0]
        ub = np.r_[r.qdlim[: r.n], r_cam.qdlim[3:r_cam.n], 100 * np.ones(9), 100]

        # Solve for the joint velocities dq
        qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub)
        qd_cam = np.concatenate((qd[:3], qd[r.n : r.n + 2]))
        qd = qd[: r.n]

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

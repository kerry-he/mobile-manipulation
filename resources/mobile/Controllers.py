#!/usr/bin/env python3
import numpy as np

import roboticstoolbox as rtb
import spatialgeometry as sg
import spatialmath as sm
import qpsolvers as qp
import numpy as np
import math

def transform_between_vectors(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    angle = np.arccos(np.dot(a, b))
    axis = np.cross(a, b)

    return sm.SE3.AngleAxis(angle, axis), angle, axis


def step_robot(r, r_cam, Tep, line_of_sight):

    wTe = r.fkine(r.q, fast=True)
    wTc = r_cam.fkine(r_cam.q, fast=True)

    eTep = np.linalg.inv(wTe) @ Tep
    cTep = np.linalg.inv(wTc) @ Tep

    # Spatial error
    et = np.sum(np.abs(eTep[:3, -1]))

    head_rotation, head_angle, _ = transform_between_vectors(np.array([1, 0, 0]), cTep[:3, 3])

    # Gain term (lambda) for control minimisation
    Y = 0.01

    # Quadratic component of objective function
    Q = np.eye(r.n + 2 + 10)

    Q[: r.n, : r.n] *= Y                                        # Robotic manipulator
    Q[:2, :2] *= 1.0 / et                                       # Mobile base
    Q[r.n : r.n + 2, r.n : r.n + 2] *= Y                        # Camera
    Q[r.n + 2 : -7, r.n + 2 : -7] = (10000.0 / np.power(et, 2)) * np.eye(3)      # Slack manipulator
    Q[r.n + 5 : -4, r.n + 5 : -4] = (1.0 / np.power(et, 5)) * np.eye(3)      # Slack manipulator
    Q[-4:-1, -4:-1] = 100.0 * np.eye(3)                                # Slack camera
    Q[-1, -1] = 100000.0 * np.power(et, 3)

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
    R, _, _ = transform_between_vectors(np.array([0., 0., 1.]), camera_pos - target_pos)

    los = sg.Cylinder(radius=0.001, 
                    length=np.linalg.norm(camera_pos - target_pos), 
                    base=(sm.SE3(middle) * R)
    )

    line_of_sight._length = np.linalg.norm(camera_pos - target_pos)
    line_of_sight._base = (sm.SE3(middle) * R).A

    c_Ain, c_bin, _ = r.new_vision_collision_damper(
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


def step_proposed(r, r_cam, Tep, centroid_sight):

    wTe = r.fkine(r.q, fast=True)
    wTc = r_cam.fkine(r_cam.q, fast=True)

    eTep = np.linalg.inv(wTe) @ Tep
    cTep = np.linalg.inv(wTc) @ Tep

    # Spatial error
    et = np.sum(np.abs(eTep[:3, -1]))

    head_rotation, head_angle, _ = transform_between_vectors(np.array([1, 0, 0]), cTep[:3, 3])

    # Gain term (lambda) for control minimisation
    Y = 0.01

    # Quadratic component of objective function
    Q = np.eye(r.n + 2 + 10)

    Q[: r.n, : r.n] *= Y                                        # Robotic manipulator
    Q[:2, :2] *= 1.0 / et                                       # Mobile base
    Q[r.n : r.n + 2, r.n : r.n + 2] *= Y                        # Camera
    Q[r.n + 2 : -7, r.n + 2 : -7] = (100.0 / np.power(et, 2)) * np.eye(3)      # Slack manipulator
    Q[r.n + 5 : -4, r.n + 5 : -4] = (1.0 / np.power(et, 5)) * np.eye(3)      # Slack manipulator
    Q[-4:-1, -4:-1] = 100.0 * np.eye(3)                                # Slack camera
    Q[-1, -1] = 10000.0 * np.power(et, 5)

    v_manip, _ = rtb.p_servo(wTe, Tep, 1.5)
    v_manip[3:] *= 1.3

    v_camera, _ = rtb.p_servo(sm.SE3(), head_rotation, 20)
    v_camera *= 1.3
    # v_cam_norm = np.linalg.norm(v_camera)
    # v_camera = v_camera / np.linalg.norm(v_camera) * 5 * (1-np.exp(-v_cam_norm*10))

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
    c_Ain, c_bin, _ = r.vision_collision_damper(
        centroid_sight,
        r.q[:r.n],
        0.3,
        0.2,
        1.0,
        start=r.link_dict["shoulder_pan_link"],
        end=r.link_dict["gripper_link"],
        camera=r_cam,
    )    

    if c_Ain is not None and c_bin is not None:
        # Stack the inequality constraints
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



def step_holistic(r, r_cam, Tep):

    wTe = r.fkine(r.q, fast=True)
    wTc = r_cam.fkine(r_cam.q, fast=True)

    eTep = np.linalg.inv(wTe) @ Tep
    cTep = np.linalg.inv(wTc) @ Tep

    # Spatial error
    et = np.sum(np.abs(eTep[:3, -1]))

    head_rotation, head_angle, _ = transform_between_vectors(np.array([1, 0, 0]), cTep[:3, 3])

    # Gain term (lambda) for control minimisation
    Y = 0.01

    # Quadratic component of objective function
    Q = np.eye(r.n + 2 + 9)

    Q[: r.n, : r.n] *= Y                                        # Robotic manipulator
    Q[:2, :2] *= 1.0 / et                                       # Mobile base
    Q[r.n : r.n + 2, r.n : r.n + 2] *= Y                        # Camera
    Q[r.n + 2 : -6, r.n + 2 : -6] = (1.0 / et) * np.eye(3)      # Slack manipulator
    Q[r.n + 5 : -3, r.n + 5 : -3] = (1.0 / np.power(et, 5)) * np.eye(3)      # Slack manipulator
    Q[-7, -7] *= 1000
    Q[-3:, -3:] = 100 * np.eye(3)                      # Slack camera


    v_manip, _ = rtb.p_servo(wTe, Tep, 1.5)
    v_manip[3:] *= 1.3

    v_camera, _ = rtb.p_servo(sm.SE3(), head_rotation, 10)
    v_camera *= 2.6

    # The equality contraints
    Aeq = np.c_[r.jacobe(r.q, fast=True), np.zeros((6, 2)), np.eye(6), np.zeros((6, 3))]
    beq = v_manip.reshape((6,))

    jacobe_cam = r_cam.jacobe(r_cam.q, fast=True)[3:, :]
    Aeq = np.r_[Aeq, np.c_[jacobe_cam[:, :3], np.zeros((3, 7)), jacobe_cam[:, 3:], np.zeros((3, 6)), np.eye(3)]]
    beq = np.r_[beq, v_camera[3:].reshape((3,))]

    # The inequality constraints for joint limit avoidance
    Ain = np.zeros((r.n + 2 + 9, r.n + 2 + 9))
    bin = np.zeros(r.n + 2 + 9)

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

    # Linear component of objective function: the manipulability Jacobian
    c = np.concatenate(
        (np.zeros(2), -r.jacobm(start=r.links[3]).reshape((r.n - 2,)), np.zeros(2), np.zeros(9))
    )

    # Get base to face end-effector
    kε = 0.5
    bTe = r.fkine(r.q, include_base=False, fast=True)
    θε = math.atan2(bTe[1, -1], bTe[0, -1])
    ε = kε * θε
    c[0] = -ε

    # The lower and upper bounds on the joint velocity and slack variable
    lb = -np.r_[r.qdlim[: r.n], r_cam.qdlim[3:r_cam.n], 100 * np.ones(6), 500 * np.ones(3)]
    ub = np.r_[r.qdlim[: r.n], r_cam.qdlim[3:r_cam.n], 100 * np.ones(6), 500 * np.ones(3)]

    # Solve for the joint velocities dq
    qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub)
    
    if qd is None:
        raise ValueError('QP failed to solve')

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



def step_p_camera(r, r_cam, Tep):

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
    Q[-4, -4] *= 1000

    # Slack component of Q
    Q[r.n :, r.n :] = (1.0 / et) * np.eye(6)

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
        (np.zeros(2), -r.jacobm(start=r.links[3]).reshape((r.n - 2,)), np.zeros(6))
    )

    # Get base to face end-effector
    kε = 0.5
    bTe = r.fkine(r.q, include_base=False, fast=True)
    θε = math.atan2(bTe[1, -1], bTe[0, -1])
    ε = kε * θε
    c[0] = -ε

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
    head_rotation, head_angle, _ = transform_between_vectors(np.array([1, 0, 0]), cTep[:3, 3])

    yaw = max(min(head_rotation.rpy()[2] * 50, r_cam.qdlim[3]), -r_cam.qdlim[3])
    pitch = max(min(head_rotation.rpy()[1] * 50, r_cam.qdlim[4]), -r_cam.qdlim[4])

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


def step_separate_base(r, r_cam, Tep):

    # wTe = r.fkine(r.q, fast=True)
    # wTe = r._base.A[:] / 2 + r.fkine(r.q, fast=True) / 2
    wTe = r._base.A[:] @ sm.SE3(1.1281 * 3 / 5, 0, 0).A
    
    Tep_temp = sm.SE3(Tep[:3, 3]).A
    Tep_temp[:2, 3] -= Tep[:2, 0] * 0.2
    eTep = np.linalg.inv(wTe) @ Tep
    
    # Spatial error
    et = np.sum(np.abs(eTep[:2, -1]))

    # Gain term (lambda) for control minimisation
    Y = 0.01

    # Quadratic component of objective function
    Q = np.eye(r.n + 6)

    # Joint velocity component of Q
    Q[: r.n, : r.n] *= Y
    Q[:2, :2] *= 1.0 / et

    # Slack component of Q
    Q[r.n :, r.n :] = (1.0 / et) * np.eye(6)

    v, _ = rtb.p_servo(wTe, Tep, 1.5)

    v[2:] *= 0

    # The equality contraints
    Aeq = np.c_[r.jacobe(r.q, start="base0", end="base_link", tool=sm.SE3(1.1281, 0, 0).A, fast=True), np.zeros((6, 8)), np.eye(6)]
    beq = v.reshape((6,))

    Aeq_arm = np.c_[np.zeros((8, 2)), np.eye(8), np.zeros((8, 6))]
    beq_arm = np.zeros((8,))

    Aeq = np.r_[Aeq, Aeq_arm]
    beq = np.r_[beq, beq_arm]

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
    Ain[2:, 2:] = 0
    bin[2:] = 0

    # Linear component of objective function: the manipulability Jacobian
    # c = np.concatenate(
    #     (np.zeros(2), -r.jacobm(start=r.links[3]).reshape((r.n - 2,)), np.zeros(6))
    # )
    c = np.concatenate(
        (np.zeros(2), np.zeros(8), np.zeros(6))
    )    

    # Get base to face end-effector
    kε = 0.5
    bTe = r.fkine(r.q, include_base=False, fast=True)
    θε = math.atan2(bTe[1, -1], bTe[0, -1])
    ε = kε * θε
    c[0] = -0

    # The lower and upper bounds on the joint velocity and slack variable
    lb = -np.r_[r.qdlim[: r.n], 100 * np.ones(6)]
    ub = np.r_[r.qdlim[: r.n], 100 * np.ones(6)]


    # Simple camera PID
    wTc = r_cam.fkine(r_cam.q, fast=True)
    cTep = np.linalg.inv(wTc) @ Tep

    # Spatial error
    head_rotation, head_angle, _ = transform_between_vectors(np.array([1, 0, 0]), cTep[:3, 3])

    yaw = max(min(head_rotation.rpy()[2] * 10, r_cam.qdlim[3]), -r_cam.qdlim[3])
    pitch = max(min(head_rotation.rpy()[1] * 10, r_cam.qdlim[4]), -r_cam.qdlim[4])


    # Solve for the joint velocities dq
    qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub)
    qd = qd[: r.n]

    qd_cam = np.r_[qd[:3], yaw, pitch]

    if et > 0.5:
        qd *= 0.7 / et
        qd_cam *= 0.7 / et
    else:
        qd *= 1.4
        qd_cam *= 1.4

    if et < 0.03:
        return True, qd, qd_cam
    else:
        return False, qd, qd_cam



def step_separate_arm(r, r_cam, Tep):

    # wTe = r.fkine(r.q, fast=True)
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

    # Slack component of Q
    Q[r.n :, r.n :] = (1.0 / et) * np.eye(6)

    v, _ = rtb.p_servo(wTe, Tep, 1.5)

    v[3:] *= 1.3

    # The equality contraints
    Aeq = np.c_[r.jacobe(r.q, fast=True), np.eye(6)]
    beq = v.reshape((6,))

    # Aeq_arm = np.c_[np.eye(2), np.zeros((2, 8)), np.zeros((2, 6))]
    # beq_arm = np.zeros((2,))

    # Aeq = np.r_[Aeq, Aeq_arm]
    # beq = np.r_[beq, beq_arm]

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
        (np.zeros(2), -r.jacobm(start=r.links[3]).reshape((r.n - 2,)), np.zeros(6))
    ) * 0

    # Get base to face end-effector
    kε = 0.5
    bTe = r.fkine(r.q, include_base=False, fast=True)
    θε = math.atan2(bTe[1, -1], bTe[0, -1])
    ε = kε * θε
    c[0] = -0

    # The lower and upper bounds on the joint velocity and slack variable
    lb = -np.r_[r.qdlim[: r.n], 10 * np.ones(6)]
    ub = np.r_[r.qdlim[: r.n], 10 * np.ones(6)]


    # Simple camera PID
    wTc = r_cam.fkine(r_cam.q, fast=True)
    cTep = np.linalg.inv(wTc) @ Tep

    # Spatial error
    head_rotation, head_angle, _ = transform_between_vectors(np.array([1, 0, 0]), cTep[:3, 3])

    yaw = max(min(head_rotation.rpy()[2] * 10, r_cam.qdlim[3]), -r_cam.qdlim[3])
    pitch = max(min(head_rotation.rpy()[1] * 10, r_cam.qdlim[4]), -r_cam.qdlim[4])


    # Solve for the joint velocities dq
    qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub)
    qd = qd[: r.n]

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

import swift
import roboticstoolbox as rtb
import spatialgeometry as sg
import spatialmath as sm
import qpsolvers as qp
import numpy as np
import math
import csv


def transform_between_vectors(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    angle = np.arccos(np.dot(a, b))
    axis = np.cross(a, b)

    return sm.SE3.AngleAxis(angle, axis), angle, axis


def rand_pose():
    R, _ = np.linalg.qr(np.random.normal(size=(3, 3)))
    r = sm.SE3()
    r.A[:3, :3] = R

    t = sm.SE3((np.random.rand(1, 3) - 0.5) * [10, 10, 1] + [0, 0, 1])

    return t * r


def step_robot(r, r_cam, Tep):

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

    yaw = max(min(head_rotation.rpy()[2] * 10, r_cam.qdlim[3]), -r_cam.qdlim[3])
    pitch = max(min(head_rotation.rpy()[1] * 10, r_cam.qdlim[4]), -r_cam.qdlim[4])

    if et > 0.5:
        qd *= 0.7 / et
    else:
        qd *= 1.4

    qd_cam = np.r_[qd[:3], yaw, pitch]

    if et < 0.02:
        return True, qd, qd_cam
    else:
        return False, qd, qd_cam


def obj_in_vision(r, r_cam, Tep):

    # Check if object is in FoV
    wTc = r_cam.fkine(r_cam.q, fast=True)
    cTep = np.linalg.inv(wTc) @ Tep
    _, head_angle, _ = transform_between_vectors(np.array([1, 0, 0]), cTep[:3, 3])

    # Draw line of sight between camera and object
    camera_pos = wTc[:3, 3]
    target_pos = Tep[:3, 3]
    middle = (camera_pos + target_pos) / 2
    R, _, _ = transform_between_vectors(np.array([0., 0., 1.]), camera_pos - target_pos)
    los = sg.Cylinder(
        radius=0.001, 
        length=np.linalg.norm(camera_pos - target_pos), 
        base=sm.SE3(middle) * R
    )

    _, _, c_din = r.vision_collision_damper(
        line_of_sight,
        r.q[:r.n],
        0.3,
        0.1,
        1.0,
        start=r.link_dict["shoulder_pan_link"],
        end=r.link_dict["wrist_roll_link"],
        camera=r_cam
    )
    
    if isinstance(c_din, float):
        c_din = [c_din]

    if c_din is not None:
        print(min(c_din) > 0.05, head_angle < np.deg2rad(45/2), min(c_din) > 0.05 and head_angle < np.deg2rad(45/2))
        return min(c_din) > 0.05 and head_angle < np.deg2rad(45/2)
    else:
        return head_angle < np.deg2rad(45/2)


goal_pos_list = [
    sm.SE3([ 4.17, -0.93,  0.70]),
    sm.SE3([ 4.70, -3.89,  1.17]),
    sm.SE3([-2.82, -0.33,  1.36]),
    sm.SE3([ 1.70,  4.13,  0.61]),
    sm.SE3([-4.57, -1.94,  0.69]),
    sm.SE3([ 2.83,  1.75,  0.86]),
    sm.SE3([ 4.99, -0.25,  1.43]),
    sm.SE3([-1.84,  2.40,  1.44]),
    sm.SE3([-4.68,  4.09,  0.89]),
    sm.SE3([-0.29, -0.43,  1.30])
]

goal_orient_list = [
    sm.SE3.Eul([1.41, 0.97, 2.66]),
    sm.SE3.Eul([1.97, 0.16, 2.39]),
    sm.SE3.Eul([-1.43,  1.38, -1.98]),
    sm.SE3.Eul([0.65, 0.75, 1.35]),
    sm.SE3.Eul([-0.94,  1.48, -1.92]),
    sm.SE3.Eul([-2.41,  1.36, -1.12]),
    sm.SE3.Eul([2.77, 0.36, 1.38]),
    sm.SE3.Eul([1.70, 1.46, 1.31]),
    sm.SE3.Eul([-0.81,  1.06, -0.81]),
    sm.SE3.Eul([2.72, 0.31, 0.67])
]

env = swift.Swift()
env.launch(realtime=True)


for goal in range(10):

    ax_goal = sg.Axes(0.1)
    env.add(ax_goal)

    fetch = rtb.models.Fetch()
    fetch.q = fetch.qr
    env.add(fetch)

    fetch_camera = rtb.models.FetchCamera()
    fetch_camera.q = fetch_camera.qr
    env.add(fetch_camera)
    sight_base = sm.SE3.Ry(np.pi/2) * sm.SE3(0.0, 0.0, 2.5)
    line_of_sight = sg.Cylinder(radius=0.001, length=5.0, base=sight_base)
    line_of_sight._base = fetch_camera.fkine(fetch_camera.q, fast=True) @ sight_base.A
    env.add(line_of_sight)

    arrived = False
    dt = 0.025

    # Behind
    env.set_camera_pose([-2, 3, 0.7], [-2, 0.0, 0.5])
    wTep = goal_pos_list[goal] * goal_orient_list[goal]
    ax_goal.base = wTep

    env.step()

    total_count = 0
    seen_count = 0

    while not arrived:

        arrived, fetch.qd, fetch_camera.qd = step_robot(fetch, fetch_camera, wTep.A)
        env.step(dt)

        # Reset bases
        base_new = fetch.fkine(fetch._q, end=fetch.links[2], fast=True)
        fetch._base.A[:] = base_new
        fetch.q[:2] = 0

        base_new = fetch_camera.fkine(fetch_camera._q, end=fetch_camera.links[2], fast=True)
        fetch_camera._base.A[:] = base_new
        fetch_camera.q[:2] = 0

        line_of_sight._base = fetch_camera.fkine(fetch_camera.q, fast=True) @ sight_base.A

        total_count += 1
        seen_count += obj_in_vision(fetch, fetch_camera, wTep.A)

        if (total_count * dt) > 100:
            print("Simulation time out")
            break

    print(goal)
    print("Vision: ", seen_count / total_count * 100, "%")
    print("Time elapsed: ", total_count * dt, "s")
    print("Success: ", arrived)
    print()

    vision_pc =  seen_count / total_count * 100
    time_elapsed = total_count * dt
    is_success = arrived

    with open('data_baseline.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([goal, vision_pc, time_elapsed, is_success])

    env.restart()
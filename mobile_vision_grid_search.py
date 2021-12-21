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


def step_robot(r, r_cam, Tep, centroid_sight, w_pos, pow_pos, w_vis, pow_vis):

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
    Q[r.n + 2 : -7, r.n + 2 : -7] = (w_pos / np.power(et, pow_pos)) * np.eye(3)      # Slack manipulator
    Q[r.n + 5 : -4, r.n + 5 : -4] = (1.0 / np.power(et, 5)) * np.eye(3)      # Slack manipulator
    Q[-4:-1, -4:-1] = 100 * np.eye(3)                                # Slack camera
    Q[-1, -1] = w_vis * np.power(et, pow_vis)

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

    line_of_sight._base = (sm.SE3(middle) * R).A

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
        return min(c_din) > 0.05 and head_angle < np.deg2rad(45/2)
    else:
        return head_angle < np.deg2rad(45/2)        


# Grid search parameters
w_pos_list = [1, 10, 100, 1000]
pow_pos_list = [1, 2, 3, 4]
w_vis_list = [100, 1000, 10000, 100000]
pow_vis_list = [3, 5, 7, 9]



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

load_iter = 0
curr_iter = 0

env = swift.Swift()
env.launch(realtime=True)

for w_pos in w_pos_list:
    for pow_pos in pow_pos_list:
        for w_vis in w_vis_list:
            for pow_vis in pow_vis_list:
                vision_pc_total = 0
                time_elapsed_total = 0
                is_succes_total = 0
                max_vis = 0
                min_vis = 100

                curr_iter += 1

                if curr_iter <= load_iter:
                    continue

                for goal in range(len(goal_pos_list)):

                    ax_goal = sg.Axes(0.1)
                    env.add(ax_goal)

                    fetch = rtb.models.Fetch()
                    fetch.q = fetch.qr
                    env.add(fetch)

                    fetch_camera = rtb.models.FetchCamera()
                    fetch_camera.q = fetch_camera.qr
                    env.add(fetch_camera)

                    sight_base = sm.SE3.Ry(np.pi/2) * sm.SE3(0.0, 0.0, 2.5)
                    centroid_sight = sg.Cylinder(radius=0.001, 
                                                length=5.0, 
                                                base=fetch_camera.fkine(fetch_camera.q, fast=True) @ sight_base.A
                    )
                    env.add(centroid_sight)

                    line_of_sight = sg.Cylinder(radius=0.001, 
                                                length=5.0, 
                                                base=fetch_camera.fkine(fetch_camera.q, fast=True) @ sight_base.A
                    )
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
                        try:
                            arrived, fetch.qd, fetch_camera.qd = step_robot(fetch, fetch_camera, wTep.A, centroid_sight, w_pos, pow_pos, w_vis, pow_vis)
                        except Exception as e:
                            print(e)
                            break

                        env.step(dt)

                        # Reset bases
                        base_new = fetch.fkine(fetch._q, end=fetch.links[2], fast=True)
                        fetch._base.A[:] = base_new
                        fetch.q[:2] = 0

                        base_new = fetch_camera.fkine(fetch_camera._q, end=fetch_camera.links[2], fast=True)
                        fetch_camera._base.A[:] = base_new
                        fetch_camera.q[:2] = 0

                        centroid_sight._base = fetch_camera.fkine(fetch_camera.q, fast=True) @ sight_base.A

                        total_count += 1
                        seen_count += obj_in_vision(fetch, fetch_camera, wTep.A)

                        if (total_count * dt) > 50:
                            print("Simulation time out")
                            break

                    total_count = max(1, total_count)
                    print(w_pos, pow_pos, w_vis, pow_vis, goal)
                    print("Vision: ", seen_count / total_count * 100, "%")
                    print("Time elapsed: ", total_count * dt, "s")
                    print("Success: ", arrived)
                    print()

                    vision_pc =  seen_count / total_count * 100
                    time_elapsed = total_count * dt
                    is_success = arrived
                    max_vis = max(vision_pc, max_vis)
                    min_vis = min(vision_pc, max_vis)

                    vision_pc_total += seen_count / total_count * 100 / 10
                    time_elapsed_total += total_count * dt / 10
                    is_succes_total += arrived

                    with open('all_data.csv', 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        writer.writerow([w_pos, pow_pos, w_vis, pow_vis, goal, vision_pc, time_elapsed, is_success])

                    env.restart()
                    
                with open('avg_data.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow([w_pos, pow_pos, w_vis, pow_vis, vision_pc_total, max_vis, min_vis, time_elapsed_total, is_succes_total])
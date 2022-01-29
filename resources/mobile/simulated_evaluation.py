import swift
import roboticstoolbox as rtb
import spatialgeometry as sg
import spatialmath as sm
import qpsolvers as qp
import numpy as np
import math
import csv

from Controllers import *

np.random.seed(1337)


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
        end=r.link_dict["gripper_link"],
        camera=r_cam
    )
    
    if isinstance(c_din, float):
        c_din = [c_din]

    if c_din is not None:
        return min(c_din) > 0.0 and head_angle < np.deg2rad(45/2), head_angle < np.deg2rad(45/2)
    else:
        return head_angle < np.deg2rad(45/2), head_angle < np.deg2rad(45/2)


if __name__ == "__main__":

    env = swift.Swift()
    env.launch(realtime=False, headless=True)
    # env.launch(realtime=True)

    moving_object = False
    total_runs = 1000

    load_run = 0

    for run in range(total_runs):

        ax_goal = sg.Axes(0.1)
        
        fetch = rtb.models.Fetch()
        fetch.q = np.random.uniform(low=[0, 0, 0, -1.6056, -1.221, 0, -2.251, 0, -2.160, 0], 
                                    high=[0, 0, 0, 1.6056, 1.518, 6.283, 2.251, 6.283, 2.160, 6.283], 
                                    size=10
        )     

        fetch_camera = rtb.models.FetchCamera()
        fetch_camera.q = fetch_camera.qr

        sight_base = sm.SE3.Ry(np.pi/2) * sm.SE3(0.0, 0.0, 2.5)
        centroid_sight = sg.Cylinder(radius=0.001, 
                                     length=5.0, 
                                     base=fetch_camera.fkine(fetch_camera.q, fast=True) @ sight_base.A
        )
        
        line_of_sight = sg.Cylinder(radius=0.001, 
                                    length=5.0, 
                                    base=fetch_camera.fkine(fetch_camera.q, fast=True) @ sight_base.A
        )
        

        arrived = False
        separate_arrived = False
        dt = 0.025


        # vel = 0.06
        moving_time = 10 # in seconds
        # direction = np.random.rand(2) * 2 - 1
        # direction = direction / np.linalg.norm(direction)
        ang_vel = 0.1 # in rad/dt
        speed_drop = ang_vel / (moving_time / dt)
        current_angle = 0

        stop_time = 20 # in seconds

        
        wTep = rand_pose()
        spawn_pose = [wTep.A[0, 3], wTep.A[1, 3]]        

        if run < load_run:
            continue

        env.set_camera_pose([-2, 3, 0.7], [-2, 0.0, 0.5])
        env.add(ax_goal)   
        env.add(fetch)
        env.add(fetch_camera)
        env.add(centroid_sight)
        env.add(line_of_sight)


        ax_goal.base = wTep

        env.step()

        total_count = 0
        seen_count = 0
        fov_count = 0

        while not arrived:

            if moving_object: 
                wTep.A[0,3] = spawn_pose[0] + np.sin(current_angle) * 0.15                                      
                wTep.A[1,3] = spawn_pose[1] + np.cos(current_angle) * 0.15

                current_angle += ang_vel

                min_vel = ang_vel / 50 if dt * total_count < stop_time else 0
                ang_vel = max(min_vel, ang_vel - speed_drop)


            try:
                arrived, fetch.qd, fetch_camera.qd = step_p_camera(fetch, fetch_camera, wTep.A)
                # if not separate_arrived:
                #     separate_arrived, fetch.qd, fetch_camera.qd = step_separate_base(fetch, fetch_camera, wTep.A)
                # else:
                #     arrived, fetch.qd, fetch_camera.qd = step_separate_arm(fetch, fetch_camera, wTep.A)
            except Exception as e:
                print(e)
            env.step(dt)

            # Reset bases
            base_new = fetch.fkine(fetch._q, end=fetch.links[2], fast=True)
            fetch._base.A[:] = base_new
            fetch.q[:2] = 0

            base_new = fetch_camera.fkine(fetch_camera._q, end=fetch_camera.links[2], fast=True)
            fetch_camera._base.A[:] = base_new
            fetch_camera.q[:2] = 0

            total_count += 1
            seen, fov = obj_in_vision(fetch, fetch_camera, wTep.A)
            seen_count += seen
            fov_count += fov

            centroid_sight._base = fetch_camera.fkine(fetch_camera.q, fast=True) @ sight_base.A

            if (total_count * dt) > 50:
                print("Simulation time out")
                break

        print(run, "/", total_runs)
        print("Vision: ", seen_count / total_count * 100, "%")
        print("FoV: ", fov_count / total_count * 100, "%")
        print("Time elapsed: ", total_count * dt, "s")
        print("Success: ", arrived)
        print()

        vision_pc =  seen_count / total_count * 100
        fov_pc = fov_count / total_count * 100
        time_elapsed = total_count * dt
        is_success = arrived

        with open('data_alt.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([run, vision_pc, time_elapsed, is_success, fov_pc])

        env.restart()
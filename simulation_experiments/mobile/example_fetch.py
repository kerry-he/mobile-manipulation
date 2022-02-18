import swift
import spatialgeometry as sg
import roboticstoolbox as rtb
import spatialmath as sm
import numpy as np
import qpsolvers as qp
import math


def transform_between_vectors(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    angle = np.arccos(np.dot(a, b))
    axis = np.cross(a, b)

    return sm.SE3.AngleAxis(angle, axis), angle, axis


# Launch the simulator Swift
env = swift.Swift()
env.launch()

# Create a Fetch and Camera robot object
fetch = rtb.models.Fetch()
fetch_camera = rtb.models.FetchCamera()

# Set joint angles to ready configuration
fetch.q = fetch.qr
fetch_camera.q = fetch_camera.qr

# Make target object obstacles with velocities
target = sg.Sphere(radius=0.05, base=sm.SE3(0.52, 0.4, 0.3))

# Make line of sight object, modeled as cylinder
sight_base = sm.SE3.Ry(np.pi / 2) * sm.SE3(0.0, 0.0, 2.5)
centroid_sight = sg.Cylinder(
    radius=0.001,
    length=5.0,
    base=fetch_camera.fkine(fetch_camera.q, fast=True) @ sight_base.A,
)


# Add the Panda and shapes to the simulator
env.add(fetch)
env.add(fetch_camera)
env.add(centroid_sight)
env.add(target)

# Set the desired end-effector pose to the location of target
Tep = fetch.fkine(fetch.q)
Tep.A[:3, 3] = target.base.t

env.step()


def step():

    # Find end-effector pose in world frame
    wTe = fetch.fkine(fetch.q, fast=True)
    # Find camera pose in world frame
    wTc = fetch_camera.fkine(fetch_camera.q, fast=True)

    # Find transform between end-effector and goal
    eTep = np.linalg.inv(wTe) @ Tep.A
    # Find transform between camera and goal
    cTep = np.linalg.inv(wTc) @ Tep.A

    # Spatial error
    et = np.sum(np.abs(eTep[:3, -1]))

    head_rotation, head_angle, _ = transform_between_vectors(
        np.array([1, 0, 0]), cTep[:3, 3]
    )

    # Gain term (lambda) for control minimisation
    Y = 0.01

    # Quadratic component of objective function
    Q = np.eye(fetch.n + 2 + 10)

    Q[: fetch.n, : fetch.n] *= Y  # Robotic manipulator
    Q[:2, :2] *= 1.0 / et  # Mobile base
    Q[fetch.n : fetch.n + 2, fetch.n : fetch.n + 2] *= Y  # Camera
    Q[fetch.n + 2 : -7, fetch.n + 2 : -7] = (1000.0 / np.power(et, 2)) * np.eye(
        3
    )  # Slack manipulator
    Q[fetch.n + 5 : -4, fetch.n + 5 : -4] = (Y / np.power(et, 5)) * np.eye(
        3
    )  # Slack manipulator
    Q[-4:-1, -4:-1] = 100 * np.eye(3)  # Slack camera
    Q[-1, -1] = 1000.0 * np.power(et, 3)

    v_manip, _ = rtb.p_servo(wTe, Tep, 1.5)
    v_manip[3:] *= 1.3

    v_camera, _ = rtb.p_servo(sm.SE3(), head_rotation, 20)
    v_camera *= 1.3

    # The equality contraints
    Aeq = np.c_[
        fetch.jacobe(fetch.q, fast=True), np.zeros((6, 2)), np.eye(6), np.zeros((6, 4))
    ]
    beq = v_manip.reshape((6,))

    jacobe_cam = fetch_camera.jacobe(fetch_camera.q, fast=True)[3:, :]
    Aeq = np.r_[
        Aeq,
        np.c_[
            jacobe_cam[:, :3],
            np.zeros((3, 7)),
            jacobe_cam[:, 3:],
            np.zeros((3, 6)),
            np.eye(3),
            np.zeros((3, 1)),
        ],
    ]
    beq = np.r_[beq, v_camera[3:].reshape((3,))]

    # The inequality constraints for joint limit avoidance
    Ain = np.zeros((fetch.n + 2 + 10, fetch.n + 2 + 10))
    bin = np.zeros(fetch.n + 2 + 10)

    # The minimum angle (in radians) in which the joint is allowed to approach
    # to its limit
    ps = 0.1

    # The influence angle (in radians) in which the velocity damper
    # becomes active
    pi = 0.9

    # Form the joint limit velocity damper
    Ain[: fetch.n, : fetch.n], bin[: fetch.n] = fetch.joint_velocity_damper(
        ps, pi, fetch.n
    )

    Ain_torso, bin_torso = fetch_camera.joint_velocity_damper(0.0, 0.05, fetch_camera.n)
    Ain[2, 2] = Ain_torso[2, 2]
    bin[2] = bin_torso[2]

    Ain_cam, bin_cam = fetch_camera.joint_velocity_damper(ps, pi, fetch_camera.n)
    Ain[fetch.n : fetch.n + 2, fetch.n : fetch.n + 2] = Ain_cam[3:, 3:]
    bin[fetch.n : fetch.n + 2] = bin_cam[3:]

    # Draw line of sight between camera and object
    camera_pos = wTc[:3, 3]
    target_pos = Tep.A[:3, 3]

    middle = (camera_pos + target_pos) / 2
    R, _, _ = transform_between_vectors(
        np.array([0.0, 0.0, 1.0]), camera_pos - target_pos
    )

    los = sg.Cylinder(
        radius=0.001,
        length=np.linalg.norm(camera_pos - target_pos),
        base=(sm.SE3(middle) * R),
    )

    c_Ain, c_bin, _ = fetch.vision_collision_damper(
        los,
        fetch.q[: fetch.n],
        0.3,
        0.2,
        1.0,
        start=fetch.link_dict["shoulder_pan_link"],
        end=fetch.link_dict["gripper_link"],
        camera=fetch_camera,
    )

    if c_Ain is not None and c_bin is not None:
        Ain = np.r_[Ain, c_Ain]
        bin = np.r_[bin, c_bin]

    # Linear component of objective function: the manipulability Jacobian
    c = np.concatenate(
        (
            np.zeros(2),
            -fetch.jacobm(start=fetch.links[3]).reshape((fetch.n - 2,)),
            np.zeros(2),
            np.zeros(10),
        )
    )

    # Get base to face end-effector
    kε = 0.5
    bTe = fetch.fkine(fetch.q, include_base=False, fast=True)
    θε = math.atan2(bTe[1, -1], bTe[0, -1])
    ε = kε * θε
    c[0] = -ε

    # The lower and upper bounds on the joint velocity and slack variable
    lb = -np.r_[
        fetch.qdlim[: fetch.n],
        fetch_camera.qdlim[3 : fetch_camera.n],
        100 * np.ones(9),
        0,
    ]
    ub = np.r_[
        fetch.qdlim[: fetch.n],
        fetch_camera.qdlim[3 : fetch_camera.n],
        100 * np.ones(9),
        100,
    ]

    # Solve for the joint velocities dq
    qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub)
    qd_cam = np.concatenate((qd[:3], qd[fetch.n : fetch.n + 2]))
    qd = qd[: fetch.n]

    if et > 0.5:
        qd *= 0.7 / et
        qd_cam *= 0.7 / et
    else:
        qd *= 1.4
        qd_cam *= 1.4

    arrived = et < 0.02

    fetch.qd = qd
    fetch_camera.qd_cam = qd_cam
    centroid_sight._base = fetch_camera.fkine(fetch_camera.q, fast=True) @ sight_base.A

    return arrived


arrived = False
while not arrived:
    arrived = step()
    env.step(0.01)

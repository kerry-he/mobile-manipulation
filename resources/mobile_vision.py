import swift
import roboticstoolbox as rtb
import spatialgeometry as sg
import spatialmath as sm
import qpsolvers as qp
import numpy as np
import math

thresh = 0.5

def transform_between_vectors(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    angle = np.arccos(np.dot(a, b))
    axis = np.cross(a, b)

    return sm.SE3.AngleAxis(angle, axis), angle, axis


def step_robot(r, r_cam, Tep):

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
    Q[r.n + 5 : -3, r.n + 5 : -3] = (1.0 / (et * et * et * et)) * np.eye(3)      # Slack manipulator
    Q[-3:, -3:] = 10 * np.eye(3)                                # Slack camera

    print((1.0 / (et * et * et * et)))

    v_manip, _ = rtb.p_servo(wTe, Tep, 1.5)
    v_manip[3:] *= 1.3

    v_camera, _ = rtb.p_servo(sm.SE3(), head_rotation, 3)
    v_camera *= 1.3
    # v_cam_norm = np.linalg.norm(v_camera)
    # v_camera = v_camera / np.linalg.norm(v_camera) * 5 * (1-np.exp(-v_cam_norm*10))

    # The equality contraints
    if False:
        Aeq = np.c_[r.jacobe(r.q, fast=True), np.zeros((6, 2)), np.eye(6), np.zeros((6, 3))][:3, :]
        beq = v_manip[:3].reshape((3,))
    else:
        Aeq = np.c_[r.jacobe(r.q, fast=True), np.zeros((6, 2)), np.eye(6), np.zeros((6, 3))]
        beq = v_manip.reshape((6,))
        thresh

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
    Ain_cam, bin_cam = r_cam.joint_velocity_damper(ps, pi, r_cam.n)
    Ain[r.n : r.n + 2, r.n : r.n + 2] = Ain_cam[3:, 3:]
    bin[r.n : r.n + 2] = bin_cam[3:]

    c_Ain, c_bin = r.vision_collision_damper(
        line_of_sight,
        r.q[:r.n],
        0.3,
        0.1,
        0.1,
        start=r.link_dict["shoulder_pan_link"],
        end=r.link_dict["wrist_roll_link"],
        camera=r_cam
    )

    if c_Ain is not None and c_bin is not None and et > thresh:
        # Stack the inequality constraints
        c_Ain[:, :2] = 0
        Ain = np.r_[Ain, c_Ain]
        bin = np.r_[bin, c_bin]


    # Linear component of objective function: the manipulability Jacobian
    c = np.concatenate(
        (np.zeros(2), -r.jacobm(start=r.links[3]).reshape((r.n - 2,)), np.zeros(2), np.zeros(9))
    )

    # Get base to face end-effector
    kε = 0.5
    bTe = r.fkine(r.q, include_base=False, fast=True)
    θε = math.atan2(bTe[1, -1], bTe[0, -1])
    ε = kε * θε
    c[0] = -0

    # The lower and upper bounds on the joint velocity and slack variable
    lb = -np.r_[r.qdlim[: r.n], r_cam.qdlim[3:r_cam.n], 100 * np.ones(9)]
    ub = np.r_[r.qdlim[: r.n], r_cam.qdlim[3:r_cam.n], 100 * np.ones(9)]

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


env = swift.Swift()
env.launch(realtime=True)

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
wTep = fetch.fkine(fetch.q) * sm.SE3.Rz(np.pi)
wTep.A[:3, :3] = np.diag([-1, 1, -1])
# wTep.A[0, -1] -= 3.0
# wTep.A[1, -1] += 3.0
# wTep.A[2, -1] += 0.75
wTep.A[0, -1] += (np.random.rand() - 0.5) * 10
wTep.A[1, -1] += (np.random.rand() - 0.5) * 10
wTep.A[2, -1] += (np.random.rand() - 0.5) * 1
ax_goal.base = wTep
env.step()

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

    # env.remove(line_of_sight)

    # wTc = fetch_camera.fkine(fetch_camera.q, fast=True)
    # cTep = np.linalg.inv(wTc) @ wTep.A
    # ct = np.sum(np.abs(cTep[:3, -1])) - 0.15 

    # sight_base = sm.SE3.Ry(np.pi/2) * sm.SE3(0.0, 0.0, ct/2)
    # line_of_sight = sg.Cylinder(radius=0.001, length=ct, base=fetch_camera.fkine(fetch_camera.q, fast=True) @ sight_base.A)
    # env.add(line_of_sight)
    line_of_sight._base = fetch_camera.fkine(fetch_camera.q, fast=True) @ sight_base.A

env.hold()

import swift
import roboticstoolbox as rtb
import spatialgeometry as sg
import spatialmath as sm
import qpsolvers as qp
import numpy as np
import math


def step_robot(r, Tep):

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

    # Form the velocity damper inequality contraint for each collision
    # object on the robot to the collision in the scene
    c_Ain, c_bin = r.link_collision_damper(
        table,
        r.q[:r.n],
        0.3,
        0.05,
        1.0,
        start=r.link_dict["base_link"],
        end=r.link_dict["r_gripper_finger_link"],
    )

    # If there are any parts of the robot within the influence distance
    # to the collision in the scene
    if c_Ain is not None and c_bin is not None:
        c_Ain = np.c_[c_Ain, np.zeros((c_Ain.shape[0], 6))]

        # Stack the inequality constraints
        Ain = np.r_[Ain, c_Ain]
        bin = np.r_[bin, c_bin]

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

    if et > 0.5:
        qd *= 0.7 / et
    else:
        qd *= 1.4

    if et < 0.02:
        return True, qd
    else:
        return False, qd


env = swift.Swift()
env.launch(realtime=True)

ax_goal = sg.Axes(0.1)
env.add(ax_goal)

fetch = rtb.models.Fetch()
fetch.q = fetch.qr
env.add(fetch)

arrived = False
dt = 0.025

# Behind
env.set_camera_pose([-2, 3, 0.7], [-2, 0.0, 0.5])
wTep = fetch.fkine(fetch.q) * sm.SE3.Rz(np.pi)
wTep.A[:3, :3] = np.diag([-1, 1, -1])
wTep.A[0, -1] += 3.0
# wTep.A[1, -1] += 3.0
wTep.A[2, -1] += 0.1
# wTep.A[0, -1] += (np.random.rand() - 0.5) * 10
# wTep.A[1, -1] += (np.random.rand() - 0.5) * 10
# wTep.A[2, -1] += (np.random.rand() - 0.5) * 1
ax_goal.base = wTep

table = sg.Cuboid(scale=(1.0, 1.0, 0.1), base=sm.SE3(wTep.A[0, -1] + 0.4, wTep.A[1, -1],  wTep.A[2, -1] - 0.1))
env.add(table)

env.step()


while not arrived:

    arrived, fetch.qd = step_robot(fetch, wTep.A)
    env.step(dt)

    # Reset bases
    base_new = fetch.fkine(fetch._q, end=fetch.links[2], fast=True)
    fetch._base.A[:] = base_new
    fetch.q[:2] = 0

env.hold()

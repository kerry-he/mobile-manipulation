import swift
import spatialgeometry as sg
import roboticstoolbox as rtb
import spatialmath as sm
import numpy as np
import qpsolvers as qp


def rotation_between_vectors(a, b):
    # Finds the shortest rotation between two vectors using angle-axis,
    # then outputs it as a 4x4 transformation matrix
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    angle = np.arccos(np.dot(a, b))
    axis = np.cross(a, b)

    return sm.SE3.AngleAxis(angle, axis)


# Launch the simulator Swift
env = swift.Swift()
env.launch()

# Create a Panda robot object
panda = rtb.models.Panda()

# Set joint angles to ready configuration
panda.q = panda.qr

# Number of joint in the panda which we are controlling
n = 7

# Make a target
target = sg.Sphere(radius=0.02, base=sm.SE3(0.25, 0.0, 0.0))

# Make a camera
camera = sg.Sphere(radius=0.02, base=sm.SE3(0.0, 0.0, 1.0))


los_mid = sm.SE3((camera.base.t + target.base.t) / 2)
los_orientation = rotation_between_vectors(np.array([0., 0., 1.]), camera.base.t - target.base.t)

los = sg.Cylinder(radius=0.001, 
                  length=2.0, 
                  base=(los_mid * los_orientation)
)      

# Add the Panda and shapes to the simulator
env.add(panda)
env.add(target)
env.add(camera)
env.add(los)

# Set the desired end-effector pose to the location of target
Tep = panda.fkine(panda.q)
Tep.A[:3, 3] = target.base.t
Tep.A[2, 3] += 0.15

time = 0


def step(time):
    Te = panda.fkine(panda.q)

    # Transform from the end-effector to desired pose
    eTep = Te.inv() * Tep

    # Spatial error
    et = np.sum(np.abs(eTep.A[:3, -1]))

    # Calulate the required end-effector spatial velocity for the robot
    # to approach the goal. Gain is set to 1.0
    v, _ = rtb.p_servo(Te, Tep, 1, 0.01)

    # The inequality constraints for joint limit avoidance
    n_slack = 3 + 1 + 1

    # Weighting function used for objective function
    def w_lambda(et, alpha, gamma):
        return alpha * np.power(et, gamma)

    # Quadratic component of objective function
    Q = np.eye(n + n_slack)
    Q[:n, :n]                       *= 0.01                         # Joint velocity
    Q[n: n + 3, n: n + 3]           *= w_lambda(et, 5.0, -1.0)      # Linear velocity slack
    Q[n + 3: n + 4, n + 3: n + 4]   *= w_lambda(et, 10.0, -1.0)     # Gripper orientation slack
    Q[-1, -1]                       *= w_lambda(et, 10000.0, 0.0)   # Self-occlusion slack

    Aeq = np.c_[panda.jacobe(panda.q)[:3], np.eye(3), np.zeros((3, 2))]
    beq = v[:3].reshape((3,))

    Ain = np.zeros((n + n_slack, n + n_slack))
    bin = np.zeros(n + n_slack)

    # The minimum angle (in radians) in which the joint is allowed to approach
    # to its limit
    ps = 0.3

    # The influence angle (in radians) in which the velocity damper
    # becomes active
    pi = 0.9

    # Form the joint limit velocity damper
    Ain[:n, :n], bin[:n] = panda.joint_velocity_damper(ps, pi, n)

    # End effector orientation inequality constraint
    psi_limit = np.deg2rad(30)

    z_ee = panda.fkine(panda.q).A[:3, 2]
    z_world = np.array([0, 0, 1])

    psi = np.arccos(np.dot(-z_ee, z_world) /
                        (np.linalg.norm(z_ee) * np.linalg.norm(z_world)))

    n_psi = np.cross(z_ee, z_world)

    J_psi = n_psi.T @ panda.jacob0(panda.q)[3:]
    J_psi = J_psi.reshape((1, 7))

    damper = 1.0 * (np.cos(psi) - np.cos(psi_limit)
                    ) / (1 - np.cos(psi_limit))

    c_Ain = np.c_[J_psi, np.zeros((1, n_slack))]

    Ain = np.r_[Ain, c_Ain]
    bin = np.r_[bin, damper]

    # End effector palm orientation equality constraint
    y_ee = panda.fkine(panda.q).A[:3, 1]
    
    phi = np.arccos(np.dot(y_ee, z_world) /
                        (np.linalg.norm(y_ee) * np.linalg.norm(z_world)))

    n_phi = np.cross(z_world, y_ee)

    J_face = n_phi.T @ panda.jacob0(panda.q)[3:]
    J_face = J_face.reshape((1, 7))

    c_Aeq = np.c_[J_face, np.zeros((1, 3)), 1, 0]

    Aeq = np.r_[Aeq, c_Aeq]
    beq = np.r_[beq, np.cos(phi)]

    # Linear component of objective function: the manipulability Jacobian
    c = np.r_[-panda.jacobm(panda.q).reshape((n,)),
                np.zeros(n_slack)]

    # Self-occlusion avoidance constraint
    c_Ain, c_bin, _ = panda.newest_vision_collision_damper(
        target,
        camera=camera.base,
        q=panda.q[: n],
        di=0.3,
        ds=0.05,
        xi=1.0,
        end=panda.link_dict["panda_hand"],
        start=panda.link_dict["panda_link1"],
    )    

    if c_Ain is not None and c_bin is not None:
        c_Ain = np.c_[c_Ain, np.zeros((c_Ain.shape[0], 4)), -np.ones((c_Ain.shape[0], 1))]

        Ain = np.r_[Ain, c_Ain]
        bin = np.r_[bin, c_bin]                    

    # The lower and upper bounds on the joint velocity and slack variable
    lb = -np.r_[panda.qdlim[:n], 100 * np.ones(4), 0]
    ub = np.r_[panda.qdlim[:n], 100 * np.ones(n_slack)]

    qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub)
    panda.qd = qd[:n]


    # Update object velocity
    target.v[0] = np.sin(time * 2) / 2
    target.v[1] = np.cos(time * 2) / 2

    Tep.A[:2, 3] = target.base.t[:2]

    # Update line of sight
    los_mid = sm.SE3((camera.base.t + target.base.t) / 2)
    los_orientation = rotation_between_vectors(np.array([0., 0., 1.]), camera.base.t - target.base.t)

    los.base = los_mid * los_orientation

    return

arrived = False
while not arrived:
    step(time)
    time += 0.01
    env.step(0.01)
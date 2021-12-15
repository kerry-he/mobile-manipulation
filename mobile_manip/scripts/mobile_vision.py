#!/usr/bin/env python3
import rospy
import tf
import numpy as np
import time

import roboticstoolbox as rtb
import spatialgeometry as sg
import spatialmath as sm
import qpsolvers as qp
import numpy as np
import math

from sensor_msgs.msg import JointState
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from control_msgs.msg import JointJog, GripperCommandGoal, GripperCommandAction
from fetch_driver_msgs.msg import GripperState
import actionlib



def transform_between_vectors(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    angle = np.arccos(np.dot(a, b))
    axis = np.cross(a, b)

    return sm.SE3.AngleAxis(angle, axis), angle, axis


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


class MobileManipController:
    def __init__(self):
        self.fetch = rtb.models.Fetch()

        self.wTep = None
        self.waypoint = 0
        self.wTep_f = None

        self.gripper_closed = False



        self.tf_broadcaster = tf.TransformBroadcaster()
        self.tf_listener  = tf.TransformListener()
        self.jointpos_sub = rospy.Subscriber("joint_states", JointState, self.jointpos_callback)
        self.jointpos_pub = rospy.Publisher("joint_states2", JointState, queue_size=1)

        self.gripper_sub = rospy.Subscriber("/gripper_state", GripperState, self.gripper_callback)
        self.jointvel_pub = rospy.Publisher("/arm_controller/joint_velocity/command", JointJog, queue_size=1)
        self.basevel_pub  = rospy.Publisher("/base_controller/command", Twist, queue_size=1)
        # self.gripper_pub = rospy.Publisher("/gripper_controller/gripper_action/goal", GripperCommandActionGoal, queue_size=1)
        self.gripper_action = actionlib.SimpleActionClient('/gripper_controller/gripper_action', GripperCommandAction)
        self.gripper_action.wait_for_server()


        self.main_timer = rospy.Timer(rospy.Duration(0.01), self.main_callback)

    def gripper_callback(self, data):
        self.gripper_closed = (data.joints[0].effort > 100)

    def jointpos_callback(self, data): 
        # Check if message is a gripper state or manipulator state
        if len(data.name) > 2:
            # Read joint positions
            self.fetch.q[:2] = 0
            self.fetch.q[2:] = np.array([data.position[6],
                                         data.position[7],
                                         data.position[8],
                                         data.position[9],
                                         data.position[10],
                                         data.position[11],
                                         data.position[12]])
            self.jointpos_pub.publish(data)

    def main_callback(self, event):
        print(1)
        # Read base position
        try:
            # Read base frame coordinates
            (trans, rot) = self.tf_listener.lookupTransform('map', 'base_link', rospy.Time(0))

            T = sm.SE3()
            T.A[:3, :3] = sm.UnitQuaternion(rot[3], rot[:3]).R
            T.A[:3, 3] = trans
            self.fetch._base = T
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            # print(e)
            pass


        # Read marker position
        try:
            # Read base frame coordinates
            (trans, rot) = self.tf_listener.lookupTransform('map', 'ar_marker_1', rospy.Time(0))

            T = sm.SE3()
            T.A[:3, :3] = sm.UnitQuaternion(rot[3], rot[:3]).R @ sm.SE3.Ry(0).R
            if self.waypoint == 0:
                T.A[:3, 3] = trans + sm.UnitQuaternion(rot[3], rot[:3]).R @ np.array([-0.2, 0.2, 0.05]) 
                self.wTep = T
            elif self.waypoint == 1:
                T.A[:3, 3] = trans + sm.UnitQuaternion(rot[3], rot[:3]).R @ np.array([0., 0.2, 0.025]) 
                self.wTep = T
            elif self.waypoint == 2:
                self.wTep = sm.SE3([0, 0, 0.2]) * self.wTep_f 
            else:
                self.wTep = self.wTep_f


            self.tf_broadcaster.sendTransform(tuple(self.wTep.t),
                                            tuple(sm.UnitQuaternion(self.wTep.R).A[1:]) + tuple([sm.UnitQuaternion(self.wTep.R).A[0]]),
                                            rospy.Time.now(),
                                            "goal",
                                            "map")            
            
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            print(e)
            pass


        if self.wTep_f is None and self.fetch._base is not None:
            self.wTep_f = sm.SE3(self.fetch.fkine(self.fetch.q, fast=False))


        # Calculate using Jesse's controller
        try:
            arrived, qd = step_robot(self.fetch, self.wTep.A)
        except:
            print("QP Solve failed")
            self.stop_robot()
            return


        if arrived:
            if self.waypoint == 0:
                # self.stop_robot()
                # rospy.sleep(0.5)

                # for i in range(100):
                #     v_base = Twist()
                #     v_base.linear.x = 0.1
                #     self.basevel_pub.publish(v_base)
                #     # rospy.sleep(0.025)

                # self.stop_robot()
                # msg = GripperCommandActionGoal()
                # msg.goal.command.position = 0.0
                # self.gripper_pub.publish(msg)    
                # rospy.sleep(2.0)         
                # goal = GripperCommandGoal()
                # goal.command.position = 0.0
                # self.gripper_action.send_goal(goal)
                # self.gripper_action.wait_for_result()

                # for i in range(100):
                #     v_base = Twist()
                #     v_base.linear.x = -0.1
                #     self.basevel_pub.publish(v_base)
                #     # rospy.sleep(0.025)
    

                self.waypoint += 1
            elif self.waypoint == 1:
                self.stop_robot()
                # rospy.sleep(1.0)

                # msg = GripperCommandActionGoal()
                # msg.goal.command.position = 0.0
                # self.gripper_pub.publish(msg)
                # rospy.sleep(2.0)
                goal = GripperCommandGoal()
                goal.command.position = 0.0
                self.gripper_action.send_goal(goal)
                self.gripper_action.wait_for_result()


                self.waypoint += 1
            elif self.waypoint == 2:
                self.waypoint += 1
            elif self.waypoint == 3:
                self.stop_robot()
                goal = GripperCommandGoal()
                goal.command.position = 1.0
                self.gripper_action.send_goal(goal)
                self.gripper_action.wait_for_result()

                # msg = GripperCommandActionGoal()
                # msg.goal.command.position = 1.0
                # self.gripper_pub.publish(msg)

        print(self.waypoint)

        # Publish base and joint velocities
        v_base = Twist()
        v_base.angular.z = qd[0]
        v_base.linear.x = qd[1]
        self.basevel_pub.publish(v_base)

        v_joint = JointJog()
        v_joint.velocities = np.append(0, qd[2:])
        self.jointvel_pub.publish(v_joint)


        # print("qd", qd)
        # print("xd", self.fetch.jacobe(self.fetch.q, fast=True) @ qd)
        # print("joint pos", self.fetch.q[2:])
        # print("measured vel", np.array([data.velocity[6],
        #                                 data.velocity[7],
        #                                 data.velocity[8],
        #                                 data.velocity[9],
        #                                 data.velocity[10],
        #                                 data.velocity[11],
        #                                 data.velocity[12]]))

        return

    def stop_robot(self):
            v_base = Twist()
            self.basevel_pub.publish(v_base)

            v_joint = JointJog()
            v_joint.velocities = np.zeros(8)
            self.jointvel_pub.publish(v_joint)

    def init_wTep(self):
        #self.wTep = self.fetch.fkine(self.fetch.q) * sm.SE3.Rz(0)
        self.wTep = sm.SE3([0, -2, 0]) * self.fetch.fkine(self.fetch.q)
        self.wTep.A[:3, :3] = (sm.SE3.Rz(np.pi) * self.fetch._base).R
        #self.wTep.A[0, -1] += 1.0
        #self.wTep.A[2, -1] += 1

        #print(self.fetch._base)
        #print(self.fetch.fkine(self.fetch.q))
        #print(self.wTep)

        self.tf_broadcaster.sendTransform(tuple(self.wTep.t),
                                          tuple(sm.UnitQuaternion(self.wTep.R).A[1:]) + tuple([sm.UnitQuaternion(self.wTep.R).A[0]]),
                                          rospy.Time.now(),
                                          "goal",
                                          "map")


if __name__ == '__main__':
    try:
        rospy.init_node("mobile_manip", anonymous=True)
        controller = MobileManipController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


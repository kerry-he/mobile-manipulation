#!/usr/bin/env python3
import rospy
import tf
import numpy as np
import time
import swift

import roboticstoolbox as rtb
import spatialgeometry as sg
import spatialmath as sm
import qpsolvers as qp
import numpy as np
import math
import subprocess


from sensor_msgs.msg import JointState
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from control_msgs.msg import JointJog, GripperCommandGoal, GripperCommandAction
from fetch_driver_msgs.msg import GripperState
import actionlib

from Controllers import *


class MobileManipController:
    def __init__(self):
        self.fetch = rtb.models.Fetch()
        self.fetch_cam = rtb.models.FetchCamera()

        self.wTep = None

        self.tf_broadcaster = tf.TransformBroadcaster()
        self.tf_listener = tf.TransformListener()

        self.jointpos_sub = rospy.Subscriber(
            "joint_states", JointState, self.jointpos_callback)

        self.jointvel_pub = rospy.Publisher(
            "/arm_controller/joint_velocity/command", JointJog, queue_size=1)
        self.headvel_pub = rospy.Publisher(
            "/head_controller/joint_velocity/command", JointJog, queue_size=1)

        self.basevel_pub = rospy.Publisher(
            "/base_controller/command", Twist, queue_size=1)
        self.gripper_action = actionlib.SimpleActionClient(
            '/gripper_controller/gripper_action', GripperCommandAction)
        self.gripper_action.wait_for_server()

        self.env = swift.Swift()
        self.env.launch(realtime=True)

        self.env.add(self.fetch)
        self.env.add(self.fetch_cam)

        self.total_count = 0
        self.seen_count = 0

        self.start_time = rospy.Time.now().to_sec()

        self.error_list = np.random.rand(100)
        self.separate_arrived = False

        self.main_timer = rospy.Timer(rospy.Duration(0.01), self.main_callback)

    def jointpos_callback(self, data):
        # Check if message is a gripper state or manipulator state
        if len(data.name) > 2:
            # Read joint positions
            self.fetch.q[:2] = 0
            self.fetch.q[2:] = np.array([data.position[2],
                                         data.position[6],
                                         data.position[7],
                                         data.position[8],
                                         data.position[9],
                                         data.position[10],
                                         data.position[11],
                                         data.position[12]])

            self.fetch_cam.q[:2] = 0
            self.fetch_cam.q[2:] = np.array([data.position[2],
                                             data.position[4],
                                             data.position[5]])

    def main_callback(self, event):
        # Read base position
        try:
            # Read base frame coordinates
            (trans, rot) = self.tf_listener.lookupTransform(
                'map', 'base_link', rospy.Time(0))

            T = sm.SE3()
            T.A[:3, :3] = sm.UnitQuaternion(rot[3], rot[:3]).R
            T.A[:3, 3] = trans

            self.fetch._base = T
            self.fetch_cam._base = T

            # Give initial guess about where the marker is
            if self.wTep is None:
                self.wTep = T * sm.SE3([-0.25, 2, 0.75])

                # Rotation of the gripper
                goal_to_base = self.fetch._base.t - self.wTep.t
                self.wTep.A[:3, :3] = sm.SE3.Rz(np.arctan2(
                    goal_to_base[1], goal_to_base[0]) + np.pi).A[:3, :3]

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            # print(e)
            pass

        if self.wTep is None:
            return

        # Spatial error
        wTe = self.fetch.fkine(self.fetch.q, fast=True)
        eTep = np.linalg.inv(wTe) @ self.wTep.A
        et = np.sum(np.abs(eTep[:3, -1]))

        # Read marker position
        try:
            # Read base frame coordinates
            (trans, rot) = self.tf_listener.lookupTransform(
                'map', 'apple', rospy.Time(0))

            # Set target pose to be in front of marker
            T = sm.SE3()
            T.A[:3, :3] = sm.UnitQuaternion(rot[3], rot[:3]).R @ sm.SE3.Ry(0).R
            # + sm.UnitQuaternion(rot[3], rot[:3]).R @ np.array([-0.2, 0, 0])
            T.A[:3, 3] = trans

            # Only update position if detected object is close enough, and if the robot isn't too close
            if np.linalg.norm(self.wTep.t - trans) < et / 3 and et > 0.1:
                self.wTep.A[:3, 3] = trans + \
                    self.wTep.R @ np.array([0.025, 0, 0])
                self.seen_count += 1

            self.tf_broadcaster.sendTransform(tuple(self.wTep.t),
                                              tuple(sm.UnitQuaternion(
                                                  self.wTep.R).A[1:]) + tuple([sm.UnitQuaternion(self.wTep.R).A[0]]),
                                              rospy.Time.now(),
                                              "goal",
                                              "map")

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            # print(e)
            pass

        self.total_count += 1

        # Calculate using Jesse's controller
        arrived, qd, qd_cam = step_holistic(
            self.fetch, self.fetch_cam, self.wTep.A)

        self.error_list = np.concatenate((self.error_list[1:], np.array([et])))
        arrived = arrived or np.std(self.error_list) < 0.0025

        print(et, np.std(self.error_list))

        # arrived = False
        # if not self.separate_arrived:
        #     self.separate_arrived, qd, qd_cam = step_separate_base(self.fetch, self.fetch_cam, self.wTep.A)
        # else:
        #     arrived, qd, qd_cam = step_separate_arm(self.fetch, self.fetch_cam, self.wTep.A)

        arrived = False
        if not self.separate_arrived:
            self.separate_arrived, qd, qd_cam = step_separate_base(
                self.fetch, self.fetch_cam, self.wTep.A)
        else:

            command = "source TODO; holistop; python MoveItPython2"
            args = []
            for i in range(16):
                args.append(self.wTep.A[i/4, i % 4])

            command = command + " ".join(args)

            ret = subprocess.run(command, capture_output=True, shell=True)

        qd /= 10
        qd_cam /= 10

        if arrived:
            self.stop_robot()

            goal = GripperCommandGoal()
            goal.command.position = 0.0
            self.gripper_action.send_goal(goal)
            self.gripper_action.wait_for_result()

            print("\n\nExperiment complete!")
            print("Vision: ", self.seen_count / self.total_count * 100, "%")
            print("Time elapsed: ", rospy.Time.now().to_sec() - self.start_time)
            print("\n\n")

        # Publish base and joint velocities
        v_base = Twist()
        v_base.angular.z = qd[0]
        v_base.linear.x = qd[1]
        self.basevel_pub.publish(v_base)

        v_joint = JointJog()
        v_joint.velocities = qd[2:]
        self.jointvel_pub.publish(v_joint)

        v_head = JointJog()
        v_head.velocities = qd_cam[-2:]
        self.headvel_pub.publish(v_head)

        sight_base = sm.SE3.Ry(np.pi/2) * sm.SE3(0.0, 0.0, 2.5)
        line_of_sight = sg.Cylinder(radius=0.001, length=5.0, base=self.fetch_cam.fkine(
            self.fetch_cam.q, fast=True) @ sight_base.A)
        line_of_sight._base = self.fetch_cam.fkine(
            self.fetch_cam.q, fast=True) @ sight_base.A

        # self.env.add(line_of_sight)

        self.env.step(0.01)

        return

    def stop_robot(self):
        v_base = Twist()
        self.basevel_pub.publish(v_base)

        v_joint = JointJog()
        v_joint.velocities = np.zeros(8)
        self.jointvel_pub.publish(v_joint)

        v_head = JointJog()
        v_head.velocities = np.zeros(2)
        self.headvel_pub.publish(v_head)


if __name__ == '__main__':
    try:
        rospy.init_node("mobile_manip", anonymous=True)
        controller = MobileManipController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

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

from sensor_msgs.msg import JointState
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from control_msgs.msg import JointJog, GripperCommandGoal, GripperCommandAction
from fetch_driver_msgs.msg import GripperState
import actionlib
import copy, sys
import timeit

from Controllers import *

class MobileManipController:
    def __init__(self):
        # Setup Swift environment
        self.env = swift.Swift()
        self.env.launch(realtime=False, headless=True)

        self.fetch = rtb.models.Fetch()
        self.fetch_cam = rtb.models.FetchCamera()
        self.env.add(self.fetch)
        self.env.add(self.fetch_cam)

        self.sight_base = sm.SE3.Ry(np.pi/2) * sm.SE3(0.0, 0.0, 2.5)
        self.centroid_sight = sg.Cylinder(radius=0.001, 
                                          length=5.0, 
                                          base=self.fetch_cam.fkine(self.fetch_cam.q, fast=True) @ self.sight_base.A
        )
        self.env.add(self.centroid_sight)    



        self.line_of_sight_base = sm.SE3.Ry(np.pi/2) * sm.SE3(0.0, 0.0, 2.5)
        self.line_of_sight = sg.Cylinder(radius=0.001, 
                                          length=5.0, 
                                          base=self.fetch_cam.fkine(self.fetch_cam.q, fast=True) @ self.line_of_sight_base.A
        )
        self.env.add(self.line_of_sight)    
 

        self.table = sg.Cuboid(scale=(1.0, 5.0, 1.0), 
                               base=sm.SE3(0., 0., 0.)
        )
        self.env.add(self.table)

        self.ax_goal = sg.Axes(0.1)
        self.env.add(self.ax_goal)        

        self.wTep = None
        self.wTep_waypoint = None

        self.timeout = 120

        self.env.step(0.01)

        # ROS subscribers and publishers
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.tf_listener  = tf.TransformListener()

        self.jointpos_sub = rospy.Subscriber("joint_states", JointState, self.jointpos_callback)
        
        self.jointvel_pub = rospy.Publisher("/arm_controller/joint_velocity/command", JointJog, queue_size=1)
        self.headvel_pub = rospy.Publisher("/head_controller/joint_velocity/command", JointJog, queue_size=1)

        self.basevel_pub  = rospy.Publisher("/base_controller/command", Twist, queue_size=1)
        self.gripper_action = actionlib.SimpleActionClient('/gripper_controller/gripper_action', GripperCommandAction)
        self.gripper_action.wait_for_server()

        self.controller = sys.argv[1]
        self.initial_guess = [float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])]


        self.total_count = 0
        self.seen_count = 0

        self.start_time = rospy.Time.now().to_sec()

        self.error_list = np.random.rand(10)
        self.separate_arrived = False
        self.waypoint_arrived = False

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

        # _, _, distance = self.fetch.vision_collision_damper(
        #     self.centroid_sight,
        #     self.fetch.q[:self.fetch.n],
        #     0.3,
        #     0.2,
        #     1.0,
        #     start=self.fetch.link_dict["shoulder_pan_link"],
        #     end=self.fetch.link_dict["gripper_link"],
        #     camera=self.fetch_cam,
        # )
        # if self.wTep is not None:

        try:
            # Read base frame coordinates
            (trans, rot) = self.tf_listener.lookupTransform('map', 'base_link', rospy.Time(0))

            T = sm.SE3()
            T.A[:3, :3] = sm.UnitQuaternion(rot[3], rot[:3]).R
            T.A[:3, 3] = trans

            self.fetch._base = T
            self.fetch_cam._base = T

            # Give initial guess about where the marker is
            if self.wTep is None:
                # self.wTep = T * sm.SE3([3., 0., 0.9])
                # self.wTep = T * sm.SE3([2.5, -3., 0.9])
                # self.wTep = T * sm.SE3([-0.25, 2., 0.9])
                # self.wTep = T * sm.SE3([-3., 2., 0.9])
                self.wTep = T * sm.SE3(self.initial_guess)

                # Rotation of the gripper
                goal_to_base = self.fetch._base.t - self.wTep.t
                self.wTep.A[:3, :3] = sm.SE3.Rz(np.arctan2(goal_to_base[1], goal_to_base[0]) + np.pi).A[:3, :3]
                self.wTep_waypoint = copy.deepcopy(self.wTep)
                self.ax_goal._base = self.wTep.A

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            # print(e)
            pass

        if self.wTep is None:
            return

        wTc = self.fetch_cam.fkine(self.fetch_cam.q, fast=True)
        # cTep = np.linalg.inv(wTc) @ self.Tep
        # _, head_angle, _ = transform_between_vectors(np.array([1, 0, 0]), cTep[:3, 3])

        # Draw line of sight between camera and object
        camera_pos = wTc[:3, 3]
        target_pos = self.wTep.A[:3, 3]
        middle = (camera_pos + target_pos) / 2
        R, _, _ = transform_between_vectors(np.array([0., 0., 1.]), camera_pos - target_pos)

        self.line_of_sight._base = (sm.SE3(middle) * R).A

        _, _, distance = self.fetch.vision_collision_damper(
            self.line_of_sight,
            self.fetch.q[:self.fetch.n],
            0.3,
            0.1,
            1.0,
            start=self.fetch.link_dict["shoulder_pan_link"],
            end=self.fetch.link_dict["wrist_roll_link"],
            camera=self.fetch_cam
        )



        # Spatial error
        wTe = self.fetch.fkine(self.fetch.q, fast=True)
        eTep = np.linalg.inv(wTe) @ self.wTep.A
        et = np.sum(np.abs(eTep[:3, -1]))
        # print(et)
        # print(et)

        # Read marker position
        try:
            # Read base frame coordinates
            (trans, rot) = self.tf_listener.lookupTransform('map', 'apple', rospy.Time(0))

            # Set target pose to be in front of marker
            T = sm.SE3()
            T.A[:3, :3] = sm.UnitQuaternion(rot[3], rot[:3]).R @ sm.SE3.Ry(0).R
            T.A[:3, 3] = trans

            # Only update position if detected object is close enough, and if the robot isn't too close
            # and not blocking the line-of-sight
            # print(np.linalg.norm(self.wTep.t - trans) < et / 3, et > 0.175, min(distance) > 0.1)
            # print(np.linalg.norm(self.wTep.t - trans), et, min(distance))
            if np.linalg.norm(self.wTep.t - trans) < et / 3 and et > 0.175 and min(distance) > 0.1:
                self.wTep.A[:3, 3] = trans + self.wTep.R @ np.array([0.045, 0, 0])
                self.wTep_waypoint.A[:3, 3] = trans + self.wTep.R @ np.array([-0.1, 0, 0])
                self.seen_count += 1

                # Update table position
                if et > 0.3:
                    self.table._base = (sm.SE3(sm.SE3.Rz(4.05).A[:3, :3] @ np.array([0.4, 0., -0.55])) * self.wTep).A
                    self.table._base[:3, :3] = sm.SE3.Rz(4.05).A[:3, :3]

            self.tf_broadcaster.sendTransform(tuple(self.wTep.t),
                                              tuple(sm.UnitQuaternion(self.wTep.R).A[1:]) + tuple([sm.UnitQuaternion(self.wTep.R).A[0]]),
                                              rospy.Time.now(),
                                              "goal",
                                              "map")
            
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            # print(e)
            pass

        self.total_count += 1

        if self.waypoint_arrived:
            wTep = self.wTep
            self.table._base[:3, -1] = [1000, 1000, 1000]
        else:
            wTep = self.wTep_waypoint

        # Calculate using Jesse's controller
        # start_time = timeit.default_timer()
        if self.controller == "proposed":
            arrived, qd, qd_cam = step_robot(self.fetch, self.fetch_cam, wTep.A, self.table, self.centroid_sight)
        elif self.controller == "pcamera":
            arrived, qd, qd_cam = step_p_camera(self.fetch, self.fetch_cam, wTep.A, self.table)
        elif self.controller == "holistic":
            arrived, qd, qd_cam = step_holistic(self.fetch, self.fetch_cam, wTep.A, self.table)
        elif self.controller == "separate":
            arrived = False
            if not self.separate_arrived:
                self.separate_arrived, qd, qd_cam = step_separate_base(self.fetch, self.fetch_cam, wTep.A)
            else:
                arrived, qd, qd_cam = step_separate_arm(self.fetch, self.fetch_cam, wTep.A, self.table)
                self.error_list = np.concatenate((self.error_list[1:], np.array([et])))
        else:
            print("invalid controller name!!!")
        # end_time =  timeit.default_timer()
        # print(end_time - start_time)
        if self.controller != "separate":
            self.error_list = np.concatenate((self.error_list[1:], np.array([et])))

        # print(np.std(self.error_list))

        # arrived = arrived or np.std(self.error_list) < 0.00005
        # print(et, np.std(self.error_list))
        # print(et)

        qd /= 7.5
        qd_cam /= 7.5


        _wTe = self.fetch.fkine(self.fetch.q, fast=True)
        _eTep = np.linalg.inv(_wTe) @ wTep.A
        _et = np.sum(np.abs(_eTep[:3, -1]))

        time = rospy.Time.now().to_sec() - self.start_time
        # print(_et, time, self.timeout)  

        if arrived:
            if not self.waypoint_arrived:
                self.waypoint_arrived = True
                self.timeout = time + 12                
                print("Reached waypoint")
            else:
                if time >= self.timeout:
                    print("Timeout occured: ", time, self.timeout)

                self.stop_robot()

                goal = GripperCommandGoal()
                goal.command.position = 0.0
                self.gripper_action.send_goal(goal)
                self.gripper_action.wait_for_result()

                print("\n\nExperiment complete!")
                print("Vision: ", self.seen_count / self.total_count * 100, "%")
                print("Time elapsed: ", rospy.Time.now().to_sec() - self.start_time)
                print("\n\n")

        publish=True
        # Publish base and joint velocities
        v_base = Twist()
        v_base.angular.z = qd[0]
        v_base.linear.x = qd[1]

        v_joint = JointJog()
        v_joint.velocities = qd[2:]

        v_head = JointJog()
        v_head.velocities = qd_cam[-2:]

        if publish:
            self.basevel_pub.publish(v_base)
            self.headvel_pub.publish(v_head)
            self.jointvel_pub.publish(v_joint)

        self.centroid_sight._base = self.fetch_cam.fkine(self.fetch_cam.q, fast=True) @ self.sight_base.A
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


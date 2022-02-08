#!/usr/bin/env python3
import rospy
import tf
import numpy as np
import time

import swift
import timeit

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
from rv_msgs.msg import JointVelocity, ManipulatorState
from std_msgs.msg import String

from enum import Enum

class Alg(Enum):
    Ours = 1
    Slack = 2
    NEO = 3
    PBVS = 4
    MoveIt = 5

CURRENT_ALG = Alg.Ours

if CURRENT_ALG == Alg.Ours:
    from ours import Ours as Controller
elif CURRENT_ALG == Alg.NEO:
    from neo import NEO as Controller



import actionlib

def transform_between_vectors(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    angle = np.arccos(np.dot(a, b))
    axis = np.cross(a, b)

    return sm.SE3.AngleAxis(angle, axis), angle, axis

class StationaryManipController:
    def __init__(self):

        self.SIMULATED = False

        self.env = swift.Swift()
        self.env.launch(realtime=False, headless=False)

        self.panda = rtb.models.Panda()
        if self.SIMULATED:
            self.panda.q = self.panda.qr
        self.env.add(self.panda)

        self.wTep = None

        self.qd = np.zeros(7)

        self.controller = Controller()

        self.camera = None
        self.NUM_OBJECTS = 1
        self.min_idx = None
        self.obj = [None] * self.NUM_OBJECTS
        self.collisions = []

        self.initialised = False
        self.arrived = False
        self.waypoint = False

        self.ax_goal = sg.Axes(0.1)
        self.env.add(self.ax_goal)  

        self.table = sg.Cuboid(scale=(2.0, 2.0, 0.1), base=sm.SE3(0., 0., -0.05))
        self.env.add(self.table)

        self.controller.init([], np.array([0.5, 0.5, 1.0]), self.table)

        self.occluded = [0] * self.NUM_OBJECTS
        self.start_time = None

        self.tf_broadcaster = tf.TransformBroadcaster()
        self.tf_listener  = tf.TransformListener()

        self.jointpos_sub = rospy.Subscriber("/franka_state_controller/joint_states", JointState, self.jointpos_callback, queue_size=1)
        self.finish_pub = rospy.Publisher("kerrys_prviate_topic", String, queue_size=1)
        self.jointvel_pub = rospy.Publisher("/arm/joint/velocity", JointVelocity, queue_size=1)

        self.FINISHED = False
        self.gripper_action = actionlib.SimpleActionClient('/franka_gripper/gripper_action', GripperCommandAction)
        self.gripper_action.wait_for_server()

        self.main_timer = rospy.Timer(rospy.Duration(0.002), self.main_callback)

        

        # Apple detection and velocity estimation
        self.apple_dt = 1 / 30.0

        self.vel_x = [0] * 5
        self.vel_y = [0] * 5

        self.prev_pos = None

        self.apple_timer = rospy.Timer(rospy.Duration(self.apple_dt), self.apple_callback)



        # Metrics
        self.count = 0
        self.total_dist = 0

        self.metric_mode = False



    def apple_callback(self, event):
        try:
            (trans, rot) = self.tf_listener.lookupTransform('panda_link0', 'apple0', rospy.Time(0))
            self.obj[0] = sm.SE3(trans)
        except:
            return

        if self.prev_pos is None:
            self.prev_pos = [trans[0], trans[1]]
            return

        current_vel_x = (self.prev_pos[0] - trans[0]) / self.apple_dt
        current_vel_y = (self.prev_pos[1] - trans[1]) / self.apple_dt

        self.vel_x = [current_vel_x] + self.vel_x[:-1]
        self.vel_y = [current_vel_y] + self.vel_y[:-1]
      
        if len(self.collisions) > 0:
            self.collisions[0].v[:2] = np.array([np.mean(self.vel_x), np.mean(self.vel_y)])

        self.prev_pos = [trans[0], trans[1]]        



    def initialise_collision(self):
        # Read all positions of camera and objects
        try:
            (trans, rot) = self.tf_listener.lookupTransform('panda_link0', 'camera_rgb_frame', rospy.Time(0))
            self.camera = sm.SE3(trans)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            # print(e)
            pass

        # for i in range(self.NUM_OBJECTS):
        #     try:
        #         (trans, rot) = self.tf_listener.lookupTransform('panda_link0', 'apple' + str(i), rospy.Time(0))
        #         self.obj[i] = sm.SE3(trans)
                
        #     except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
        #         # print(e)
        #         pass



        self.initialised = all(self.obj + [self.camera])

        # Create line of sight collisions for each object
        if self.initialised:
            for i in range(self.NUM_OBJECTS):
                camera_pos = self.camera.t
                target_pos = self.obj[i].t

                middle = (camera_pos + target_pos) / 2
                R, _, _ = transform_between_vectors(
                    np.array([0.0, 0.0, 1.0]), camera_pos - target_pos
                )
                # print(sm.SE3(middle) * R)

                # line of sight between camera and object we want to avoid
                if len(self.collisions) == 0:
                    s0 = sg.Cylinder(
                        radius=0.001,
                        length=np.linalg.norm(camera_pos - target_pos),
                        base=sm.SE3(middle) * R,
                    )
                    self.collisions.append(s0)
                    self.env.add(s0)
                else:
                    self.collisions[0]._base = (sm.SE3(middle) * R).A
        

            # Find out where the desired grasped object is
            self.min_idx = 0

            self.wTep = sm.SE3(self.obj[0]) * sm.SE3.Rx(np.pi)
            self.wTep.A[2, -1] = 0.3

            if CURRENT_ALG == Alg.NEO:
                gripper_z_axis = np.subtract(self.wTep.A[:3, 3], [0.22026039175, -0.0303012059091, 0.497918336168])
                gripper_z_axis = gripper_z_axis / np.linalg.norm(gripper_z_axis)
                gripper_y_axis = np.cross(gripper_z_axis, [0,0,1])
                gripper_x_axis = np.cross(gripper_y_axis, [0,0,-1])

                self.wTep.A[:3, 0] = gripper_x_axis
                self.wTep.A[:3, 1] = gripper_y_axis
                self.wTep.A[:3, 2] = [0,0,-1]

            

    def jointpos_callback(self, data): 
        start_time = timeit.default_timer()

        self.initialise_collision()
        if self.SIMULATED:
            self.env.step(0.025)
        else:
            self.env.step(0.001)     

        if not self.initialised:
            return

        # Read joint positions
        if not self.SIMULATED:
            self.panda.q = np.array([data.position[0],
                                     data.position[1],
                                     data.position[2],
                                     data.position[3],
                                     data.position[4],
                                     data.position[5],
                                     data.position[6]])

        # if self.wTep is None:
        #     self.wTep = self.panda.fkine(self.panda.q) * sm.SE3.Tx(0.0) * sm.SE3.Ty(0.0) * sm.SE3.Tz(0.1)

        # v, arrived = rtb.p_servo(self.panda.fkine(self.panda.q), self.wTep, 1)
        # self.qd = np.linalg.pinv(self.panda.jacobe(self.panda.q)) @ v    

        self.ax_goal.base = sm.SE3(0., 0., 0.1) * self.wTep
        qd, _, occluded = self.controller.step(self.panda,
            sm.SE3(0., 0., 0.1) * self.wTep, 
            self.NUM_OBJECTS,
            self.panda.n,
            self.collisions,
            self.table,
            self.min_idx,
            self.camera.t,
            self.obj[0].t)

        self.qd = qd[:self.panda.n] / 2
        self.occluded = np.add(occluded, self.occluded)
 

        if self.SIMULATED:
            self.panda.qd = self.qd



        # Calculate distance metric
        if not self.metric_mode:
            # If apple is stationary, assume experiment hasn't started or has finished
            self.metric_mode = np.linalg.norm(self.collisions[0].v[:2]) > 0.075
            print("Average distance error metric: ", self.total_dist / max(1, self.count))
        else:
            # Else, update the distance metric
            p_panda = self.panda.fkine(self.panda.q).t
            p_apple = self.obj[0].t

            self.total_dist += np.linalg.norm((p_apple - p_panda)[:2])
            self.count += 1

            self.metric_mode = np.linalg.norm(self.collisions[0].v[:2]) > 0.025
        
            
    def main_callback(self, event):
        # Publish base and joint velocities
        if not self.arrived:
            qd = JointVelocity()
            qd.joints = list(self.qd)
            # print(qd)
            if not self.SIMULATED:
                self.jointvel_pub.publish(qd)
        elif not self.SIMULATED:
            if not self.FINISHED:
                total_time = timeit.default_timer() - self.start_time
                
                print("total_time:", total_time)
                print("vision score", self.occluded)

                self.finish_pub.publish("")
                self.FINISHED = True

            # goal = GripperCommandGoal()
            # goal.command.position = 0.0
            # self.gripper_action.send_goal(goal)
            # self.gripper_action.wait_for_result()


        else:
            print("SIMULATION COMPLETE :)")

        return     


if __name__ == '__main__':
    try:
        rospy.init_node("stationary_manip", anonymous=True)
        controller = StationaryManipController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


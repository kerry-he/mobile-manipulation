#!/usr/bin/env python3
import rospy
import tf
import numpy as np
import time
from datetime import datetime


import swift
import timeit

import roboticstoolbox as rtb
import spatialgeometry as sg
import spatialmath as sm
import qpsolvers as qp
import numpy as np
import math
import cv2

from sensor_msgs.msg import JointState
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from control_msgs.msg import JointJog, GripperCommandGoal, GripperCommandAction
from rv_msgs.msg import JointVelocity, ManipulatorState
from std_msgs.msg import String

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

from enum import Enum

from std_srvs.srv import Empty
from dynamic_workspace_controller.srv import WorkspaceTrajectory

class Alg(Enum):
    Ours = 1
    NEO = 2
 

CURRENT_ALG = Alg.NEO
# SPEED = 100 # mm/s
# ARGS = []
# TRAJ_NAME = "Flower"


SPEEDS = [50, 100]
TRAJECTORIES = [
    ("Circle", []), 
    ("Flower", []),
    ("DoubleInfinity", []),
    ("DiagonalDoubleInifinity", []),
    ("InscribedCircle", []),
    ("RadiusedRandomWalk", [100, 50.0/140, 10.0/140, 0]),
    ("RadiusedRandomWalk", [100, 50.0/140, 10.0/140, 1]),
    ("Rectangle", [200.0, 200.0]),
    ("Rectangle", [50.0, 200.0]),
    ("Rectangle", [200.0, 50.0])
]

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


STARTUP, INITTING, RUNNING, FINISHED = 0, 1, 2, 3

class StationaryManipController:
    def __init__(self):
        self.trial_number = 0

        self.SIMULATED = False

        self.state = STARTUP

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

        self.image_sub = rospy.Subscriber(
            "/camera/color/image_raw", Image, self.image_callback, queue_size=1)

        self.bridge = CvBridge()
        self.video_out = None
        self.image_shape = None

        
        self.log_file = open("results.csv", "a")

        # Metrics
        self.count = 0
        self.total_dist = 0

        self.metric_mode = False

        # Service for sending robot home
        rospy.wait_for_service('/arm/home')

        self.go_home = rospy.ServiceProxy('/arm/home', Empty)

        rospy.wait_for_service('/run_workspace_trajectory')
        self.run_workspace_traj = rospy.ServiceProxy('/run_workspace_trajectory', WorkspaceTrajectory)

        rospy.wait_for_service('/reset_workspace')

        self.reset_workspace_top = rospy.ServiceProxy('/reset_workspace_top', Empty)
        self.reset_workspace_center = rospy.ServiceProxy('/reset_workspace', Empty)

        # if TRAJ_NAME in ["Circle"]:
        #     self.reset_workspace = rospy.ServiceProxy('/reset_workspace_top', Empty)
        # else:
        #     self.reset_workspace = rospy.ServiceProxy('/reset_workspace', Empty)
        

        print("Finished init")





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
            (trans, rot) = self.tf_listener.lookupTransform('panda_link0', 'overhead_cam', rospy.Time(0))
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


        if self.metric_mode:
            p_panda = self.panda.fkine(self.panda.q).t
            p_apple = self.obj[0].t

            self.total_dist += np.linalg.norm((p_apple - p_panda)[:2])
            self.count += 1

        # Calculate distance metric
        # if not self.metric_mode:
        #     # If apple is stationary, assume experiment hasn't started or has finished
        #     self.metric_mode = np.linalg.norm(self.collisions[0].v[:2]) > 0.075
        #     print("Average distance error metric: ", self.total_dist / max(1, self.count))
        # else:
        #     # Else, update the distance metric
        #     p_panda = self.panda.fkine(self.panda.q).t
        #     p_apple = self.obj[0].t

        #     self.total_dist += np.linalg.norm((p_apple - p_panda)[:2])
        #     self.count += 1

        #     self.metric_mode = np.linalg.norm(self.collisions[0].v[:2]) > 0.025
        
    
    def change_state(self, new_state):
        self.state_change_time = time.time()
        self.state = new_state
        print("Changing to state: ", self.state)

    def image_callback(self, data):
        if self.video_out is not None:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
                self.video_out.write(cv_image)
            except CvBridgeError as e:
                print(e)


        # Will only happen once to get image shape
        if self.image_shape is None:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
                self.image_shape = cv_image.shape
            except CvBridgeError as e:
                print(e)


    def create_video_writer(self, trajectory_name, speed, trial_number, method):
        if method == Alg.Ours:
            method = "VMC"
        elif method == Alg.NEO:
            method = "NEO"

        date_time_str = datetime.now().strftime("%H:%M:%S")
        filename_base = f"trial_{trajectory_name}_speed{speed}_{method}_{trial_number}_{date_time_str}"
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.video_out = cv2.VideoWriter(filename_base + ".avi", fourcc, 30.0, (self.image_shape[1], self.image_shape[0]))
        self.trial_log_file = open(filename_base + ".csv", "w")
        self.trial_log_file.write("Time, EE x, EE y, EE z, EE qx, EE qy, EE qz, EE qw, Observed x, Observed y, Observed z, Actual x, Actual y, Actual z, Joint angles...\n")

    def log_data(self):
        (ee_trans, ee_rot) = self.tf_listener.lookupTransform('panda_link0', 'panda_EE', rospy.Time(0))
        (observed_trans, ee_rot) = self.tf_listener.lookupTransform('panda_link0', 'apple0', rospy.Time(0))
        (actual_trans, ee_rot) = self.tf_listener.lookupTransform('panda_link0', 'dynamic_workspace', rospy.Time(0))

        log_vals = [time.time() - self.state_change_time, *ee_trans, *ee_rot, *observed_trans, *actual_trans, *self.panda.q]
        log_str = ",".join([str(v) for v in log_vals])
        # print(log_str)

        self.trial_log_file.write(
            log_str + "\n"
        )


    def main_callback(self, event):
        if self.initialised:
            traj_name, args = TRAJECTORIES[int(self.trial_number / len(SPEEDS))]
            speed = SPEEDS[self.trial_number % len(SPEEDS)]

            if traj_name in ["Circle", "Rectangle"]:
                self.reset_workspace = self.reset_workspace_top
            else:
                self.reset_workspace = self.reset_workspace_center

        if self.state == STARTUP:
            if self.initialised:
                self.change_state(INITTING)
            pass

        elif self.state == INITTING:

            

            self.reset_workspace()
            self.go_home()
            time.sleep(2)

            self.total_dist = 0
            self.count = 0
            self.metric_mode = True
            

            self.change_state(RUNNING)
            
            self.create_video_writer(traj_name, speed, self.trial_number, CURRENT_ALG)
            self.run_workspace_traj(traj_name, speed * 60, args)

        elif self.state == RUNNING:
            
            qd = JointVelocity()
            qd.joints = list(self.qd)
            
            if not self.SIMULATED:
                self.jointvel_pub.publish(qd)
            
            self.log_data()
            


        
            if time.time() - self.state_change_time > 30:
                # Experiment running for x seconds. Time to stop it. 
                self.change_state(FINISHED)

        elif self.state == FINISHED:
            print("Experiment Finished!")
            ave_error = self.total_dist / max(1, self.count)
            print("Average distance error metric: ", ave_error)

            self.log_file.write(f"{traj_name}, {speed}, {self.trial_number}, {CURRENT_ALG}, {ave_error}\n")
            self.log_file.flush()
            self.state = -1
            self.video_out.release()
            self.video_out = None

            self.trial_log_file.close()

            self.go_home()
            self.reset_workspace()
            self.trial_number += 1
            time.sleep(5)
            self.state = INITTING

        
        


        # Publish base and joint velocities
        # if not self.arrived:
        #     qd = JointVelocity()
        #     qd.joints = list(self.qd)
        #     # print(qd)
        #     if not self.SIMULATED:
        #         self.jointvel_pub.publish(qd)
        # elif not self.SIMULATED:
        #     if not self.FINISHED:
        #         total_time = timeit.default_timer() - self.start_time
                
        #         print("total_time:", total_time)
        #         print("vision score", self.occluded)

        #         self.finish_pub.publish("")
        #         self.FINISHED = True

            # goal = GripperCommandGoal()
            # goal.command.position = 0.0
            # self.gripper_action.send_goal(goal)
            # self.gripper_action.wait_for_result()


        # else:
        #     print("SIMULATION COMPLETE :)")

        return     

if __name__ == '__main__':
    try:
        rospy.init_node("stationary_manip", anonymous=True)
        controller = StationaryManipController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


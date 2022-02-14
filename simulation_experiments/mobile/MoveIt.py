"""
@author Rhys Newbury, Tin Tran
"""

from baseController import BaseController
import numpy as np
import math
import roboticstoolbox as rtb
import spatialgeometry as sg
import spatialmath as sm
import qpsolvers as qp
from scipy.spatial.transform import Rotation as R
from visualization_msgs.msg import MarkerArray

import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from moveit_msgs.srv import GetStateValidity, GetStateValidityRequest
import copy
import timeit
import spatialmath as sm
from baseController import BaseController
import tf
# from moveit_msgs.srv import GetStateValidity
from moveit_msgs.msg import VisibilityConstraint, Constraints
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import AttachedCollisionObject


CONSIDER_COLLISIONS = True


class MoveIt(BaseController):

    def __init__(self):

        rospy.init_node('kerry_MoveIt')
        self.group_name = "arm_with_torso"
        self.arm_commander = moveit_commander.MoveGroupCommander(
            self.group_name)

        self.consider_colls = CONSIDER_COLLISIONS

        self.listener = tf.TransformListener()
        
        self.robot = moveit_commander.RobotCommander()
        self.display_trajectory_publisher = rospy.Publisher(
            '/move_group/display_planned_path',
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20)

        self.cylinder_name = "camera_vision"
        self.scene = moveit_commander.PlanningSceneInterface()

        self.marker_pub = rospy.Publisher(
            '/markersssssss',
            MarkerArray,
            queue_size=20)


        self.index = 0
        self.planningTime = 0

        rospy.wait_for_service("check_state_validity")
        self.state_valid_service = rospy.ServiceProxy(
            'check_state_validity', GetStateValidity)
        # rospy.wait_for_service("/check_state_validity")
        # self.checkCollison = rospy.ServiceProxy("/check_state_validity", GetStateValidity)

    def move_camera_pose(self, joint_angles):
        print("not sure if this is needed")

    def init(self, init_joint_angles, init_head_angle):

        self.separate_arrived = False
        self.finished_ang = False

        self.move_joint_angle(init_joint_angles)
        self.move_camera_pose(init_head_angle)

        self.index = 0
        self.planningTime = 0
        self.failed = False
        self.prev_timestep = 0.025

        # robot_state = self.commmander.get_current_state()
        # robot_state.position = init_joint_angles



    def move_joint_angle(self, angles):

        # The go command can be called with joint values, poses, or without any
        # parameters if you have already set the pose or joint target for the group
        joint_goal = self.arm_commander.get_current_joint_values()
        
        # print("joint_goal", joint_goal)


        joint_goal[:] = angles[2:]

        
        
        # joint_goal[0] = angles[1]
        # print(len(angles), joint_goal, len(joint_goal))
        self.arm_commander.go(joint_goal, wait=True)

        # Calling ``stop()`` ensures that there is no residual movement
        self.arm_commander.stop()

    def step(self, r, r_cam, Tep, centroid_sight):
        if not self.separate_arrived:
            self.separate_arrived, qd, camera_qd = self.step_separate_base(
                r, r_cam, Tep)

            if self.separate_arrived:
                self.plan_success = self.plan_arm(
                    r, r_cam, Tep, centroid_sight)

            return False, qd, camera_qd
        else:
            return self.step_separate_arm(r, r_cam, Tep)

    def generateValidArmConfig(self):
        while True:
            #   - l_wheel_joint
            #   - r_wheel_joint
            #   - torso_lift_joint
            #   - bellows_joint
            #   - head_pan_joint
            #   - head_tilt_joint
            #   - shoulder_pan_joint
            #   - shoulder_lift_joint
            #   - upperarm_roll_joint
            #   - elbow_flex_joint
            #   - forearm_roll_joint
            #   - wrist_flex_joint
            #   - wrist_roll_joint
            #   - l_gripper_finger_joint
            #   - r_gripper_finger_joint
            # original joint generation code
            q = np.random.uniform(low=[0, 0, 0, 0, 0, 0, -1.6056, -1.221, 0, -2.251, 0, -2.160, 0, 0.35, 0.35],
                                  high=[0, 0, 0, 0, 0, 0, 1.6056, 1.518, 6.283,
                                  2.251, 6.283, 2.160, 6.283, 0.35, 0.35],
                                  size=15)


            req = GetStateValidityRequest()
            req.group_name = self.group_name
            req.robot_state = self.arm_commander.get_current_state()
            # print(req.robot_state.joint_state)
            # input()
            req.robot_state.joint_state.position = list(q)
            # print(req)
            res = self.state_valid_service(req)

            if res.valid:
                # TODO get subset of joints
                return q[3:13]
            else:
                print("INVALID JOINT STATE")

    def getCameraPosition(self):
        self.listener.waitForTransform("/base_link", "/head_camera_rgb_frame", rospy.Time(), rospy.Duration(4.0))
        (trans,rot) = self.listener.lookupTransform('/base_link', '/head_camera_rgb_frame', rospy.Time(0))
        return trans

    def getCameraPose(self):
        self.listener.waitForTransform("/base_link", "/head_camera_rgb_frame", rospy.Time(), rospy.Duration(4.0))
        (trans,rot) = self.listener.lookupTransform('/base_link', '/head_camera_rgb_frame', rospy.Time(0))

        p = PoseStamped()

        p.header.frame_id = "head_camera_rgb_frame"

        p.pose.position.x = trans[0]
        p.pose.position.y = trans[1]
        p.pose.position.z = trans[2]

        p.pose.orientation.x = rot[0]
        p.pose.orientation.y = rot[1]
        p.pose.orientation.z = rot[2]
        p.pose.orientation.w = rot[3]

        return p

    def plan_arm(self, r, r_cam, Tep, centroid_sight):

        # wTc = r_cam.fkine(r_cam.q, fast=True)
        wTb = r._base.A
        bTw = np.linalg.inv(wTb)
        # bTc = wTc @ bTw

        bTo = bTw @ Tep
        bTc = self.getCameraPosition()
        # print(Tep)
        # print(wTc)
        # input()
        for offset in np.linspace(0, 1, num=10):
            if CONSIDER_COLLISIONS:
                success = self.add_vision_ray(
                    camera_pose = bTc, 
                    object_pose = bTo[:3, 3],
                    offset_percentage = offset,
                    index = 0)            


            position = bTo[:3, 3]

            # orientation = sm.UnitQuaternion(Tep.A[:3, :3]).A
            orientation = R.from_matrix(bTo[:3, :3]).as_quat()
            success = self.move_pose(position, orientation)
            self.curr_time = 0

            if success:
                return True
            else:
                self.remove_vision_ray(0)

    def add_vision_ray(self, camera_pose, object_pose, offset_percentage=0.01, index=0):
        '''
        camera_pose: np.array [x, y, z]
        object_pose: np.array [x, y, z]
        '''

        p = geometry_msgs.msg.PoseStamped()
        p.header.frame_id = self.arm_commander.get_planning_frame()

        center = (object_pose + camera_pose)/2
        length = np.linalg.norm(object_pose - camera_pose)
        offset_distance = offset_percentage * length
        cylinder_orientation_vector_z = (object_pose - camera_pose) / length
        center_offset = -1 * cylinder_orientation_vector_z * \
            (offset_distance/2)
        axis = np.cross(np.array([0, 0, 1]), cylinder_orientation_vector_z)
        angle = np.arccos(
            np.dot(np.array([0, 0, 1]), cylinder_orientation_vector_z))
        angle_axis = sm.SE3.AngleAxis(angle, axis)
        orientation = R.from_matrix(angle_axis.A[0:3, 0:3]).as_quat()

        center = center + cylinder_orientation_vector_z * 0.03
        
        p.pose.position.x = center[0] + center_offset[0]
        p.pose.position.y = center[1] + center_offset[1]
        p.pose.position.z = center[2] + center_offset[2]
        p.pose.orientation.x = orientation[0]
        p.pose.orientation.y = orientation[1]
        p.pose.orientation.z = orientation[2]
        p.pose.orientation.w = orientation[3]

        self.scene.add_cylinder(self.generate_cylinder_name(
            index), p, length - offset_distance, 0.05)
        return self.check_object(cylinder_exists=True, name=self.generate_cylinder_name(index))

    def step_separate_base(self, r, r_cam, Tep):

        # wTe = r.fkine(r.q, fast=True)
        # wTe = r._base.A[:] / 2 + r.fkine(r.q, fast=True) / 2
        wTe = r._base.A[:] @ sm.SE3(1.1281 * 3 / 5, 0, 0).A

        Tep_temp = sm.SE3(Tep[:3, 3]).A
        Tep_temp[:2, 3] -= Tep[:2, 0] * 0.2
        eTep = np.linalg.inv(wTe) @ Tep

        # Spatial error
        et = np.sum(np.abs(eTep[:2, -1]))

        # Gain term (lambda) for control minimisation
        Y = 0.01

        # Quadratic component of objective function
        Q = np.eye(r.n + 6)

        # Joint velocity component of Q
        Q[: r.n, : r.n] *= Y
        Q[:2, :2] *= 1.0 / et

        # Slack component of Q
        Q[r.n:, r.n:] = (1.0 / et) * np.eye(6)


        v[2:] *= 0

        # The equality contraints
        Aeq = np.c_[r.jacobe(r.q, start="base0", end="base_link", tool=sm.SE3(
            1.1281, 0, 0).A, fast=True), np.zeros((6, 8)), np.eye(6)]
        beq = v.reshape((6,))

    # def step_pid(r, r_cam, Tep):
    def step_separate_base(self, r, r_cam, Tep):

        wTb = r._base.A

        bTbp = np.linalg.inv(wTb) @ Tep

        # Spatial error
        bt = np.sum(np.abs(bTbp[:2, -1]))

        vb_lin = (np.linalg.norm(bTbp[:2, -1]) - 0.59) * 5
        vb_ang = np.arctan2(bTbp[1, -1], bTbp[0, -1]) * 35 * min(1.0, vb_lin/8.0)

        vb_lin = max(min(vb_lin, r_cam.qdlim[1]), -r_cam.qdlim[1])
        vb_ang = max(min(vb_ang, r_cam.qdlim[0]), -r_cam.qdlim[0])  
        

        if not self.finished_ang:
            vb_lin = 0
            self.finished_ang = abs(vb_ang / 50) < 0.01

        if bt < 0.6:
            arrived = True
            vb_lin = 0.0
            vb_ang = 0.0
        else:
            arrived = False
        # print("bt", bt, bt < 1.15)
        # print(vb_lin, vb_ang)
        # print("bTbp[:3, -1]", bTbp[:3, -1])


        # Simple camera PID
        wTc = r_cam.fkine(r_cam.q, fast=True)
        cTep = np.linalg.inv(wTc) @ Tep

        # Spatial error
        head_rotation, head_angle, _ = BaseController.transform_between_vectors(
            np.array([1, 0, 0]), cTep[:3, 3])

        yaw = max(min(head_rotation.rpy()[
                  2] * 10, r_cam.qdlim[3]), -r_cam.qdlim[3])
        pitch = max(
            min(head_rotation.rpy()[1] * 10, r_cam.qdlim[4]), -r_cam.qdlim[4])

        # Solve for the joint velocities dq
        qd = np.array([vb_ang, vb_lin, 0., 0., 0., 0., 0., 0., 0., 0.])
        qd_cam = np.array([vb_ang, vb_lin, 0., yaw, pitch])

        if bt > 0.5:
            qd *= 0.7 / bt
            qd_cam *= 0.7 / bt
        else:
            qd *= 1.4
            qd_cam *= 1.4

        return arrived, qd, qd_cam        


    def check_object(self, cylinder_exists, name, timeout=4):
        cylinder_name = name
        start = rospy.get_time()
        seconds = rospy.get_time()
        while (seconds - start < timeout) and not rospy.is_shutdown():
            does_exist = cylinder_name in self.scene.get_known_object_names()
            # Test if we are in the expected state
            if (cylinder_exists == does_exist):
                return True
            # Sleep so that we give other threads time on the processor
            rospy.sleep(0.1)

            seconds = rospy.get_time()
        return False

    def remove_vision_ray(self, index):
        self.scene.remove_world_object(self.generate_cylinder_name(index))
        return self.check_object(cylinder_exists=False, name=self.generate_cylinder_name(index))

    def generate_cylinder_name(self, index):
        return f"{self.cylinder_name}_{index}"

    def move_pose(self, pose, quaternion):

        pose_goal = self.arm_commander.get_current_pose().pose

        pose_goal.position.x = pose[0]
        pose_goal.position.y = pose[1]
        pose_goal.position.z = pose[2]

        # scalar part should be first
        pose_goal.orientation.x = float(quaternion[0])
        pose_goal.orientation.y = float(quaternion[1])
        pose_goal.orientation.z = float(quaternion[2])
        pose_goal.orientation.w = float(quaternion[3])

        stamped = PoseStamped()
        stamped.header.frame_id = "base_link"
        stamped.pose = pose_goal

        # vc = VisibilityConstraint()
        # vc.target_pose = stamped
        # vc.sensor_pose = self.getCameraPose()
        # vc.cone_sides = 29
        # vc.max_view_angle = 45
        # vc.max_range_angle = 45
        # vc.sensor_view_direction = 2
        # vc.weight = 1.0

        # # self.marker_pub.publish(vc.get_markers())

        # cons = Constraints()
        # cons.visibility_constraints.append(vc)
        # self.arm_commander.set_path_constraints(cons)

        # print(self.arm_commander.get_path_constraints())
        # input()

        # vc.sensor_frame_id

        # pose_goal = geometry_msgs.msg.Pose()
        # pose_goal.orientation.w = 1.0
        # pose_goal.position.x = 0.4
        # pose_goal.position.y = 0.1
        # pose_goal.position.z = 0.4
        self.arm_commander.set_pose_target(pose_goal)
        start_time = timeit.default_timer()
        plan = self.arm_commander.plan()

        # input("help")


        self.arm_commander.clear_path_constraints()
        
        end_time = timeit.default_timer()

        self.planningTime = end_time - start_time        
        # self.arm_commander.execute(plan[1])
        self.plan = plan[1]
        return plan[0]

    def cleanup(self):
        self.scene.remove_world_object(self.generate_cylinder_name(0))
        self.check_object(cylinder_exists=False,
                          name=self.generate_cylinder_name(0))

    def step_separate_arm(self, r, r_cam, Tep):

        if not self.plan_success:
            # raise NotImplementedError("todo")
            self.failed = True
            return True, [0]*10, [0]*5

        step = self.plan.joint_trajectory.points[self.index]
        qd = step.velocities

        self.index += 1
        arrived = self.index == len(self.plan.joint_trajectory.points) - 1

        time_from_start = step.time_from_start.secs + \
            step.time_from_start.nsecs/1000000000
        self.prev_timestep = time_from_start - self.curr_time
        self.curr_time = time_from_start



        # Simple camera PID
        wTc = r_cam.fkine(r_cam.q, fast=True)
        cTep = np.linalg.inv(wTc) @ Tep

        # Spatial error
        head_rotation, head_angle, _ = BaseController.transform_between_vectors(np.array([1, 0, 0]), cTep[:3, 3])

        yaw = max(min(head_rotation.rpy()[2] * 10, r_cam.qdlim[3]), -r_cam.qdlim[3])
        pitch = max(min(head_rotation.rpy()[1] * 10, r_cam.qdlim[4]), -r_cam.qdlim[4])



        # print("qd", qd)

        return arrived, (0, 0) + qd, [0,0, qd[0], yaw, pitch]  

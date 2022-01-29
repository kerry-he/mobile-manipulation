

from baseController import BaseController
import numpy as np, math
from resources.static_camera.MoveIt import CONSIDER_COLLISIONS
import roboticstoolbox as rtb
import spatialgeometry as sg
import spatialmath as sm
import qpsolvers as qp
from scipy.spatial.transform import Rotation as R


import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import copy, timeit
import spatialmath as sm
from baseController import BaseController

CONSIDER_COLLISIONS = True


class MoveIt(BaseController):

    def __init__(self):
        
        rospy.init_node('kerry_MoveIt')
        self.arm_commander = moveit_commander.MoveGroupCommander('manipulator')
        
        self.robot = moveit_commander.RobotCommander()
        self.display_trajectory_publisher = rospy.Publisher(
            '/move_group/display_planned_path',
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20)

        self.cylinder_name = "camera_vision"
        self.scene = moveit_commander.PlanningSceneInterface()

        self.index = 0
        self.planningTime = 0

    def move_camera_pose(self, joint_angles):
        # TODO
        pass

    def init(self, init_joint_angles, init_head_angle):

        self.seperate_arrived = False

        self.move_joint_angle(init_joint_angles)
        self.move_camera_pose(init_head_angle)

        self.index = 0
        self.planningTime = 0

    def step(self, r, r_cam, Tep, centroid_sight):
        if not self.separate_arrived:
            self.separate_arrived, qd, camera_qd = self.step_separate_base(r, r_cam, Tep)

            if self.seperate_arrived:
                self.plan_success = self.plan_arm(r, r_cam, Tep, centroid_sight)

            return self.seperate_arrived, qd, camera_qd
        else:
            return self.step_separate_arm(r, r_cam, Tep)

    def plan_arm(self, r, r_cam, Tep, centroid_sight):


        wTc = r_cam.fkine(r_cam.q, fast=True)
        
        for offset in np.linspace(0, 1, num=10):
            if CONSIDER_COLLISIONS:
                success = self.add_vision_ray(
                    camera_pose = wTc.A[:3, 3], 
                    object_pose = Tep.A[:3, 3],
                    offset_percentage = offset,
                    index = 0)            


            position = Tep.A[:3, 3]

            # orientation = sm.UnitQuaternion(Tep.A[:3, :3]).A
            orientation = R.from_matrix(Tep.A[:3, :3]).as_quat()
            success = self.move_pose(position, orientation)
            self.curr_time = 0

            if success:
                return True
            else:
                self.remove_vision_ray(0)


    def add_vision_ray(self, camera_pose, object_pose, offset_percentage = 0.01, index=0):
        '''
        camera_pose: np.array [x, y, z]
        object_pose: np.array [x, y, z]
        '''

        p = geometry_msgs.msg.PoseStamped()
        p.header.frame_id = self.commander.get_planning_frame()

        center = (object_pose + camera_pose)/2
        length = np.linalg.norm(object_pose - camera_pose)
        offset_distance = offset_percentage * length
        cylinder_orientation_vector_z = (object_pose - camera_pose) / length
        center_offset = -1 * cylinder_orientation_vector_z * (offset_distance/2)
        axis = np.cross(np.array([0,0,1]), cylinder_orientation_vector_z)
        angle = np.arccos(np.dot(np.array([0,0,1]), cylinder_orientation_vector_z))
        angle_axis = sm.SE3.AngleAxis(angle, axis)
        orientation = R.from_matrix(angle_axis.A[0:3, 0:3]).as_quat()
        
        p.pose.position.x = center[0] + center_offset[0]
        p.pose.position.y = center[1] + center_offset[1]
        p.pose.position.z = center[2] + center_offset[2]
        p.pose.orientation.x = orientation[0]
        p.pose.orientation.y = orientation[1]
        p.pose.orientation.z = orientation[2]
        p.pose.orientation.w = orientation[3]

        self.scene.add_cylinder(self.generate_cylinder_name(index), p, length - offset_distance, 0.05)
        return self.check_object(cylinder_exists=True, name = self.generate_cylinder_name(index))


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
        Q[r.n :, r.n :] = (1.0 / et) * np.eye(6)

        v, _ = rtb.p_servo(wTe, Tep, 1.5)

        v[2:] *= 0

        # The equality contraints
        Aeq = np.c_[r.jacobe(r.q, start="base0", end="base_link", tool=sm.SE3(1.1281, 0, 0).A, fast=True), np.zeros((6, 8)), np.eye(6)]
        beq = v.reshape((6,))

        Aeq_arm = np.c_[np.zeros((8, 2)), np.eye(8), np.zeros((8, 6))]
        beq_arm = np.zeros((8,))

        Aeq = np.r_[Aeq, Aeq_arm]
        beq = np.r_[beq, beq_arm]

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
        Ain[2:, 2:] = 0
        bin[2:] = 0

        # Linear component of objective function: the manipulability Jacobian
        # c = np.concatenate(
        #     (np.zeros(2), -r.jacobm(start=r.links[3]).reshape((r.n - 2,)), np.zeros(6))
        # )
        c = np.concatenate(
            (np.zeros(2), np.zeros(8), np.zeros(6))
        )    

        # Get base to face end-effector
        kε = 0.5
        bTe = r.fkine(r.q, include_base=False, fast=True)
        θε = math.atan2(bTe[1, -1], bTe[0, -1])
        ε = kε * θε
        c[0] = -0

        # The lower and upper bounds on the joint velocity and slack variable
        lb = -np.r_[r.qdlim[: r.n], 100 * np.ones(6)]
        ub = np.r_[r.qdlim[: r.n], 100 * np.ones(6)]


        # Simple camera PID
        wTc = r_cam.fkine(r_cam.q, fast=True)
        cTep = np.linalg.inv(wTc) @ Tep

        # Spatial error
        head_rotation, head_angle, _ = BaseController.transform_between_vectors(np.array([1, 0, 0]), cTep[:3, 3])

        yaw = max(min(head_rotation.rpy()[2] * 10, r_cam.qdlim[3]), -r_cam.qdlim[3])
        pitch = max(min(head_rotation.rpy()[1] * 10, r_cam.qdlim[4]), -r_cam.qdlim[4])


        # Solve for the joint velocities dq
        qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub)
        qd = qd[: r.n]

        qd_cam = np.r_[qd[:3], yaw, pitch]

        if et > 0.5:
            qd *= 0.7 / et
            qd_cam *= 0.7 / et
        else:
            qd *= 1.4
            qd_cam *= 1.4

        if et < 0.03:
            return True, qd, qd_cam
        else:
            return False, qd, qd_cam

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
        
        pose_goal = self.commander.get_current_pose().pose

        pose_goal.position.x = pose[0]
        pose_goal.position.y = pose[1]
        pose_goal.position.z = pose[2]

        # scalar part should be first
        pose_goal.orientation.x = float(quaternion[0])
        pose_goal.orientation.y = float(quaternion[1])
        pose_goal.orientation.z = float(quaternion[2])
        pose_goal.orientation.w = float(quaternion[3])

        # pose_goal = geometry_msgs.msg.Pose()
        # pose_goal.orientation.w = 1.0
        # pose_goal.position.x = 0.4
        # pose_goal.position.y = 0.1
        # pose_goal.position.z = 0.4
        self.commander.set_pose_target(pose_goal)
        start_time = timeit.default_timer()
        plan = self.commander.plan()
        
        end_time = timeit.default_timer()

        self.planningTime = end_time - start_time        
        self.commander.execute(plan[1])
        self.plan = plan[1]
        return plan[0]


    def cleanup(self):
        self.scene.remove_world_object(self.generate_cylinder_name(0))
        self.check_object(cylinder_exists=False, name = self.generate_cylinder_name(0))

    def step_separate_arm(self, r, r_cam, Tep):

        if not self.plan_success:
            raise NotImplementedError("todo")
           
        step = self.plan.joint_trajectory.points[self.index]
        qd = step.velocities

        self.index += 1
        arrived = self.index == len(self.plan.joint_trajectory.points) - 1

        time_from_start = step.time_from_start.secs + step.time_from_start.nsecs/1000000000
        self.prev_timestep = time_from_start - self.curr_time
        self.curr_time = time_from_start

        return arrived, qd, [0,0]  

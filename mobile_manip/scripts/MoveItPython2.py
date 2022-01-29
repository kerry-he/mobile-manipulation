

import numpy as np, math
import qpsolvers as qp, sys
from scipy.spatial.transform import Rotation as R

import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import copy, timeit

CONSIDER_COLLISIONS = True


class MoveIt():

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

        Tep = np.eye(4)
        for i in range(16):
            Tep[i/4, i % 4] = sys.argv[i+1]
        
        self.plan_arm(Tep)

    def angle_axis_to_matrix(self, axis, theta):
        """
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        """
        axis = np.asarray(axis)
        axis = axis / math.sqrt(np.dot(axis, axis))
        a = math.cos(theta / 2.0)
        b, c, d = -axis * math.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


    def transform_between_vectors(self, a, b):
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)

        angle = np.arccos(np.dot(a, b))
        axis = np.cross(a, b)

        return self.angle_axis_to_matrix(axis, angle), angle, axis


    def step(self, r, r_cam, Tep, centroid_sight):
        if not self.separate_arrived:
            self.separate_arrived, qd, camera_qd = self.step_separate_base(r, r_cam, Tep)

            if self.seperate_arrived:
                self.plan_success = 

            return self.seperate_arrived, qd, camera_qd
        else:
            return self.step_separate_arm(r, r_cam, Tep)

    def getCameraPosition(self):
        raise NotImplementedError("todo")

    def plan_arm(self, Tep):


        wTc = self.getCameraPosition()
        
        for offset in np.linspace(0, 1, num=10):
            if CONSIDER_COLLISIONS:
                success = self.add_vision_ray(
                    camera_pose = wTc[:3, 3], 
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

        angle_axis = self.angle_axis_to_matrix(axis, angle)
        
        orientation = R.from_matrix(angle_axis).as_quat()
        
        p.pose.position.x = center[0] + center_offset[0]
        p.pose.position.y = center[1] + center_offset[1]
        p.pose.position.z = center[2] + center_offset[2]
        p.pose.orientation.x = orientation[0]
        p.pose.orientation.y = orientation[1]
        p.pose.orientation.z = orientation[2]
        p.pose.orientation.w = orientation[3]

        self.scene.add_cylinder(self.generate_cylinder_name(index), p, length - offset_distance, 0.05)
        return self.check_object(cylinder_exists=True, name = self.generate_cylinder_name(index))


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

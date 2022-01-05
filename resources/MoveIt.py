from base_environment import NUM_OBJECTS
import roboticstoolbox as rtb
import spatialmath as sm
import numpy as np
from roboticstoolbox.backends import swift
from scipy.spatial.transform import Rotation as R
import time

import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import copy, timeit
import spatialmath as sm
from baseController import BaseController


class kerry_moveit:

    def __init__(self):
        
        # We have to declare all Moveit related stuff before
        # Swift because it takes a while to get everything ready
        # Otherwise spawning objects might not work
        self.commander = moveit_commander.MoveGroupCommander('manipulator')
        self.robot = moveit_commander.RobotCommander()
        self.display_trajectory_publisher = rospy.Publisher(
            '/move_group/display_planned_path',
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20)

        self.cylinder_name = "camera_vision"
        self.scene = moveit_commander.PlanningSceneInterface()

        # Make and instance of the Swift simulator and open it
        # self.env = swift.Swift()
        # self.env.launch(realtime=True)
        # # Make a panda model and set its joint angles to the ready joint configuration
        # self.panda = rtb.models.Panda()
        # self.panda.q = self.panda.qr
        # # Add the robot to the simulator
        # self.env.add(self.panda)

        self.index = 0


    def init(self, collisions, camera_pose):
        for (index, i) in enumerate(collisions):
            self.add_vision_ray(
                camera_pose = i._base, 
                object_pose = camera_pose, 
                offset = 0,
                index = index)
        
        self.move_pose(collisions[-1]._base)
        self.curr_time = 0

    def step(self, panda, Tep, NUM_OBJECTS, n, collisions):
        
        step = self.plan.joint_trajectory.points[self.index]
        qd = step.velocities

        self.index += 1
        arrived = self.index == len(self.plan.joint_trajectory.points)

        occluded, _, _ = self.calcVelocityDamper(collisions, NUM_OBJECTS, n)

        time_from_start = step.time_from_start.secs + step.time_from_start.nsecs/1000000000
        self.prev_timestep = time_from_start - self.curr_time
        self.curr_time = time_from_start

        return qd, arrived, occluded

    def display_trajectory(self, traj):
        display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        display_trajectory.trajectory_start = self.robot.get_current_state()
        display_trajectory.trajectory.append(traj)
        self.display_trajectory_publisher.publish(display_trajectory)

    def move_cartesian(self):
        waypoints = []
        wpose = self.commander.get_current_pose().pose
        wpose.position.x = wpose.position.x + 0.1  # First move up (z)
        waypoints.append(copy.deepcopy(wpose))
        (plan, fraction) = self.commander.compute_cartesian_path(
                                            waypoints,   # waypoints to follow
                                            0.0001,        # eef_step
                                            0.0)         # jump_threshold
        plan = self.commander.retime_trajectory(self.commander.get_current_state(), plan, 1.0)
        self.display_trajectory(plan)
        self.commander.execute(plan)
        return plan

    def move_pose(self, pose):
        
        pose_goal = self.commander.get_current_pose().pose

        pose_goal.position.x = pose[0]
        pose_goal.position.y = pose[1]
        pose_goal.position.z = pose[2]

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
        return plan[1]

    def play_moveit_plan_swift(self, plan):
        self.panda.q = plan.joint_trajectory.points[0].positions
        self.env.step()
        curr_time = 0
        for step in plan.joint_trajectory.points:
            self.panda.qd = step.velocities
            self.panda.qdd = step.accelerations
            time_from_start = step.time_from_start.secs + step.time_from_start.nsecs/1000000000
            self.env.step(time_from_start - curr_time)
            curr_time = time_from_start

    def move_named_pose(self, name):
        self.commander.set_max_velocity_scaling_factor(0.5)
        self.commander.set_named_target(name)
        plan = self.commander.plan()
        self.commander.execute(plan[1])
        return plan[1]

    def add_vision_ray(self, camera_pose, object_pose, offset = 0.01, index=0):
        '''
        camera_pose: np.array [x, y, z]
        object_pose: np.array [x, y, z]
        '''

        p = geometry_msgs.msg.PoseStamped()
        p.header.frame_id = self.commander.get_planning_frame()

        center = (object_pose + camera_pose)/2
        length = np.linalg.norm(object_pose - camera_pose)
        cylinder_orientation_vector_z = (object_pose - camera_pose) / length
        center_offset = -1 * cylinder_orientation_vector_z * (offset/2)
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

        self.scene.add_cylinder(f"{self.cylinder_name}_{index}", p, length - offset, 0.04)
        return self.check_object(cylinder_exists=True)

    def check_object(self, cylinder_exists, timeout=4):
        cylinder_name = self.cylinder_name
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

    def remove_vision_ray(self):
        self.scene.remove_world_object(self.cylinder_name)
        return self.check_object(cylinder_exists=False)

    def cleanup(self):

        for i in range(NUM_OBJECTS):
            self.scene.remove_world_object(f"{self.cylinder_name}_{i}")
            self.check_object(cylinder_exists=False)

if __name__ == "__main__":
    rospy.init_node('kerry_MoveIt')
    exp = kerry_moveit()
    exp.add_vision_ray(
        camera_pose = np.array([0.4,-0.5,0.52]), 
        object_pose = np.array([0.4,0.5,0.52]), 
        offset = 0)
    input("worked?")
    plan = exp.move_pose()
    exp.play_moveit_plan_swift(plan)
    exp.remove_vision_ray()
    # exp.move_named_pose('ready')
    # plan = exp.move_pose()
    # exp.play_moveit_plan_swift(plan)
    # print(exp.panda.q)
    # input()
    # plan = exp.move_named_pose('ready')
    # exp.play_moveit_plan_swift(plan)
    # print(exp.panda.q)

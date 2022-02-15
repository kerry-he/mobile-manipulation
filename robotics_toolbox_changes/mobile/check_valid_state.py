import rospy
from moveit_msgs.srv import GetStateValidityRequest, GetStateValidity
from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState
import moveit_commander



class StateValidity():
    def __init__(self):
        # subscribe to joint joint states
        rospy.Subscriber("joint_states", JointState, self.jointStatesCB, queue_size=1)
        # prepare service for collision check
        self.sv_srv = rospy.ServiceProxy('/check_state_validity', GetStateValidity)
        # wait for service to become available
        self.sv_srv.wait_for_service()
        rospy.loginfo('service is avaiable')
        self.commander = moveit_commander.MoveGroupCommander('arm_with_torso')
        self.rs = self.commander.get_current_state()
        # input()
        # prepare msg to interface with moveit
        # self.rs = RobotState()
        # self.rs.joint_state.name = ['panda_joint1',
        #                             'panda_joint2',
        #                             'panda_joint3',
        #                             'panda_joint4',
        #                             'panda_joint5',
        #                             'panda_joint6',
        #                             'panda_joint7']
        # self.rs.joint_state.position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.joint_states_received = False


    def checkCollision(self):
        '''
        check if robotis in collision
        '''
        if self.getStateValidity().valid:
            rospy.loginfo('robot not in collision, all ok!')
        else:
            rospy.logwarn('robot in collision')


    def jointStatesCB(self, msg):
        '''
        update robot state
        '''
        # self.rs.joint_state.position = [msg.position[0], 
        #                                 msg.position[1],
        #                                 msg.position[2],
        #                                 msg.position[3],
        #                                 msg.position[4],
        #                                 msg.position[5],
        #                                 msg.position[6]]
        # self.joint_states_received = True
        self.rs = self.commander.get_current_state()
        self.checkCollision()


    def getStateValidity(self, group_name="manipulator", constraints=None):
        '''
        Given a RobotState and a group name and an optional Constraints
        return the validity of the State
        '''
        gsvr = GetStateValidityRequest()
        gsvr.robot_state = self.rs
        gsvr.group_name = group_name
        if constraints != None:
            gsvr.constraints = constraints
        result = self.sv_srv.call(gsvr)
        return result


    def start_collision_checker(self):
        while not self.joint_states_received:
            rospy.sleep(0.1)
        rospy.loginfo('joint states received! continue')
        self.checkCollision()
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('collision_checker_node', anonymous=False)
    collision_checker_node = StateValidity()
    collision_checker_node.start_collision_checker()
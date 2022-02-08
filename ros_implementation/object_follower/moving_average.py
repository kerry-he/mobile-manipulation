from filterpy.kalman import KalmanFilter
import numpy as np, glob
import matplotlib.pyplot as plt
from std_msgs.msg import Float64
import rospy, tf

class KalmanRos:
    def __init__(self):
        self.f = KalmanFilter(dim_x=4, dim_z=2)

        self.initial_state_set = False

        self.dt = 1/30.0

        self.vel_x = [0] * 5
        self.vel_y = [0] * 5


        rospy.Timer(rospy.Duration(1/30.0), self.main)

        self.tf_listener = tf.TransformListener()

        self.p1x = rospy.Publisher("kalman_pos_x", Float64, queue_size=1)
        self.p1y = rospy.Publisher("kalman_pos_y", Float64, queue_size=1)

        self.p2x = rospy.Publisher("measured_pos_x", Float64, queue_size=1)
        self.p2y = rospy.Publisher("measured_pos_y", Float64, queue_size=1)

        self.v1x = rospy.Publisher("kalman_vel_x", Float64, queue_size=1)
        self.v1y = rospy.Publisher("kalman_vel_y", Float64, queue_size=1)

        self.v2x = rospy.Publisher("measured_vel_x", Float64, queue_size=1)
        self.v2y = rospy.Publisher("measured_vel_y", Float64, queue_size=1)

        self.prev_pos = None

        rospy.spin()

    def main(self, useless):
        try:
            self.tf_listener.waitForTransform('camera_rgb_optical_frame', 'apple0', rospy.Time.now(), rospy.Duration(4))
            (trans, rot) = self.tf_listener.lookupTransform('camera_rgb_optical_frame', 'apple0', rospy.Time(0))
        except:
            return
        if self.prev_pos is None:
            self.prev_pos = [trans[0], trans[1]]
            return

        current_vel_x = (self.prev_pos[0] - trans[0]) / self.dt
        current_vel_y = (self.prev_pos[1] - trans[1]) / self.dt

        self.vel_x = [current_vel_x] + self.vel_x[:-1]
        self.vel_y = [current_vel_y] + self.vel_y[:-1]
      
        self.v1x.publish(np.mean(self.vel_x))
        self.v1y.publish(np.mean(self.vel_y))

        self.v2x.publish((self.prev_pos[0] - trans[0]) / self.dt)
        self.v2y.publish((self.prev_pos[1] - trans[1]) / self.dt)

        self.prev_pos = [trans[0], trans[1]]

if __name__ == "__main__":
    rospy.init_node("kalman_ros")
    KalmanRos()
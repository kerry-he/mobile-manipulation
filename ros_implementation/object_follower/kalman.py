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
        # transition matrix
        self.f.F = np.array([[1.0,0, self.dt,0],
                        [0.,1.,0,self.dt],
                        [0,0,1,0],
                        [0,0,0,1]])

        # measurement function
        self.f.H = np.array([[1, 0, 0, 0], [0,1, 0, 0]])

        # covairance function
        self.f.P *= 0.0000001
        # low measurement noise
        self.f.R = 0.0000001


        from filterpy.common import Q_discrete_white_noise
        self.f.Q = Q_discrete_white_noise(dim=4, dt=0.1, var=0.13)
        
        self.prev_pos = None

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

        rospy.spin()

    def main(self, useless):
        try:
            self.tf_listener.waitForTransform('camera_rgb_optical_frame', 'apple0', rospy.Time.now(), rospy.Duration(4))
            (trans, rot) = self.tf_listener.lookupTransform('camera_rgb_optical_frame', 'apple0', rospy.Time(0))
        except:
            return

        if not self.initial_state_set:
            self.initial_state_set = True
            self.f.x = np.array([trans[0], trans[1], 0, 0])

        else:
            self.f.predict()
            self.f.update([trans[0], trans[1]])


            self.p1x.publish(self.f.x[0])
            self.p1y.publish(self.f.x[1])

            self.p2x.publish(trans[0])
            self.p2y.publish(trans[1])

            self.v1x.publish(self.f.x[2])
            self.v1y.publish(self.f.x[3])

            self.v2x.publish((self.prev_pos[0] - trans[0]) / self.dt)
            self.v2y.publish((self.prev_pos[1] - trans[1]) / self.dt)

        self.prev_pos = [trans[0], trans[1]]

if __name__ == "__main__":
    rospy.init_node("kalman_ros")
    KalmanRos()
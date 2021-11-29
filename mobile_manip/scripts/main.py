#!/usr/bin/env python
import rospy
from std_msgs.msg import String


class MobileManipController:
    def __init__(self):
        self.sub = rospy.Subscriber("chatter", String, self.callback)
        self.pub = rospy.Publisher("chatter", String, queue_size=10)


        self.pub.publish("Hi")    


    def callback(self, data): 
        rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)


if __name__ == '__main__':
    try:
        rospy.init_node("mobile_manip", anonymous=True)
        controller = MobileManipController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
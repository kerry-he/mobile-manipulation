#!/usr/bin/env python
import sys
import tf
import rospy
import timeit
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

#from __future__ import print_function


class image_converter:

    def __init__(self):
        self.image_pub = rospy.Publisher(
            "/image_converter/output_video", Image, queue_size=1)

        self.cameraMatrix = None
        self.cameraMatrixInv = None
        self.depth_image = None

        self.bridge = CvBridge()

        self.image_sub = rospy.Subscriber(
            "/camera/color/image_raw", Image, self.callback, queue_size=1)
        self.camera_info_sub = rospy.Subscriber(
            "/camera/color/camera_info", CameraInfo, self.cameraCallback, queue_size=1)

        self.appleHeight = 1.22
        self.x = None
        self.y = None

        self.br = tf.TransformBroadcaster()

    def cameraCallback(self, data):
        
        if self.cameraMatrix is None:
            self.cameraMatrix = np.array([[data.K[0], data.K[1], data.K[2]], [
                                         data.K[3], data.K[4], data.K[5]], [data.K[6], data.K[7], data.K[8]]])
            self.cameraMatrixInv = np.linalg.inv(self.cameraMatrix)

#   def depth_callback(self, data):
#       self.depth_image = self.bridge.imgmsg_to_cv2(data, "passthrough")

    def callback(self, data):
        # print("image")
        
        if self.cameraMatrixInv is None:
            return

        t1 = timeit.default_timer()
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        
        lower1 = np.array([0, 100, 100])
        upper1 = np.array([5, 255, 255])

        lower2 = np.array([170, 100, 100])
        upper2 = np.array([180, 255, 255])

        # colors = np.array([0, 0, 255])
        colors = (0, 0, 255)

        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        kernel = np.ones((9, 9), np.uint8)
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = mask1 + mask2
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # t3 = timeit.default_timer()
        

        contour = cv2.findContours(
            mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        if len(contour) > 0:
            c = max(contour, key=cv2.contourArea)
            (self.x, self.y), radius = cv2.minEnclosingCircle(c)

            # Moments
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # Specify min size of object to be detected
            if radius > 1:
                # print(colors)
                cv2.circle(cv_image, (int(self.x), int(self.y)),
                           int(radius), colors, 2)
                cv2.putText(cv_image, "OBJECT", (int(self.x - (radius+1)),
                            int(self.y - (radius+1))), cv2.FONT_HERSHEY_COMPLEX, 0.8, colors, 2)

        if self.x is None or self.y is None:
            print("No object")
            # cv2.imshow("Image window", cv_image)
            # cv2.waitKey(1)
            return

        t2 = timeit.default_timer()
        print(1/(t2-t1))
        # cv2.imshow("hsv window", hsv)
        # cv2.imshow("mask window", mask)

        # cv2.imshow("Image window", mask)
        # cv2.imshow("Image window", cv_image)
        # cv2.waitKey(1)

        world_x, world_y, world_z = np.matmul(
            self.cameraMatrixInv, np.array([self.x, self.y, 1])) * self.appleHeight

        self.br.sendTransform((world_x, world_y, world_z),
                              tf.transformations.quaternion_from_euler(
                                  0, 0, 0),
                              rospy.Time.now(),
                              "apple0",
                              "overhead_cam")

        # print(world_x, world_y, world_z)

        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        except CvBridgeError as e:
            print(e)


def main(args):
    rospy.init_node('image_converter', anonymous=True)
    ic = image_converter()
    print("Running")
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)

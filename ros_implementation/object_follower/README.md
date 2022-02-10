## Installation

This packages uses Python 3. If you use a ROS version built with Python 2, additional steps are necessary to run the node.

- You need to build both cv_bridge and geometry2 module of ROS with Python 3. I recommend using a workspace separate from other ROS packages. Clone the package to the workspace. You might need to adjust some of the following instructions depending on your Python installation.
  ```Shell
  git clone -b melodic https://github.com/ros/geometry2.git
  git clone -b melodic https://github.com/ros-perception/vision_opencv.git
  ```
- If you use catkin_make, compile with
  ```Shell
  catkin_make -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
  ```
- For catkin tools, use
  ```Shell
  catkin config -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
  catkin build
  ```

## Setting up The Code
  - Adjust 'find_object_images.py' (line 32) with the height of the camera above the table.
    - To check the results of this, look at the 'apple0' tf frame, and check alignment against the pointcloud.
  - Adjust 'find_object_images.py' (line 27 + 27) with the correct topic names for the image and camera info.
  - Check 'stationary_vision.py' (line 31) for the controller which is being run
    - Adjust as nessacary

## Running the experiments
1. Run 'python3 find_object_images.py'
2. Record a video, we recommend using video_recorder from [image_view](http://wiki.ros.org/image_view). Make sure to give correct topics and files name 
3. Run 'python3 stationary_vision.py'
4. Record the final output of metric from the terminal running stationary_vision. 
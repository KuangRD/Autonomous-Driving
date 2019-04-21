# Unscented Kalman Filter Project Starter Code
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
---

[Project Repository](https://github.com/udacity/CarND-Unscented-Kalman-Filter-Project)

[Project Rubrics](https://review.udacity.com/#!/rubrics/783/view)

Self-Driving Car Engineer Nanodegree Program

In this project utilize an Unscented Kalman Filter to estimate the state of a moving object of interest with noisy lidar and radar measurements. Passing the project requires obtaining RMSE values that are lower that the tolerance outlined in the project reburic. 

This project involves the Term 2 Simulator which can be downloaded [here](https://github.com/udacity/self-driving-car-sim/releases)

This repository includes two files that can be used to set up and intall [uWebSocketIO](https://github.com/uWebSockets/uWebSockets) for either Linux or Mac systems. For windows you can use either Docker, VMware, or even [Windows 10 Bash on Ubuntu](https://www.howtogeek.com/249966/how-to-install-and-use-the-linux-bash-shell-on-windows-10/) to install uWebSocketIO. Please see [this concept in the classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/16cf4a78-4fc7-49e1-8621-3450ca938b77) for the required version and installation scripts.

Once the install for uWebSocketIO is complete, the main program can be built and ran by doing the following from the project top directory.

1. mkdir build
2. cd build
3. cmake ..
4. make
5. ./UnscentedKF

Note that the programs that need to be written to accomplish the project are src/ukf.cpp, src/ukf.h, tools.cpp, and tools.h

The program main.cpp has already been filled out, but feel free to modify it.

Here is the main protcol that main.cpp uses for uWebSocketIO in communicating with the simulator.


INPUT: values provided by the simulator to the c++ program

["sensor_measurement"] => the measurment that the simulator observed (either lidar or radar)


OUTPUT: values provided by the c++ program to the simulator

["estimate_x"] <= kalman filter estimated position x
["estimate_y"] <= kalman filter estimated position y
["rmse_x"]
["rmse_y"]
["rmse_vx"]
["rmse_vy"]

---

## Other Important Dependencies

* cmake >= v3.5
* make >= v4.1
* gcc/g++ >= v5.4

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./UnscentedKF` Previous versions use i/o from text files.  The current state uses i/o
from the simulator.

## Editor Settings

We've purposefully kept editor configuration files out of this repo in order to
keep it as simple and environment agnostic as possible. However, we recommend
using the following settings:

* indent using spaces
* set tab width to 2 spaces (keeps the matrices in source code aligned)

## Code Style

Please stick to [Google's C++ style guide](https://google.github.io/styleguide/cppguide.html) as much as possible.

## Generating Additional Data

This is optional!

If you'd like to generate your own radar and lidar data, see the
[utilities repo](https://github.com/udacity/CarND-Mercedes-SF-Utilities) for
Matlab scripts that can generate additional data.

## Project Instructions and Rubric

This information is only accessible by people who are already enrolled in Term 2
of CarND. If you are enrolled, see [the project page](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/c3eb3583-17b2-4d83-abf7-d852ae1b9fff/concepts/f437b8b0-f2d8-43b0-9662-72ac4e4029c1)
for instructions and the project rubric.

## Tips

Check out the coding quizzes and coding quiz answers from the lesson
* Use the coding quizzes from the lecture to help guide you. You have already implemented the prediction step and radar update step for the unscented Kalman filter. In the project, you will also need to code the update step for lidar.
### Normalize Angles
* Don't forget to normalize angles so that angles are between -\pi−π and \piπ. The lectures explained how to do this.
### Don't Forget to Tune Parameters and Initialize Variables
* In the starter code, we have given values for the process noise and measurement noise. You will need to tune the process noise parameters std_a_ and std_yawdd_ in order to get your solution working on both datasets. The measurement noise parameters for lidar and radar should be left as given.
* You will also need to initialize your state vector x and state covariance matrix P with appropriate values.
* If you are having trouble finding appropriate values for your parameters, consider analyzing the data file first. Between time intervals, how much does the object tend to accelerate? What is the maximum acceleration? What is the standard deviation of the acceleration? You can calculate approximate accelerations by dividing changes in velocity by the change in time.
### Check for Divide By Zero
* Check for divides by zero.
### Debug
* If you implement your solution based on the code taught in the unscented Kalman filter lesson and also find appropriate parameters, you can reach the required RMSE values in the rubric! If you find your code hangs, try adding print statements to figure out why. Your code might be correct but you might need to do more parameter tuning or adjust your initialization values.
###  Ideas for Standing out
* Use NIS to help tune your parameters
* Visualize the ground truth, sensor measurements, and your Kalman filter results
* Compare your UKF and EKF project results. Both projects use the same data file. RMSE, especially for v_x and v_y should be lower for the UKF project than the EKF project. Why might that be?

## Bonus Challenge
[Run Away Robot with Unscented Kalman Filter Bonus Challenge Starter Code](https://github.com/udacity/CarND-Catch-Run-Away-Car-UKF)
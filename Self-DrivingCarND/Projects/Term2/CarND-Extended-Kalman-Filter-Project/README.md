# Extended Kalman Filter Project Starter Code
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
---

[Project Repository](https://github.com/udacity/CarND-Extended-Kalman-Filter-Project)

[Project Rubrics](https://review.udacity.com/#!/rubrics/748/view)


[//]: # (Image References)

[MathMDCodingBAK]: ./pic/MathMDCodingBAK.PNG "Model Visualization"


In this project you will utilize a kalman filter to estimate the state of a moving object of interest with noisy lidar and radar measurements. Passing the project requires obtaining RMSE values that are lower that the tolerance outlined in the project rubric. 

This project involves the Term 2 Simulator which can be downloaded [here](https://github.com/udacity/self-driving-car-sim/releases)

This repository includes two files that can be used to set up and install [uWebSocketIO](https://github.com/uWebSockets/uWebSockets) for either Linux or Mac systems. For windows you can use either Docker, VMware, or even [Windows 10 Bash on Ubuntu](https://www.howtogeek.com/249966/how-to-install-and-use-the-linux-bash-shell-on-windows-10/) to install uWebSocketIO. Please see [this concept in the classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/16cf4a78-4fc7-49e1-8621-3450ca938b77) for the required version and installation scripts.

Once the install for uWebSocketIO is complete, the main program can be built and run by doing the following from the project top directory.

1. mkdir build
2. cd build
3. cmake ..
4. make
5. ./ExtendedKF

Note that the programs that need to be written to accomplish the project are src/FusionEKF.cpp, src/FusionEKF.h, kalman_filter.cpp, kalman_filter.h, tools.cpp, and tools.h

The program main.cpp has already been filled out, but feel free to modify it.

Here is the main protcol that main.cpp uses for uWebSocketIO in communicating with the simulator.


INPUT: values provided by the simulator to the c++ program

["sensor_measurement"] => the measurement that the simulator observed (either lidar or radar)


OUTPUT: values provided by the c++ program to the simulator

["estimate_x"] <= kalman filter estimated position x
["estimate_y"] <= kalman filter estimated position y
["rmse_x"]
["rmse_y"]
["rmse_vx"]
["rmse_vy"]

---

## Other Important Dependencies

* cmake >= 3.5
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make` 
   * On windows, you may need to run: `cmake .. -G "Unix Makefiles" && make`
4. Run it: `./ExtendedKF `

## Editor Settings

We've purposefully kept editor configuration files out of this repo in order to
keep it as simple and environment agnostic as possible. However, we recommend
using the following settings:

* indent using spaces
* set tab width to 2 spaces (keeps the matrices in source code aligned)

## Code Style

Please (do your best to) stick to [Google's C++ style guide](https://google.github.io/styleguide/cppguide.html).

## Generating Additional Data

This is optional!

If you'd like to generate your own radar and lidar data, see the
[utilities repo](https://github.com/udacity/CarND-Mercedes-SF-Utilities) for
Matlab scripts that can generate additional data.

## Project Instructions and Rubric

Note: regardless of the changes you make, your project must be buildable using
cmake and make!

More information is only accessible by people who are already enrolled in Term 2
of CarND. If you are enrolled, see [the project resources page](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/382ebfd6-1d55-4487-84a5-b6a5a4ba1e47)
for instructions and the project rubric.

## Hints!

* You don't have to follow this directory structure, but if you do, your work
  will span all of the .cpp files here. Keep an eye out for TODOs.

## Call for IDE Profiles Pull Requests

Help your fellow students!

We decided to create Makefiles with cmake to keep this project as platform
agnostic as possible. Similarly, we omitted IDE profiles in order to we ensure
that students don't feel pressured to use one IDE or another.

However! We'd love to help people get up and running with their IDEs of choice.
If you've created a profile for an IDE that you think other students would
appreciate, we'd love to have you add the requisite profile files and
instructions to ide_profiles/. For example if you wanted to add a VS Code
profile, you'd add:

* /ide_profiles/vscode/.vscode
* /ide_profiles/vscode/README.md

The README should explain what the profile does, how to take advantage of it,
and how to install it.

Regardless of the IDE used, every submitted project must
still be compilable with cmake and make.

## Tips & Tricks

### Summary of What Needs to Be Done
1. In `tools.cpp`, fill in the functions that calculate root mean squared error (RMSE) and the Jacobian matrix.
2. Fill in the code in `FusionEKF.cpp`. You'll need to initialize the Kalman Filter, prepare the Q and F matrices for the prediction step, and call the radar and lidar update functions.
3. In `kalman_filter.cpp`, fill out the `Predict()`, `Update()`, and `UpdateEKF()` functions.
### Tips and Tricks
#### Review the Previous Lessons
* Review the previous lessons! Andrei, Dominik and co. have given you everything you need. In fact, you've built most of an Extended Kalman Filter already! Take a look at the programming assignments and apply the techniques you used to this project.
#### No Need to Tune Parameters
* The R matrix values and Q noise values are provided for you. There is no need to tune these parameters for this project.

![alt text](MathMDCodingBAK)

#### Initializing the State Vector
You'll need to initialize the state vector with the first sensor measurement.
Although radar gives velocity data in the form of the range rate ![](http://latex.codecogs.com/gif.latex?\\dot{\rho})  
, a radar measurement does not contain enough information to determine the state variable velocities v_xv 
x
​	  and v_yv 
y
​	 . You can, however, use the radar measurements \rhoρ and \phiϕ to initialize the state variable locations p_xp 
x
​	  and p_yp 
y
​	 .
#### Calculating y = z - H * x'
For lidar measurements, the error equation is `y = z - H * x'`. For radar measurements, the functions that map the x vector [px, py, vx, vy] to polar coordinates are non-linear. Instead of using H to calculate `y = z - H * x'`, for radar measurements you'll have to use the equations that map from cartesian to polar coordinates: `y = z - h(x')`.
#### Normalizing Angles
* In C++, `atan2()` returns values between -pi and pi. When calculating phi in `y = z - h(x)` for radar measurements, the resulting angle phi in the y vector should be adjusted so that it is between -pi and pi. The Kalman filter is expecting small angle values between the range -pi and pi. HINT: when working in radians, you can add 2\pi2π or subtract 2\pi2π until the angle is within the desired range.
#### Avoid Divide by Zero throughout the Implementation
* Before and while calculating the Jacobian matrix Hj, make sure your code avoids dividing by zero. For example, both the x and y values might be zero or px * px + py * py might be close to zero. What should be done in those cases?
#### Test Your Implementation
* Test! We're giving you the ability to analyze your output data and calculate RMSE. As you make changes, keep testing your algorithm! If you are getting stuck, add print statements to pinpoint any issues. But please remove extra print statements before turning in the code.
### Ideas for Standing out!
The Kalman Filter general processing flow that you've learned in the preceding lessons gives you the basic knowledge needed to track an object. However, there are ways that you can make your algorithm more efficient!

* Dealing with the first frame, in particular, offers opportunities for improvement.
* Experiment and see how low your RMSE can go!
* Try removing radar or lidar data from the filter. Observe how your estimations change when running against a single sensor type! Do the results make sense given what you know about the nature of radar and lidar data?
* We give you starter code, but you are not required to use it! You may want to start from scratch if: You want a bigger challenge! You want to redesign the project architecture. There are many valid design patterns for approaching the Kalman Filter algorithm. Feel free to experiment and try your own! You want to use a different coding style, eg. functional programming. While C++ code naturally tends towards being object-oriented in nature, it's perfectly reasonable to attempt a functional approach. Give it a shot and maybe you can improve its efficiency!

## Optional Resources
To complete the project, you only need the files in the github repo; however, we are also providing some extra resources that you can use to develop your solution:

* A [Sensor Fusion utilities repo](https://github.com/udacity/CarND-Mercedes-SF-Utilities) containing Matlab scripts that will generate more sample data (generating your own sample data is completely optional)
* A visualization package that you can also find within [the Sensor Fusion utilities repo](https://github.com/udacity/CarND-Mercedes-SF-Utilities)
# Sensors used for autonomous car

The example in the class is the Chewbacca of Mercedes Benz.

## Camera

![alt test][image1]
**Stereo Camera** gather both image data, as well as distance information.

**Traffic Signal Camera** equipped with special lens to give the camera sufficient range to detect the signal from far away.

## RADAR Strength and Weaknesses

RADAR: **RA**dio **D**etection **A**nd **R**anging

RADAR was a mature technology for vehicle, in adaptive cruise control, blind spot warning, collision warning and collision avoidance.

### Strength
* RADAR measure velocity directly based on Doppler Effect, while other sensors measuring velocity by calculating the difference between two readings.

* RADAR generating RADAR maps of the environment for localization.

* Radar waves bounces off hard surfaces, so they can provide measurements to objects without direct line of sight. RADAR can see underneath of other vehicles and spot buildings and objects that would be obscured otherwise.

* RADAR is the least affected by rain or fog of all the sensors on the car, and can have a wide field of view(about 150 degree) and a long range(200+ meters).

### Weaknesses
* Lower resolution than LIDAR and Cameras, especially in vertical direction. What's more, lower resolution also means that reflections from static objects can cause problem. **Radar Clutter**(Things like manhole cover and cans, can have high radar reflectivity even though they are relatively small.). Current automotive radars usually disregard static objects.

## LIDAR Strength and Weaknesses
LIDAR: **LI** **D**etection **A**nd **R**anging

LIDAR use an infrared laser beam(红外激光束). Most current LIDAR use light in the 900 nanometer wavelength range, some LADARs use longer wavelengths, which perform better in rain and fog.

>Footnote on Lidar

>There are other possibilities to scan the laser beams. Instead of rotating the lasers or having a rotating mirror, we can scan the lidar with a vibrating micromirror. Those lidars are in development but none are commercially available now (as of March 2017).

>Instead of mechanically moving the laser beam, a similar principle to phased array radar can be employed. Dividing a single laser beam into multiple waveguides, the phase relationship between the waveguides can be altered and thereby the direction of the laser beam shifted. A company named Quanergy is working on systems like that. The advantage is that the form factor can be much smaller and that there are no moving parts.

>Another possibility is to use the laser as a gigantic flash like with a camera and then measuring the arrival times for all the objects with one big imaging photodiode array. This is in effect a 3D camera. The components are currently very expensive and currently this is used more in space and in terrain mapping applications.)
![alt test][image2]



## Summary

![alt test][image3]
![alt test][image4]

[//]: # (Image References)

[image1]: ./pic/cameras.png
[image2]: ./pic/LIDAR_visual.png
[image3]: ./pic/comparison.png
[image4]: ./pic/properties.png

# Hardware Setup Notes

This file contains basic resources and instructions for connecting to the Kinova Gen3 robot in real life. 

## Related Resources:

- [Quick-start guide](https://www.kinovarobotics.com/sites/default/files/KINOVA_Gen3_Ultra_lightweight_robot-%20Quick_Start_Guide.pdf) (physical version also availible in packing case)
- [User Manual](https://www.kinovarobotics.com/sites/default/files/UG-014_KINOVA_Gen3_Ultra_lightweight_robot-User_guide_EN_R01.pdf)
- [Github](https://github.com/Kinovarobotics/kortex)

## Drake Interface

Coming soon.

## Basics

### Powerup

- Clamp base firmly to mounting surface
- Connect emergency stop
- Connect power supply
- Check that emergency stop is disabled (twist up)
- Hold silver power button for 3 seconds
- Wait for top light to be solid green

### Power Down

- Hold power button for several seconds
- Make sure robot is in stable position
- Disconnect power supply
- Power off joystick (hold central X-box button 6 seconds)

### Moving manually
- Hold left button (slightly indented) to move robot around freely (gravity compensation mode)
- Hold right button (sticking out slightly) to move robot around but keep end-effector orientation.

### Joystick Control

- Connect X-box controller via USB
- See [Quick-start guide](https://www.kinovarobotics.com/sites/default/files/KINOVA_Gen3_Ultra_lightweight_robot-%20Quick_Start_Guide.pdf) for detailed controls
- Particularly useful: (A) and (B) move to different "home" positions

### Webapp Control

- Connect via ethernet
- Disable wifi 
- Set computer IP to 192.168.1.11
- In browser, go to 192.168.1.10 (robot's IP)
- Log in with admin/admin

### Gripper

#### Robotiq 2F-85

- Once attached, make sure correct gripper is selected in Webapp: Configurations/Arm/Product/End Effector Type
- L/R triggers on joystick open and close the gripper

#### Robotiq Hand-e

Coming soon.

### Python API

- Install instructions can be found [here](https://github.com/Kinovarobotics/kortex/blob/master/api_python/examples/readme.md#install-kortex-python-api-and-required-dependencies)
- Connect via ethernet
- Disable wifi 
- Set computer IP to 192.168.1.11
- See [examples](https://github.com/Kinovarobotics/kortex/tree/master/api_python/examples)

### C++ API

Coming soon. 


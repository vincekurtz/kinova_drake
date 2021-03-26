# Kinvoa Drake

Code for controlling a Kinova Gen3 Manipulator via Drake.

## Dependencies

Simulation and hardware:

- OpenCV
- [Drake](https://drake.mit.edu/)
- [Meshcat](https://github.com/rdeits/meshcat) visualizer
- [Open3d](http://www.open3d.org/docs/latest/introduction.html) point cloud library
- matplotlib
- numpy
- python3
- scipy

Hardware only:

- GStreamer (for accessing camera streams)
- OpenCV with GStreamer bindings
- [Kinova kortex API](https://github.com/Kinovarobotics/kortex)

## Running examples

There are basic examples of various things (both simulation and hardware) in the `examples` directory. These must be run
from this directory with, e.g., `python3 -m examples.peg_pickup_demo`.

### Simple Example

The simplest examples can be run with
```
python3 -m examples.simulation_demo
```
(in simulation) and
```
python3 -m examples.hardware_demo
```
(on the hardware).

These examples demonstrate how to send a simple end-effector position, velocity, or wrench
references. Various parameters can be set in the respective files (`examples/simulation_demo.py`
or `examples/hardware_demo.py`) such as gripper type, reference type, whether to make system
diagram plots, etc. 

### Peg Pickup Example

The peg pickup examples found in `examples/peg_pickup_demo.py` (simulation) and `examples/peg_pickup_demo_hw.py`
(hardware) show how to use a `CommandSequence` object to design and track a series of setpoints.

Specifically, these setpoints are designed so that the robot reaches and picks up a peg in a known location. 

### Point Cloud Example

Examples of using the robot's depth camera to pick up an object at an a-priori unknown location
can be found in `examples/point_cloud_demo.py` (simulation) and `examples/point_cloud_demo_hw.py` 
(hardware). 

We use a genetic algorithm and a simple geometric heuristic (described [here](http://manipulation.csail.mit.edu/clutter.html))
to select a grasp location. 



#! .venv/bin/python

from pyrosim import World, PyroSim
import time


def robot(pyrosim: PyroSim):
    (
        pyrosim.start_urdf("robot.urdf")
        .add_cube("torso", [0, 0, 1.5], [1, 1, 1])
        .add_cube("rightleg", [0.5, 0, -0.5], [1, 1, 1])
        .add_cube("leftleg", [-0.5, 0, -0.5], [1, 1, 1])
        .add_joint("torso_rightleg", "torso", "rightleg", "revolute", [0.5, 0, 1])
        .add_joint("torso_leftleg", "torso", "leftleg", "revolute", [-0.5, 0, 1])
        .end()
    )


def brain(pyrosim: PyroSim):
    (
        pyrosim.start_nndf("brain.nndf")
        .add_sensor_neuron(0, "torso")
        .add_sensor_neuron(1, "rightleg")
        .add_sensor_neuron(2, "leftleg")
        .add_motor_neuron(3, "torso_rightleg")
        .add_motor_neuron(4, "torso_leftleg")
        .add_synapse(1, 3, -0.5)
        .add_synapse(2, 3, 0.5)
        .add_synapse(1, 4, -0.5)
        .add_synapse(2, 4, 0.5)
        .end()
    )


with World(samples=1000, robots={"robot": (robot, brain)}) as world:
    for i in range(1000):
        for robot_instance in world.robots.values():
            robot_instance.sense(i)
            robot_instance.think(i)
            robot_instance.act(i)
        world.step()
        time.sleep(0.01)

#!/usr/bin/env python3
"""Example MuJoCo simulation script."""

from mujoco_sim import MuJoCoWorld, MuJoCoSim
import time


def robot(sim: MuJoCoSim):
    """Create a simple bipedal robot in MJCF format."""
    (
        sim.start_mjcf("robot.mjcf")
        .add_body("torso", [0, 0, 1.5], [0.5, 0.5, 0.5])
        .add_body(
            "rightleg",
            [0.3, 0, -0.5],
            [0.2, 0.2, 0.5],
            parent="torso",
            joint_name="torso_rightleg",
            joint_type="hinge",
            joint_axis=[0, 1, 0],
        )
        .end_body()
        .add_body(
            "leftleg",
            [-0.3, 0, -0.5],
            [0.2, 0.2, 0.5],
            parent="torso",
            joint_name="torso_leftleg",
            joint_type="hinge",
            joint_axis=[0, 1, 0],
        )
        .end_body()
        .end_body()  # Close torso
        .end()
    )


def brain(sim: MuJoCoSim):
    """Create a simple neural network brain."""
    (
        sim.start_nndf("brain.nndf")
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


if __name__ == "__main__":
    with MuJoCoWorld(samples=1000, robots={"robot": (robot, brain)}) as world:
        for i in range(1000):
            for robot_instance in world.robots.values():
                robot_instance.sense(i)
                robot_instance.think(i)
                robot_instance.act(i)
            world.step()
            time.sleep(0.01)

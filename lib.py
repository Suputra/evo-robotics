import pyrosim.pyrosim as pyrosim
from dataclasses import dataclass
import numpy as np

import pybullet_data
import pybullet as p


class World:
    def __init__(self, samples=100, urdfs={}, sdfs=[]):
        physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setGravity(0, 0, -50)

        self.samples = samples
        self.robots = {}
        # make urdfs and sdfs
        for name, urdf in urdfs.items():
            self.robots[name] = Robot(name, urdf, samples)

        for sdf in sdfs:
            sdf()
            p.loadSDF(sdf.__name__ + ".sdf")

        p.loadURDF("plane.urdf")
        print(self.robots)

    def __enter__(self):
        return self

    def step(self):
        p.stepSimulation()

    def __exit__(self, type, value, traceback):
        for name, robot in self.robots.items():
            robot.save_data()
        p.disconnect()


class Robot:
    def __init__(self, name, urdf_fn, samples):
        self.name = name
        self.senses = {}
        self.samples = samples

        # Generate and load the URDF
        urdf_fn()
        self.id = p.loadURDF(urdf_fn.__name__ + ".urdf")

    def sense(self, link_name, index, sensor_type="touch"):
        """Get and store sensor reading in one concise call"""
        if sensor_type == "touch":
            data = pyrosim.Get_Touch_Sensor_Value_For_Link(link_name, bodyID=self.id)
        # Add other sensor types as needed

        # Create key for this sensor in the format "link_type"
        sensor_key = f"{link_name}_{sensor_type}"

        if sensor_key not in self.senses:
            # Auto-create sense array based on data shape
            self.senses[sensor_key] = np.zeros(self.samples)

        self.senses[sensor_key][index] = data
        return data

    def motor(
        self,
        joint_name,
        target_position,
        control_mode=p.POSITION_CONTROL,
        max_force=500,
    ):
        """Set motor for joint in a concise call"""
        pyrosim.Set_Motor_For_Joint(
            bodyIndex=self.id,
            jointName=joint_name,
            controlMode=control_mode,
            targetPosition=target_position,
            maxForce=max_force,
        )

    def save_data(self):
        """Save all sensor data to files"""
        for sense_name, data in self.senses.items():
            np.save(f"data/{self.name}_{sense_name}.npy", data)


## urdfs
## some notes:
# base link and joints all are absolute
# following links are based on parent joints

# urdf functions will must have the same name as the file they write to


def world():
    pyrosim.Start_SDF("world.sdf")
    pyrosim.Send_Cube(name="Box", pos=[-2, 2, 0.5], size=[1, 1, 1])
    pyrosim.End()


def robot():
    pyrosim.Start_URDF("robot.urdf")
    pyrosim.Send_Cube(name="torso", pos=[0, 0, 1.5], size=[1, 1, 1])
    pyrosim.Send_Cube(name="rightleg", pos=[0.5, 0, -0.5], size=[1, 1, 1])
    pyrosim.Send_Cube(name="leftleg", pos=[-0.5, 0, -0.5], size=[1, 1, 1])
    pyrosim.Send_Joint(
        name="torso_rightleg",
        parent="torso",
        child="rightleg",
        type="revolute",
        position=[0.5, 0, 1],
    )
    pyrosim.Send_Joint(
        name="torso_leftleg",
        parent="torso",
        child="leftleg",
        type="revolute",
        position=[-0.5, 0, 1],
    )
    pyrosim.End()

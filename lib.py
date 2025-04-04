import pyrosim.pyrosim as pyrosim
from dataclasses import dataclass
import numpy as np

import pybullet_data
import pybullet as p


class World:
    def __init__(self, samples=100, urdfs={}, sdfs=[]):
        p.connect(p.GUI)
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
        self.policy = {}
        self.samples = samples

        # Generate and load the URDF
        urdf_fn()
        self.id = p.loadURDF(urdf_fn.__name__ + ".urdf")

        # Get all link names for this robot
        self.links = []
        self.joints = []
        self.joint_indices = {}
        for i in range(p.getNumJoints(self.id)):
            info = p.getJointInfo(self.id, i)
            # Joint info[12] contains the link name
            link_name = info[12].decode("utf-8")
            self.links.append(link_name)
            joint_name = info[1].decode("utf-8")
            self.joints.append(joint_name)
            self.joint_indices[joint_name] = i
        # Add the base link (which is often not part of joints)
        self.links.append(p.getBodyInfo(self.id)[0].decode("utf-8"))

        for link in self.links:
            self.senses[link + "_touch"] = np.zeros(self.samples)

        for joint in self.joints:
            self.policy[joint] = np.zeros(self.samples)

    def sense(self, index):
        """Get and store sensor reading in one concise call"""
        for link in self.links:
            data = pyrosim.Get_Touch_Sensor_Value_For_Link(link, bodyID=self.id)

            self.senses[link + "_touch"][index] = data

    def act(self, index, control_mode=p.POSITION_CONTROL, max_force=500):
        for joint in self.joints:
            pyrosim.Set_Motor_For_Joint(
                bodyIndex=self.id,
                jointName=joint,
                controlMode=control_mode,
                targetPosition=self.policy[joint][index],
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

# urdf functions must have the same name as the file they write to


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

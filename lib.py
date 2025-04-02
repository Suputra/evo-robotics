import pyrosim.pyrosim as pyrosim
from dataclasses import dataclass
import numpy as np

import pybullet_data
import pybullet as p


class World:
    def __init__(self, samples=500, urdfs={}, sdfs=[]):
        physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setGravity(0, 0, -50)

        self.samples = samples
        self.ids = {}
        self.robots = {}
        # make urdfs and sdfs
        for name, urdf in urdfs.items():
            urdf()
            id = p.loadURDF(urdf.__name__ + ".urdf")
            # so that we can have multiple robots with the same form
            self.ids[name] = id
            self.robots[name] = {}

        for sdf in sdfs:
            sdf()
            p.loadSDF(sdf.__name__ + ".sdf")

        self.robots["plane"] = p.loadURDF("plane.urdf")

    def __enter__(self):
        return self

    def step(self):
        p.stepSimulation()

    def __exit__(self, type, value, traceback):
        for robotname, senses in self.robots.items():
            for sense, data in senses.items():
                np.save(f"data/{robotname}_{sense}.npy", data)
        p.disconnect()


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

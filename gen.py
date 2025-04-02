#! /Users/saahas/morpho-rl/.venv/bin/python
import pyrosim.pyrosim as pyrosim


def world():
    pyrosim.Start_SDF("world.sdf")
    pyrosim.Send_Cube(name="Box", pos=[-2, 2, 0.5], size=[1, 1, 1])
    pyrosim.End()


def robot():
    pyrosim.Start_URDF("body.urdf")
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


world()
robot()
## Notes:
# base link and joints all are absolute
# following links are based on parent joints

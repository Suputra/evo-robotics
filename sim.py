#! /Users/saahas/morpho-rl/.venv/bin/python
import math
import random
import numpy as np
import time

import pybullet as p
import pyrosim.pyrosim as pyrosim

from lib import World, world, robot


SAMPLES = 500
with World(urdfs={"robot1": robot}, sdfs=[world]) as w:
    w.robots["rightleg"] = np.zeros(w.samples)

    for i in range(w.samples):
        w.robots["rightleg"][i] = pyrosim.Get_Touch_Sensor_Value_For_Link("rightleg")
        time.sleep(0.01)
        pyrosim.Set_Motor_For_Joint(
            bodyIndex=robotId,
            jointName="torso_rightleg",
            controlMode=p.POSITION_CONTROL,
            targetPosition=random.random() * math.pi / 2,
            maxForce=500,
        )
        pyrosim.Set_Motor_For_Joint(
            bodyIndex=robotId,
            jointName="torso_leftleg",
            controlMode=p.POSITION_CONTROL,
            targetPosition=random.random() * math.pi / 2,
            maxForce=500,
        )

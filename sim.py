#! /Users/saahas/morpho-rl/.venv/bin/python
import math
import random
import pybullet as p
import numpy as np
import pyrosim.pyrosim as pyrosim
import pybullet_data
import time

SAMPLES = 500
physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.setGravity(0, 0, -50)
planeId = p.loadURDF("plane.urdf")
robotId = p.loadURDF("body.urdf")
pyrosim.Prepare_To_Simulate(robotId)
p.loadSDF("world.sdf")
rightleg_data = np.zeros(SAMPLES)
for i in range(SAMPLES):
    p.stepSimulation()
    rightleg_data[i] = pyrosim.Get_Touch_Sensor_Value_For_Link("rightleg")
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
print(rightleg_data)
np.save("data/rightleg.npy", rightleg_data)
p.disconnect()

#! /Users/saahas/morpho-rl/.venv/bin/python
import pybullet as p
import numpy as np
import pyrosim.pyrosim as pyrosim

import pybullet_data
import time

SAMPLES = 100

physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)

p.setGravity(0,0,-50)
planeId = p.loadURDF("plane.urdf")
robotId = p.loadURDF("body.urdf")
pyrosim.Prepare_To_Simulate(robotId)


p.loadSDF("world.sdf")

rightleg_data = np.zeros(SAMPLES)

for i in range(SAMPLES):
    p.stepSimulation()
    rightleg_data[i] = pyrosim.Get_Touch_Sensor_Value_For_Link("rightleg")
    time.sleep(0.01)

print(rightleg_data)

np.save("data/righleg.npy", rightleg_data)

p.disconnect()
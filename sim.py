#! /Users/saahas/morpho-rl/.venv/bin/python
import math
import random
import numpy as np
import time

import pybullet as p
import pyrosim.pyrosim as pyrosim

from lib import World, world, robot

ROBOTS = {
    "robot1": robot,
}

OBJECTS = {
    world,
}

SAMPLES = 500

with World(samples=SAMPLES, urdfs=ROBOTS, sdfs=OBJECTS) as w:
    # how will the robot act (dumb - no sensor use)
    for joint in w.robots["robot1"].joints:
        w.robots["robot1"].policy[joint] = np.random.rand(w.samples)

    # main loop
    for i in range(w.samples):
        w.robots["robot1"].sense(i)
        w.robots["robot1"].act(i)
        time.sleep(0.01)
        w.step()

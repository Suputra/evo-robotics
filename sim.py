#! /Users/saahas/morpho-rl/.venv/bin/python
import math
import random
import numpy as np
import time

import pybullet as p
import pyrosim.pyrosim as pyrosim

from lib import World, world, robot

robots = {
    "robot1": robot,
}

objects = {
    world,
}

SAMPLES = 500
with World(urdfs=robots, sdfs=objects) as w:
    for i in range(w.samples):
        w.robots["robot1"].sense("rightleg", i)
        time.sleep(0.01)
        w.robots["robot1"].motor("torso_rightleg", random.random() * math.pi / 2)
        w.robots["robot1"].motor("torso_leftleg", random.random() * math.pi / 2)
        w.step()

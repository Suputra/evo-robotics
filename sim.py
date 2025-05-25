#! /Users/saahas/morpho-rl/.venv/bin/python
import numpy as np
import time

from lib import World, box, robot, brain

ROBOTS = {"robot1": (robot, brain)}

OBJECTS = {
    box,
}

SAMPLES = 500

with World(samples=SAMPLES, urdfs=ROBOTS, sdfs=OBJECTS) as w:
    # main loop
    for i in range(w.samples):
        w.robots["robot1"].sense(i)
        w.robots["robot1"].act(i)
        w.robots["robot1"].think(i)
        time.sleep(0.01)
        w.step()

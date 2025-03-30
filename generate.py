import pyrosim.pyrosim as pyrosim

pyrosim.Start_SDF("box.sdf")

height = 0
for x in range(-2, 3):
    for y in range(-2, 3):
        for i in range(10):
            size = 0.9 ** i 
            pyrosim.Send_Cube(name="Box", pos=[x,y,height + size/2] , size=[size, size, size])
            height += size

pyrosim.End()


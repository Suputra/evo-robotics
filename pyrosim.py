import numpy as np
import pybullet as p
import pybullet_data
import random
import math
import os
from typing import Dict, List, Tuple, Optional, Union


class PyroSim:
    """Consolidated PyBullet simulation helper with neural network support."""

    # Neuron types
    SENSOR_NEURON = 0
    MOTOR_NEURON = 1
    HIDDEN_NEURON = 2

    def __init__(self, output_folder: str = "data"):
        self.output_folder = output_folder
        self.reset()

    def reset(self):
        """Reset internal state for new file generation."""
        self._file = None
        self._filetype = None
        self._links = []
        self._link_index = -1
        self._link_to_index = {}
        self._joint_to_index = {}
        # Create output folder if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)

    # ==================== FILE GENERATION ====================

    def start_urdf(self, filename: str):
        """Start creating a URDF file."""
        self.reset()
        self._filetype = "urdf"
        filepath = os.path.join(self.output_folder, filename)
        self._file = open(filepath, "w")
        self._file.write('<?xml version="1.0"?>\n<robot name="robot">\n')
        return self

    def start_sdf(self, filename: str):
        """Start creating an SDF file."""
        self.reset()
        self._filetype = "sdf"
        filepath = os.path.join(self.output_folder, filename)
        self._file = open(filepath, "w")
        self._file.write('<?xml version="1.0"?>\n<sdf version="1.4">\n')
        return self

    def start_nndf(self, filename: str):
        """Start creating a neural network definition file."""
        self.reset()
        self._filetype = "nndf"
        filepath = os.path.join(self.output_folder, filename)
        self._file = open(filepath, "w")
        self._file.write("<neuralNetwork>\n")
        return self

    def add_cube(
        self, name: str, pos: List[float] = [0, 0, 0], size: List[float] = [1, 1, 1]
    ):
        """Add a cube to the current file."""
        if self._filetype == "sdf":
            self._add_sdf_cube(name, pos, size)
        elif self._filetype == "urdf":
            self._add_urdf_cube(name, pos, size)
        self._link_to_index[name] = self._link_index
        self._link_index += 1
        return self

    def add_joint(
        self,
        name: str,
        parent: str,
        child: str,
        joint_type: str = "revolute",
        position: List[float] = [0, 0, 0],
    ):
        """Add a joint to the URDF."""
        if self._filetype != "urdf":
            raise ValueError("Joints can only be added to URDF files")

        pos_str = f"{position[0]} {position[1]} {position[2]}"
        self._file.write(f'  <joint name="{name}" type="{joint_type}">\n')
        self._file.write(f'    <parent link="{parent}"/>\n')
        self._file.write(f'    <child link="{child}"/>\n')
        self._file.write(f'    <origin rpy="0 0 0" xyz="{pos_str}"/>\n')
        self._file.write(f'    <axis xyz="0 1 0"/>\n')
        self._file.write(
            f'    <limit effort="1.0" lower="-3.14159" upper="3.14159" velocity="1.0"/>\n'
        )
        self._file.write(f"  </joint>\n")
        return self

    def add_sensor_neuron(self, name: Union[str, int], link_name: str):
        """Add a sensor neuron to the neural network."""
        if self._filetype != "nndf":
            raise ValueError("Neurons can only be added to NNDF files")
        self._file.write(
            f'  <neuron name="{name}" type="sensor" linkName="{link_name}"/>\n'
        )
        return self

    def add_motor_neuron(self, name: Union[str, int], joint_name: str):
        """Add a motor neuron to the neural network."""
        if self._filetype != "nndf":
            raise ValueError("Neurons can only be added to NNDF files")
        self._file.write(
            f'  <neuron name="{name}" type="motor" jointName="{joint_name}"/>\n'
        )
        return self

    def add_synapse(
        self, source: Union[str, int], target: Union[str, int], weight: float
    ):
        """Add a synapse between neurons."""
        if self._filetype != "nndf":
            raise ValueError("Synapses can only be added to NNDF files")
        self._file.write(
            f'  <synapse sourceNeuronName="{source}" targetNeuronName="{target}" weight="{weight}"/>\n'
        )
        return self

    def end(self):
        """Close the current file."""
        if self._filetype == "urdf":
            self._file.write("</robot>\n")
        elif self._filetype == "sdf":
            self._file.write("</sdf>\n")
        elif self._filetype == "nndf":
            self._file.write("</neuralNetwork>\n")
        self._file.close()
        return self

    def _add_urdf_cube(self, name: str, pos: List[float], size: List[float]):
        """Add a cube link to URDF."""
        pos_str = f"{pos[0]} {pos[1]} {pos[2]}"
        size_str = f"{size[0]} {size[1]} {size[2]}"

        self._file.write(f'  <link name="{name}">\n')

        # Inertial - calculate proper inertia for a box
        mass = size[0] * size[1] * size[2]  # Simple mass calculation
        ixx = (mass / 12.0) * (size[1] ** 2 + size[2] ** 2)
        iyy = (mass / 12.0) * (size[0] ** 2 + size[2] ** 2)
        izz = (mass / 12.0) * (size[0] ** 2 + size[1] ** 2)

        self._file.write(f"    <inertial>\n")
        self._file.write(f'      <origin rpy="0 0 0" xyz="{pos_str}"/>\n')
        self._file.write(f'      <mass value="{mass}"/>\n')
        self._file.write(
            f'      <inertia ixx="{ixx}" ixy="0.0" ixz="0.0" iyy="{iyy}" iyz="0.0" izz="{izz}"/>\n'
        )
        self._file.write(f"    </inertial>\n")

        # Visual
        self._file.write(f"    <visual>\n")
        self._file.write(f'      <origin rpy="0 0 0" xyz="{pos_str}"/>\n')
        self._file.write(f"      <geometry>\n")
        self._file.write(f'        <box size="{size_str}"/>\n')
        self._file.write(f"      </geometry>\n")
        self._file.write(f'      <material name="Cyan">\n')
        self._file.write(f'        <color rgba="0 1.0 1.0 1.0"/>\n')
        self._file.write(f"      </material>\n")
        self._file.write(f"    </visual>\n")

        # Collision
        self._file.write(f"    <collision>\n")
        self._file.write(f'      <origin rpy="0 0 0" xyz="{pos_str}"/>\n')
        self._file.write(f"      <geometry>\n")
        self._file.write(f'        <box size="{size_str}"/>\n')
        self._file.write(f"      </geometry>\n")
        self._file.write(f"    </collision>\n")

        self._file.write(f"  </link>\n")

    def _add_sdf_cube(self, name: str, pos: List[float], size: List[float]):
        """Add a cube model to SDF."""
        pos_str = f"{pos[0]} {pos[1]} {pos[2]}"
        size_str = f"{size[0]} {size[1]} {size[2]}"

        self._file.write(f'  <model name="{name}">\n')
        self._file.write(f"    <pose>{pos_str} 0 0 0</pose>\n")
        self._file.write(f'    <link name="{name}">\n')

        # Inertial - calculate proper inertia for a box
        mass = size[0] * size[1] * size[2]
        ixx = (mass / 12.0) * (size[1] ** 2 + size[2] ** 2)
        iyy = (mass / 12.0) * (size[0] ** 2 + size[2] ** 2)
        izz = (mass / 12.0) * (size[0] ** 2 + size[1] ** 2)

        self._file.write(f"      <inertial>\n")
        self._file.write(f"        <mass>{mass}</mass>\n")
        self._file.write(f"        <inertia>\n")
        self._file.write(f"          <ixx>{ixx}</ixx>\n")
        self._file.write(f"          <ixy>0</ixy>\n")
        self._file.write(f"          <ixz>0</ixz>\n")
        self._file.write(f"          <iyy>{iyy}</iyy>\n")
        self._file.write(f"          <iyz>0</iyz>\n")
        self._file.write(f"          <izz>{izz}</izz>\n")
        self._file.write(f"        </inertia>\n")
        self._file.write(f"      </inertial>\n")

        # Visual
        self._file.write(f'      <visual name="visual">\n')
        self._file.write(f"        <geometry>\n")
        self._file.write(f"          <box>\n")
        self._file.write(f"            <size>{size_str}</size>\n")
        self._file.write(f"          </box>\n")
        self._file.write(f"        </geometry>\n")
        self._file.write(f"        <material>\n")
        self._file.write(f"          <ambient>0.0 1.0 1.0 1.0</ambient>\n")
        self._file.write(f"          <diffuse>0.0 1.0 1.0 1.0</diffuse>\n")
        self._file.write(f"        </material>\n")
        self._file.write(f"      </visual>\n")

        # Collision
        self._file.write(f'      <collision name="collision">\n')
        self._file.write(f"        <geometry>\n")
        self._file.write(f"          <box>\n")
        self._file.write(f"            <size>{size_str}</size>\n")
        self._file.write(f"          </box>\n")
        self._file.write(f"        </geometry>\n")
        self._file.write(f"      </collision>\n")

        self._file.write(f"    </link>\n")
        self._file.write(f"  </model>\n")


class NeuralNetwork:
    """Simplified neural network for robot control."""

    def __init__(self, nndf_file: str, body_id: int, output_folder: str = "data"):
        self.body_id = body_id
        self.output_folder = output_folder
        self.neurons = {}
        self.synapses = {}
        self._link_to_index = {}
        self._joint_to_index = {}
        self._load_network(nndf_file)
        self._prepare_simulation()

    def _load_network(self, filename: str):
        """Load neural network from NNDF file."""
        filepath = os.path.join(self.output_folder, filename)
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if "neuron" in line and "name=" in line:
                    self._parse_neuron(line)
                elif "synapse" in line:
                    self._parse_synapse(line)

    def _parse_neuron(self, line: str):
        """Parse neuron definition from XML line."""
        parts = line.split('"')
        name = parts[1]
        neuron_type = (
            PyroSim.SENSOR_NEURON
            if "sensor" in line
            else (PyroSim.MOTOR_NEURON if "motor" in line else PyroSim.HIDDEN_NEURON)
        )

        neuron = {
            "name": name,
            "type": neuron_type,
            "value": 0.0,
            "link_name": parts[5] if "linkName" in line else None,
            "joint_name": parts[5] if "jointName" in line else None,
        }
        self.neurons[name] = neuron

    def _parse_synapse(self, line: str):
        """Parse synapse definition from XML line."""
        parts = line.split('"')
        source = parts[1]
        target = parts[3]
        weight = float(parts[5])
        self.synapses[(source, target)] = weight

    def _prepare_simulation(self):
        """Prepare PyBullet simulation mappings."""
        # Map link names to indices
        for i in range(p.getNumJoints(self.body_id)):
            joint_info = p.getJointInfo(self.body_id, i)
            link_name = joint_info[12].decode("utf-8")
            joint_name = joint_info[1].decode("utf-8")
            self._link_to_index[link_name] = i
            self._joint_to_index[joint_name] = i

        # Add base link
        base_name = p.getBodyInfo(self.body_id)[0].decode("utf-8")
        self._link_to_index[base_name] = -1

    def update(self):
        """Update all neurons in the network."""
        # Update sensor neurons
        for name, neuron in self.neurons.items():
            if neuron["type"] == PyroSim.SENSOR_NEURON:
                neuron["value"] = self._get_touch_sensor(neuron["link_name"])

        # Update hidden and motor neurons
        for name, neuron in self.neurons.items():
            if neuron["type"] != PyroSim.SENSOR_NEURON:
                neuron["value"] = 0.0
                # Sum weighted inputs from synapses
                for (source, target), weight in self.synapses.items():
                    if target == name:
                        neuron["value"] += weight * self.neurons[source]["value"]
                # Apply activation function
                neuron["value"] = math.tanh(neuron["value"])

    def _get_touch_sensor(self, link_name: str) -> float:
        """Get touch sensor value for a link."""
        if link_name not in self._link_to_index:
            return -1.0

        link_index = self._link_to_index[link_name]
        contacts = p.getContactPoints(bodyA=self.body_id)

        for contact in contacts:
            # getContactPoints returns tuples where index 3 corresponds to the
            # link index of bodyA (this robot). We previously compared against
            # index 4 which is the link index of the other body, causing the
            # sensor to never register contact.
            if contact[3] == link_index:  # linkIndexA
                return 1.0
        return -1.0

    def get_motor_commands(self) -> Dict[str, float]:
        """Get motor commands for all motor neurons."""
        commands = {}
        for name, neuron in self.neurons.items():
            if neuron["type"] == PyroSim.MOTOR_NEURON:
                commands[neuron["joint_name"]] = neuron["value"]
        return commands

    def apply_motor_commands(self, control_mode=p.POSITION_CONTROL, max_force=500):
        """Apply motor commands to the robot."""
        commands = self.get_motor_commands()
        for joint_name, target in commands.items():
            if joint_name in self._joint_to_index:
                p.setJointMotorControl2(
                    bodyIndex=self.body_id,
                    jointIndex=self._joint_to_index[joint_name],
                    controlMode=control_mode,
                    targetPosition=target,
                    force=max_force,
                )


class World:
    """Simulation world manager."""

    def __init__(self, samples=100, robots={}, objects=[], output_folder="data"):
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setGravity(0, 0, -50)

        self.samples = samples
        self.output_folder = output_folder
        self.robots = {}
        self.pyrosim = PyroSim(output_folder)

        # Create objects first
        for obj_fn in objects:
            obj_fn(self.pyrosim)
            try:
                sdf_path = os.path.join(output_folder, obj_fn.__name__ + ".sdf")
                p.loadSDF(sdf_path)
            except Exception as e:
                print(f"Warning: Could not load SDF {sdf_path}: {e}")

        # Create robots
        for name, (urdf_fn, nndf_fn) in robots.items():
            try:
                self.robots[name] = Robot(name, urdf_fn, nndf_fn, samples, self.pyrosim)
            except Exception as e:
                print(f"Warning: Could not create robot {name}: {e}")

        p.loadURDF("plane.urdf")

    def __enter__(self):
        return self

    def step(self):
        p.stepSimulation()

    def __exit__(self, type, value, traceback):
        for robot in self.robots.values():
            robot.save_data()
        p.disconnect()


class Robot:
    """Robot with neural network control."""

    def __init__(self, name: str, urdf_fn, nndf_fn, samples: int, pyrosim: PyroSim):
        self.name = name
        self.samples = samples
        self.output_folder = pyrosim.output_folder
        self.senses = {}

        # Generate files
        urdf_fn(pyrosim)
        nndf_fn(pyrosim)

        # Load robot
        urdf_path = os.path.join(pyrosim.output_folder, urdf_fn.__name__ + ".urdf")
        self.id = p.loadURDF(urdf_path)
        self.nn = NeuralNetwork(
            nndf_fn.__name__ + ".nndf", self.id, pyrosim.output_folder
        )

        # Initialize sensor data storage
        for i in range(p.getNumJoints(self.id)):
            link_name = p.getJointInfo(self.id, i)[12].decode("utf-8")
            self.senses[f"{link_name}_touch"] = np.zeros(samples)

        base_name = p.getBodyInfo(self.id)[0].decode("utf-8")
        self.senses[f"{base_name}_touch"] = np.zeros(samples)

    def sense(self, index: int):
        """Record sensor data."""
        for sensor_name in self.senses:
            link_name = sensor_name.replace("_touch", "")
            # print(self.nn._get_touch_sensor(link_name))
            self.senses[sensor_name][index] = self.nn._get_touch_sensor(link_name)

    def think(self, index: int):
        """Update neural network."""
        self.nn.update()

    def act(self, index: int):
        """Apply motor commands."""
        self.nn.apply_motor_commands()

    def save_data(self):
        """Save sensor data."""
        for sensor_name, data in self.senses.items():
            filepath = os.path.join(
                self.output_folder, f"{self.name}_{sensor_name}.npy"
            )
            np.save(filepath, data)


# ==================== ROBOT DEFINITIONS ====================


def box(pyrosim: PyroSim):
    """Create a box object."""
    pyrosim.start_sdf("box.sdf")
    pyrosim.add_cube("Box", [-2, 2, 0.5], [1, 1, 1])
    pyrosim.end()


def robot(pyrosim: PyroSim):
    """Create a simple bipedal robot."""
    pyrosim.start_urdf("robot.urdf")
    pyrosim.add_cube("torso", [0, 0, 1.5], [1, 1, 1])
    pyrosim.add_cube("rightleg", [0.5, 0, -0.5], [1, 1, 1])
    pyrosim.add_cube("leftleg", [-0.5, 0, -0.5], [1, 1, 1])
    pyrosim.add_joint("torso_rightleg", "torso", "rightleg", "revolute", [0.5, 0, 1])
    pyrosim.add_joint("torso_leftleg", "torso", "leftleg", "revolute", [-0.5, 0, 1])
    pyrosim.end()


def brain(pyrosim: PyroSim):
    """Create a simple neural network brain."""
    pyrosim.start_nndf("brain.nndf")
    pyrosim.add_sensor_neuron(0, "torso")
    pyrosim.add_sensor_neuron(1, "rightleg")
    pyrosim.add_sensor_neuron(2, "leftleg")
    pyrosim.add_motor_neuron(3, "torso_rightleg")
    pyrosim.add_motor_neuron(4, "torso_leftleg")

    # Add synapses with random weights
    pyrosim.add_synapse(1, 3, random.random())
    pyrosim.add_synapse(2, 3, random.random())
    pyrosim.add_synapse(1, 4, random.random())
    pyrosim.add_synapse(2, 4, random.random())

    pyrosim.end()

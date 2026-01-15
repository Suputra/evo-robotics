"""MuJoCo simulation module for morpho-rl.

This module provides MuJoCo-based simulation classes that mirror the structure
of the PyBullet-based pyrosim module, allowing you to use MuJoCo for new simulations.
"""

import numpy as np
import mujoco
import mujoco.viewer
import math
import os
from typing import Dict, List, Optional, Union


class MuJoCoSim:
    """MuJoCo simulation helper with MJCF file generation."""

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
        self._link_index = 0
        self._link_to_index = {}
        self._joint_to_index = {}
        self._bodies = []
        self._joints = []
        os.makedirs(self.output_folder, exist_ok=True)

    def start_mjcf(self, filename: str):
        """Start creating an MJCF (MuJoCo XML) file."""
        self.reset()
        self._filetype = "mjcf"
        self._filename = filename
        filepath = os.path.join(self.output_folder, filename)
        self._file = open(filepath, "w")
        self._file.write('<mujoco model="robot">\n')
        self._file.write('  <option gravity="0 0 -9.81" timestep="0.002"/>\n')
        self._file.write("  <worldbody>\n")
        self._file.write('    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>\n')
        self._file.write(
            '    <geom name="floor" type="plane" size="10 10 0.1" rgba="0.8 0.8 0.8 1"/>\n'
        )
        return self

    def start_nndf(self, filename: str):
        """Start creating a neural network definition file."""
        self.reset()
        self._filetype = "nndf"
        filepath = os.path.join(self.output_folder, filename)
        self._file = open(filepath, "w")
        self._file.write("<neuralNetwork>\n")
        return self

    def add_body(
        self,
        name: str,
        pos: List[float] = [0, 0, 0],
        size: List[float] = [0.5, 0.5, 0.5],
        parent: Optional[str] = None,
        joint_name: Optional[str] = None,
        joint_type: str = "hinge",
        joint_axis: List[float] = [0, 1, 0],
        joint_range: List[float] = [-3.14159, 3.14159],
    ):
        """Add a body (link) to the MJCF file."""
        if self._filetype != "mjcf":
            raise ValueError("Bodies can only be added to MJCF files")

        pos_str = f"{pos[0]} {pos[1]} {pos[2]}"
        size_str = (
            f"{size[0] / 2} {size[1] / 2} {size[2] / 2}"  # MuJoCo uses half-sizes
        )
        axis_str = f"{joint_axis[0]} {joint_axis[1]} {joint_axis[2]}"
        range_str = f"{joint_range[0]} {joint_range[1]}"

        indent = "    "
        self._file.write(f'{indent}<body name="{name}" pos="{pos_str}">\n')

        if joint_name:
            self._file.write(
                f'{indent}  <joint name="{joint_name}" type="{joint_type}" '
                f'axis="{axis_str}" range="{range_str}"/>\n'
            )
            self._joint_to_index[joint_name] = len(self._joint_to_index)

        mass = size[0] * size[1] * size[2]
        self._file.write(f'{indent}  <inertial pos="0 0 0" mass="{mass}"/>\n')
        self._file.write(
            f'{indent}  <geom name="{name}_geom" type="box" size="{size_str}" '
            f'rgba="0 1 1 1"/>\n'
        )

        self._link_to_index[name] = self._link_index
        self._link_index += 1
        self._bodies.append(name)

        return self

    def end_body(self):
        """Close the current body tag."""
        if self._filetype != "mjcf":
            raise ValueError("end_body can only be used with MJCF files")
        self._file.write("    </body>\n")
        return self

    def add_cube(
        self, name: str, pos: List[float] = [0, 0, 0], size: List[float] = [1, 1, 1]
    ):
        """Add a standalone cube body (convenience method)."""
        return self.add_body(name, pos, size)

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
        if self._filetype == "mjcf":
            self._file.write("  </worldbody>\n")
            self._write_actuators()
            self._write_sensors()
            self._file.write("</mujoco>\n")
        elif self._filetype == "nndf":
            self._file.write("</neuralNetwork>\n")
        self._file.close()
        return self

    def _write_actuators(self):
        """Write actuator definitions for all joints."""
        if self._joint_to_index:
            self._file.write("  <actuator>\n")
            for joint_name in self._joint_to_index:
                self._file.write(
                    f'    <motor name="{joint_name}_motor" joint="{joint_name}" '
                    f'gear="100" ctrllimited="true" ctrlrange="-1 1"/>\n'
                )
            self._file.write("  </actuator>\n")

    def _write_sensors(self):
        """Write sensor definitions for touch sensing."""
        if self._bodies:
            self._file.write("  <sensor>\n")
            for body_name in self._bodies:
                self._file.write(
                    f'    <touch name="{body_name}_touch" site="{body_name}_site"/>\n'
                )
            self._file.write("  </sensor>\n")


class MuJoCoNeuralNetwork:
    """Neural network for MuJoCo robot control."""

    def __init__(
        self,
        nndf_file: str,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        output_folder: str = "data",
    ):
        self.model = model
        self.data = data
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
            MuJoCoSim.SENSOR_NEURON
            if "sensor" in line
            else (
                MuJoCoSim.MOTOR_NEURON if "motor" in line else MuJoCoSim.HIDDEN_NEURON
            )
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
        """Prepare MuJoCo simulation mappings."""
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name:
                self._joint_to_index[joint_name] = i

        for i in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name:
                self._link_to_index[body_name] = i

    def update(self):
        """Update all neurons in the network."""
        for name, neuron in self.neurons.items():
            if neuron["type"] == MuJoCoSim.SENSOR_NEURON:
                neuron["value"] = self._get_touch_sensor(neuron["link_name"])

        for name, neuron in self.neurons.items():
            if neuron["type"] != MuJoCoSim.SENSOR_NEURON:
                neuron["value"] = 0.0
                for (source, target), weight in self.synapses.items():
                    if target == name:
                        neuron["value"] += weight * self.neurons[source]["value"]
                neuron["value"] = math.tanh(neuron["value"])

    def _get_touch_sensor(self, link_name: str) -> float:
        """Get touch sensor value for a body."""
        if link_name not in self._link_to_index:
            return -1.0

        body_id = self._link_to_index[link_name]
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_body = self.model.geom_bodyid[contact.geom1]
            geom2_body = self.model.geom_bodyid[contact.geom2]
            if geom1_body == body_id or geom2_body == body_id:
                return 1.0
        return -1.0

    def get_motor_commands(self) -> Dict[str, float]:
        """Get motor commands for all motor neurons."""
        commands = {}
        for name, neuron in self.neurons.items():
            if neuron["type"] == MuJoCoSim.MOTOR_NEURON:
                commands[neuron["joint_name"]] = neuron["value"]
        return commands

    def apply_motor_commands(self):
        """Apply motor commands to the robot."""
        commands = self.get_motor_commands()
        for joint_name, target in commands.items():
            actuator_name = f"{joint_name}_motor"
            try:
                actuator_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
                )
                if actuator_id >= 0:
                    self.data.ctrl[actuator_id] = target
            except Exception:
                pass


class MuJoCoWorld:
    """MuJoCo simulation world manager."""

    def __init__(
        self,
        samples: int = 100,
        robots: Dict = {},
        output_folder: str = "data",
        render: bool = True,
    ):
        self.samples = samples
        self.output_folder = output_folder
        self.robots = {}
        self.sim = MuJoCoSim(output_folder)
        self.render = render
        self._viewer = None
        self._model = None
        self._data = None

        for name, (mjcf_fn, nndf_fn) in robots.items():
            try:
                self.robots[name] = MuJoCoRobot(
                    name, mjcf_fn, nndf_fn, samples, self.sim
                )
                self._model = self.robots[name].model
                self._data = self.robots[name].data
            except Exception as e:
                print(f"Warning: Could not create robot {name}: {e}")

    def __enter__(self):
        if self.render and self._model is not None:
            self._viewer = mujoco.viewer.launch_passive(self._model, self._data)
        return self

    def step(self):
        """Step the simulation."""
        if self._model is not None and self._data is not None:
            mujoco.mj_step(self._model, self._data)
            if self._viewer is not None:
                self._viewer.sync()

    def __exit__(self, type, value, traceback):
        for robot in self.robots.values():
            robot.save_data()
        if self._viewer is not None:
            self._viewer.close()


class MuJoCoRobot:
    """Robot with neural network control for MuJoCo."""

    def __init__(
        self,
        name: str,
        mjcf_fn,
        nndf_fn,
        samples: int,
        sim: MuJoCoSim,
    ):
        self.name = name
        self.samples = samples
        self.output_folder = sim.output_folder
        self.senses = {}

        mjcf_fn(sim)
        nndf_fn(sim)

        mjcf_path = os.path.join(sim.output_folder, mjcf_fn.__name__ + ".mjcf")
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)

        self.nn = MuJoCoNeuralNetwork(
            nndf_fn.__name__ + ".nndf", self.model, self.data, sim.output_folder
        )

        for body_name in self.nn._link_to_index:
            if body_name != "world":
                self.senses[f"{body_name}_touch"] = np.zeros(samples)

    def sense(self, index: int):
        """Record sensor data."""
        for sensor_name in self.senses:
            link_name = sensor_name.replace("_touch", "")
            if index < self.samples:
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

# pyrosim2: Simulations with robots using pybullet

A streamlined, concise PyBullet simulation helper library.

This project is setup with uv. For instructions on how setup uv, check out https://suputra.github.io/snippets/uv.

## Quickstart

Checkout sim.py for a simple start script of this

setup venv:
```
uv venv 
uv pip install -r requirements.txt
chmod +x ./sim.py
```

then run:
```
./sim.py 
```

## Core Classes

- **`PyroSim`**: File generation for URDF, SDF, and NNDF files
- **`World`**: Simulation environment manager with automatic setup/cleanup
- **`Robot`**: Simplified robot with automatic sensor setup and neural network integration
- **`NeuralNetwork`**: Streamlined neural network for robot control


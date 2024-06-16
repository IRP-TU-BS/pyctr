### Concentric Tube Continuum Robot Models - The PyCtr Package

#### Overview
PyCtr is an implementation of a geometrically exact model for concentric tube continuum robots, inspired by the work of [Rucker et al. and John D. Till](https://github.com/JohnDTill/ContinuumRobotExamples). It aims to provide a functional and understandable model for educational and research purposes.

### Installation
To install the package, you can use `poetry`, `rye`, or `pip`. Use the `pyproject.toml` file for managing dependencies and installation.

#### Poetry
1. **Install Poetry (if not already installed):**
   ```bash
   pip install poetry
   ```

2. **Install the package:**
   Navigate to the project directory and run:
   ```bash
   poetry install
   ```
#### Rye
1. **Install Rye (if not already installed):**
   ```bash
   curl -sSf https://rye.astral.sh/get | bash
   ```

2. **Install the package:**
   Navigate to the project directory and run:
   ```bash
   rye sync
   ```
#### Pip
1. **Install the package**
   ```bash
   pip install <path to pyctr>
   ```

### Using the Models

#### Importing and Configuring the Robot
The package provides models for simulating the configuration and behavior of concentric tube continuum robots.

Example usage:
```python
from pyctr import ConcentricTubeContinuumRobot

# Initialize the robot with specific parameters
robot = ConcentricTubeContinuumRobot(tube_parameters)

# Change the configuration
robot.change_configuration(configuration_parameters)

# Apply external forces
robot.apply_external_forces(forces)
```

#### Visualizing the Robot
The package includes tools for visualizing the robot configurations.

Example usage:
```python
from pyctr import plot_robot

# Plot the current configuration of the robot
plot_robot(robot)
```

### Examples from Notebooks

#### Changing Configuration and Applying External Forces
This example demonstrates how to change the configuration of the robot and apply external forces.

```python
from pyctr import ConcentricTubeContinuumRobot

# Initialize the robot
robot = ConcentricTubeContinuumRobot(tube_parameters)

# Change the configuration
new_configuration = {'param1': value1, 'param2': value2}
robot.change_configuration(new_configuration)

# Apply external forces
external_forces = {'force1': value1, 'force2': value2}
robot.apply_external_forces(external_forces)

# Verify the changes
print(robot.current_state)
```

#### Plotting Robots
This example illustrates how to plot the robot using the provided visualization tools.

```python
from pyctr import ConcentricTubeContinuumRobot, plot_robot

# Initialize the robot
robot = ConcentricTubeContinuumRobot(tube_parameters)

# Plot the robot
plot_robot(robot)
```

### Future Plans
- Improve documentation
- Make the package available on PyPI
- Extend to dynamic models

This summary provides a concise overview of the PyCtr package, including installation instructions, basic usage of the models, and visualization techniques. For detailed documentation, refer to the project's documentation generated via `make html` in the `doc` folder.

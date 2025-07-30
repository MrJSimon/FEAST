# FEAST
**feast** is a simulation framework focused on the modeling and analysis of spinning systems using finite element methods. Designed for research, development, and exploration in rotor dynamics and turbomachinery, FEAST aims to be modular, extensible, and easy to integrate with modern numerical workflows.

![FEAST](documentation/images/feast_logo.png)

# Installation
Install **feast** by cloning the repository onto your local machine using the following command

    git clone https://github.com/MrJSimon/FEAST.git

### Requirements
This program was tested using

    python 3.10.9

Install the required Python packages, listed in requirements.txt, with:  

    pip install -r requirements.txt

# Getting started
Coming soon....


## 📁 Project Structure
    FEAST/
    └── Functions/
        │
        ├── Elements/
        │   ├── Beam.py
        │   ├── Shaft.py
        │   └── Disk.py
        │   # Add more element types as needed
        │
        ├── Solvers/
        │   ├── StaticSolver.py
        │   ├── DynamicSolver.py
        │   └── TimeIntegration.py
        │
        ├── StressDescription/
        │   ├── StressTensor.py
        │   ├── VonMises.py
        │   └── PrincipalStress.py
        │
        ├── StrainDescription/
        │   ├── StrainTensor.py
        │   ├── SmallStrain.py
        │   └── LargeStrain.py
        │
        ├── Explicit/
        │   ├── TimeStepper.py
        │   ├── CentralDifference.py
        │   └── StabilityCheck.py
        │
        ├── Implicit/
        │   ├── NewmarkBeta.py
        │   ├── BackwardEuler.py
        │   └── SolverWrapper.py
        │
        └── IterationMethods/
            ├── NewtonRaphson.py
            ├── EulerMethod.py
            └── ConvergenceCriteria.py

# Basic workflow
Coming soon....

# Visualizations
Coming soon....

# Output Files
Coming soon....

# Documentation
For more details, visit the [Wiki](https://github.com/MrJSimon/feast/wiki).

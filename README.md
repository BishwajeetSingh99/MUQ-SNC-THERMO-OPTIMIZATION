
# MUQ-SNC-THERMO-OPTIMIZATION

A Novel Temperature-Dependent Optimization framework for Thermodynamic Parameter of Chemical Kinetic Model

## Workflow

Step 0: Compile shared libraries
- Compile the required shared object (*.so) files before running the framework.

Step 1: Convert CHEMKIN to YAML

Example:
ck2yaml --input=chem.inp --thermo=therm.dat --transport=tran.dat --permissive

Example:
ck2yaml --input=MB_MB2D.inp --thermo=MB+MB2D_therm.dat --transport=Mb+MB2d_trans.dat --permissive

Step 2: Generate XML files
- Generate XML files for species considered in the sensitivity analysis.

Step 3: Prepare input files
- Target file
- Addendum file
- Mechanism files:
    * mech.yaml
    * therm.dat
    * tran.dat

Step 4: Run simulations

Optimization:
python3 THERMO_PARAMETER_OPTIMIZATION/run_script.py [target_file]

Nominal simulations:
python3 NOMINAL_SIMULATIONS/run_nominal_sim.py [target_file]

Sensitivity analysis:
python3 SENSITIVITY_ANALYSIS_THERMO/sens.py [target_file]

## Repository Structure

MUQ-SNC-THERMO-OPTIMIZATION/
├── THERMO_PARAMETER_OPTIMIZATION/
├── SENSITIVITY_ANALYSIS_THERMO/
├── NOMINAL_SIMULATIONS/
└── README.md


Author:
Bishwajeet Singh
ma23d007@smail.iitm.ac.in
Department of Mathematics
IIT Madras


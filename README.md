# MolecularSim

This repository contains the software and resources for *Laboratory Course Computational Physics, WS 24/25*.

## Resources

Resources such as [Lab Guide](resources/MC_Mol.pdf) and [recommended Reading](resources/Essentials_Comp_Chem_Cramer.pdf) can be found in the [resources](resources) directory.

## Contributors

[Niklas Peter](mailto:niklas.peter@student.uibk.ac.at)

[Dmytro Lazykov](mailto:dmytro.iazykov@student.uibk.ac.at)

[Christof Stein](mailto:christof.stein@student.uibk.ac.at).

## Development

For development the project is installable in editable mode via pip using additional tools from [dev] dependencies, which already install all needed requirements:
```bash
pip install -e .[dev]
```

## Running calculations

Calculations will be run using apps stored in the [src/apps](src/apps) directory. 

Currently only a small [showcase applet](src/apps/calculate_between.py) exists to show force calculations working. It can be called from the project root using example data by calling:

```bash
python src/apps/calculate_between.py -f tests/data/water_zmatrix.com -n 2 -t 10 -b 10,10,10
```

This simply generaates a system of n=2 molecules and runs potential calculations for this system.

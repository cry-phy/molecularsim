# MolecularSim

This repository contains the software and resources for *Laboratory Course Computational Physics, WS 24/25*.

## Resources

Resources such as [Lab Guide](resources/MC_Mol.pdf) and [recommended Reading](resources/Essentials_Comp_Chem_Cramer.pdf) can be found in the [resources](resources) directory.

## Contributors

[Niklas Peter](mailto:niklas.peter@student.uibk.ac.at)

[Dmytro Iazykov](mailto:dmytro.iazykov@student.uibk.ac.at)

[Christof Stein](mailto:christof.stein@student.uibk.ac.at).

## Development

For development the project is installable in editable mode via pip using additional tools from [dev] dependencies, which already install all needed requirements:
```bash
pip install -e .[dev]
```

## Running calculations

Calculations will be run using apps stored in the [src/apps](src/apps) directory. 

The following apps are implemented:

### Streamlit Webapp

Main interaction is done via [streamlit webapp](src/apps/streamlit_update.py) bunling the interactions into a nice interface.

This requires installing the app dependencies:

```bash
pip install .[apps]
```

but provides UI in browser via:

```bash
streamlit run src/apps/streamlit_update.py
```

### Headless mode

[Headless Execution](src/apps/headless.py) is used to presimulate specific data for further study. In essence it is used to prepare systems into equilibrium states for macroscopic parameter extraction.

```bash
python src/apps/run_headless.py \
    -f tests/data/water_zmatrix.com \
    -n 2 \
    -t 10 \
    -b 10,10,10 \
    -s 10000 \
    -o out/state_final.pickle \
    --seed 10
```

### Debug view

Distance calculations using minimum image convention can be inspected in [debug app](src/apps/headless.py):


### Showcase App

Currently a small [showcase applet](src/apps/calculate_between.py) exists to show how interaction with the module might work. It can be called:

```bash
python src/apps/debug_viewer.py \
  -f tests/data/water_zmatrix.com \
  -n 5 \
  -t 10 \
  -b 10,10,10 \
  --seed 1
```


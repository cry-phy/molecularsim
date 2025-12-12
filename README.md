# MolecularSim

This repository contains the software and resources for *Laboratory Course Computational Physics, WS 24/25*.

The project is structured as two different installable python packages:

- [**`molecsim`** ](molecsim/) as the core simulation packages.
- [**`molecsim-tools`**](molecsim_tools/) for visualization and development tools.

## Resources

Resources such as [Lab Guide](resources/MC_Mol.pdf) and [recommended Reading](resources/Essentials_Comp_Chem_Cramer.pdf) can be found in the [resources](resources) directory.

## Contributors

[Niklas Peter](mailto:niklas.peter@student.uibk.ac.at)

[Dmytro Iazykov](mailto:dmytro.iazykov@student.uibk.ac.at)

[Christof Stein](mailto:christof.stein@student.uibk.ac.at).

## Development setup

The repository contains two installable subprojects.

### Install core library (simulation engine)

```bash
pip install -e molecsim
```

For development and testing:

```bash
pip install -e molecsim[dev]
```

This installs the simulation engine without any UI or Streamlit dependencies.

### Install tools and apps (optional)

The tools package provides Streamlit and other analysis utilities and depends on `molecsim`:

```bash
pip install -e molecsim-tools
```

## Running calculations

Apps are intentionally not part of the core library. They live in the `molecsim-tools` package and are optionally exposed via small launcher scripts under `apps/`.

## Streamlit Web App

The main interactive interface is a Streamlit web application.

After installing `molecsim-tools`, you can either:

### Run via installed command (recommended)

```bash
molecsim-streamlit
```

### Or run the thin launcher script

```bash
python apps/streamlit_mc.py
```

This launches the UI in your browser and provides live interaction with the Monte-Carlo simulation.

## Headless mode

Headless execution is used to pre-equilibrate systems or generate data for further analysis.

Example:

```bash
python apps/headless.py \
    -f tests/data/water_zmatrix.com \
    -n 2 \
    -t 10 \
    -b 10,10,10 \
    -s 10000 \
    -o out/state_final.pickle \
    --seed 10
```

This runs a Monte-Carlo simulation without UI and writes the final system state to disk.

## Debug / inspection tools

Distance calculations and minimum-image conventions can be inspected using the debug utilities:

```bash
python apps/show_traces.py
```

These tools are intended for development and validation, not for production runs.

## Showcase app

A small showcase app demonstrates how to interact with the simulation engine programmatically:

```bash
python apps/calculate_between.py \
    -f tests/data/water_zmatrix.com \
    -n 5 \
    -t 10 \
    -b 10,10,10 \
    --seed 1
```

## Notes

- `molecsim` never depends on Streamlit or UI code.
- `molecsim-tools` may change rapidly and is intended for development, analysis, and visualization.
- The `apps/` directory only contains thin entry scripts and is not packaged.

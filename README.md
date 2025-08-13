## Territorial nitrogen flows and metabolic typologies of EU Agri-Food Systems, 1990–2019

**Description :**  
Python project to process territorial nitrogen (N) fluxes in European agricultural systems (121 territories, 30 years). 

---

## Overview of code repository
```
EUAgri-MetaN/
├── codebase
├── data
└── main.py
```

`EUAgri-MetaN/codebase` contains scripts which, if run in alphabetical order, reproduces all the model results.

`EUAgri-MetaN/codebase/utils.py` contains some common functions used in the scripts.

The scripts are divided into sections:

- a: crop production
- b: crop areas
- c: yields
- d: land use
- e: synthetic N fertilizer
- f: fertilizer allocation
- g: atmospheric deposition
- h: symbiotic fixation
- i: livestock excretion
- j: manure allocation
- k: arable land N budget 
- l: metabolic flows
- m: territorial typologies

---
## Input data and parameters

The `EUAgri-MetaN/data` directory contains almost all the data needed for the calculations.  

**Note:** Atmospheric deposition data are not included here; please see the `scripts/atmospheric_deposition.py` script for details on how to generate or process these data.

---

## Installation
**Prerequisites:**
- Python 3.10+ recommended
- It is strongly suggested to use a virtual environment

**Setup:**
```bash
python -m venv .venv  # Create a virtual environment
. .venv/bin/activate  # Activate the virtual environment
pip install -r requirements.txt
```

## Running the calculation

Run the python script `EUAgri-MetaN/main.py`.


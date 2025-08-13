"""
Script name: main.py
Description: Executes the entire project workflow.
Author: Ludovic Harter
Created: 2025-05-01
Last modified: 2025-08-13
Version: 1.0
Project: Territorial nitrogen flows and metabolic typologies of EU Agri-Food Systems, 1990–2019
License: MIT
"""

#%% Code and libraries
from codebase.a_crop_production import run_production
from codebase.b_crop_areas import run_areas
from codebase.c_yields import run_yield
from codebase.d_land_use import run_land_use
from codebase.e_synthetic_fertilizer import run_fertilizer
from codebase.f_fertilizer_allocation import run_fertilizer_allocation
from codebase.h_symbiotic_fixation import run_BNF
from codebase.i_livestock_excretion import run_excretion
from codebase.j_manure_allocation import run_manure
from codebase.k_arable_budget import run_budget
from codebase.l_metabolic_flows import run_metabolic_data
from codebase.m_territory_typologies import run_typologies

#%% --- Crop production ---

production = run_production()
areas = run_areas()
crop_production_all_categories = run_yield()

# Save
crop_production_all_categories.to_csv('data/outputs/crop_production_all_categories.csv')

#%% --- Land use ---

land_use = run_land_use()

# Save
land_use.to_csv('data/outputs/land_areas.csv')

#%% --- Synthetic fertilizer ---

total_fertilizer_quantities = run_fertilizer()
fertilizer_allocation = run_fertilizer_allocation()

# Save
fertilizer_allocation.to_csv('data/outputs/synthetic_fertilizer.csv')

#%% --- Symbiotic fixation ---

symbiotic_fixation = run_BNF()

# Save
symbiotic_fixation.to_csv('data/outputs/symbiotic_fixation.csv')

#%% --- Animal excretion ---

animal_excretion = run_excretion()

# Save
animal_excretion.to_csv('data/outputs/animal_excretion.csv')

#%% --- Manure allocation ---

manure = run_manure()

# Save
manure.to_csv('data/outputs/manure_allocation.csv')

#%% --- Arable budget ---

arable_budget = run_budget()

# Save
arable_budget.to_csv('data/outputs/arable_budget.csv')

#%% --- Metabolic data ---

metabolic = run_metabolic_data()

# Save
metabolic.to_csv('data/outputs/metabolic_data.csv')

#%% --- Typology ---

typologies = run_typologies()

# Save
typologies.to_csv('data/outputs/typologies.csv')


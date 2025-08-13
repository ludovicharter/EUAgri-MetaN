"""
Script name: h_symbiotic_fixation.py
Description: Estimates symbiotic fixation for each territory.
Author: Ludovic Harter
Created: 2025-05-01
Last modified: 2025-08-13
Version: 1.0
Project: Territorial nitrogen flows and metabolic typologies of EU Agri-Food Systems, 1990–2019
License: MIT
"""

#%% --- Libraries ---
import numpy as np
import pandas as pd

#%% Code function
def run_BNF():
    print("Running BNF workflow...")

    #%% --- Define the period and regions ---

    years = np.arange(1990, 2020)
    regions = pd.read_csv('data/regions.csv', sep=';')

    # %% --- Define a new dataset to store deposition rates ---

    # Define nitrogen-fixing crop categories
    categories = ['pulses', 'temporary grassland', 'forage legumes', 'soy']

    # Create the cartesian product of region, year, and category
    final_fixation = (
        pd.MultiIndex.from_product(
            [sorted(regions['NUTS_ID']), years, categories],
            names=["region", "year", "category"]
        )
        .to_frame(index=False)
    )

    # Map region names from the regions DataFrame
    region_name_map = regions.set_index('NUTS_ID')['name']
    region_names = final_fixation['region'].map(region_name_map)

    # Insert 'region name' as the second column
    final_fixation.insert(
        loc=1,
        column='region name',
        value=region_names
    )

    # Add metadata columns with default values
    final_fixation = final_fixation.assign(
        value=None,
        label="symbiotic fixation",
        unit="Gg N",
        confidence=None
    )

    #%% --- Import production data: soya ---

    # Iterate each year and region
    for year in years:
        for region in regions['NUTS_ID']:

            prod = pd.read_csv(f'data/outputs/intermediate_datasets/production/production_a_b_{year}.csv')
            soy = prod['soy'].loc[prod['region'] == region]

            # Proceed only if value exists
            if not soy.empty and pd.notna(soy.iloc[0]):
                production_value = soy.iloc[0]

                # Convert soy production to Gg N
                production_value = production_value * 0.0608
                # Calculate BNF from soy
                bnf_value = (production_value * 0.57 * 1.4) / 0.73
            else:
                bnf_value = 0

            # Assign value to final_fixation
            final_fixation.loc[
                (final_fixation['year'] == year) &
                (final_fixation['region'] == region) &
                (final_fixation['category'] == 'soy'),
                ['value', 'confidence']
            ] = [bnf_value, 'estimated']

    #%% --- Import production data: pulse, temporary grassland and forage legumes ---

    # Load production data once
    prod = pd.read_csv('data/outputs/crop_production_all_categories.csv')

    # Define BNF calculation parameters per crop
    bnf_params = {
        'pulses': lambda v: (v * 0.68 * 1.3) / 0.75,
        'forage legumes': lambda v: (v * 0.68 * 1.3) / 0.5,
        'temporary grassland': lambda v: 0.25 * ((v * 0.78 * 1.7) / 0.9),
    }

    # Normalize crop names in production data for easier matching
    prod['crop'] = prod['crop'].str.lower()

    # Loop over region and year combinations
    for year in years:
        for region in regions['NUTS_ID']:
            for crop, bnf_func in bnf_params.items():
                # Extract the production value safely
                mask = (
                    (prod['region'] == region) &
                    (prod['year'] == year) &
                    (prod['crop'] == crop) &
                    (prod['symbol'] == 'H')
                )
                values = prod.loc[mask, 'value']

                # Proceed only if value exists
                if not values.empty and pd.notna(values.iloc[0]):
                    production_value = values.iloc[0]
                    bnf_value = bnf_func(production_value)
                else:
                    bnf_value = 0

                # Assign to final_fixation
                final_fixation.loc[
                    (final_fixation['region'] == region) &
                    (final_fixation['year'] == year) &
                    (final_fixation['category'] == crop),
                    ['value', 'confidence']
                ] = [bnf_value, 'estimated']

    return final_fixation

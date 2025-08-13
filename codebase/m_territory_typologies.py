"""
Script name: m_territory_typologies.py
Description: Estimates typologies for each territory.
Author: Ludovic Harter
Created: 2025-05-01
Last modified: 2025-08-13
Version: 1.0
Project: Territorial nitrogen flows and metabolic typologies of EU Agri-Food Systems, 1990–2019
License: MIT
"""

#%% --- Libraries ---
import pandas as pd
import numpy as np
import codebase.utils as utils
import pickle
import matplotlib.pyplot as plt
import warnings
import geopandas as gpd
import os
import seaborn as sns
import matplotlib.patches as mpatches

#%% Code function
def run_typologies():
    print("Running typologies workflow...")

    #%% --- Import data ---

    # Load region ids
    regions = pd.read_csv('data/regions.csv', sep=';')
    # Define the period
    years = np.arange(1990, 2020)
    # Load metabolic data
    metabolic = pd.read_csv('data/outputs/metabolic_data.csv')

    # Create the output file
    indicators = [
        ('typ', 'metabolic typology', 'no unit'),
        ('urb_typ', 'urban typology', 'no unit'),
    ]

    # Build a MultiIndex over region, crop, year, and indicator code
    mi = pd.MultiIndex.from_product(
        [
            sorted(regions['NUTS_ID']),  # all NUTS regions
            years,  # all years
            [symbol for symbol, *_ in indicators]  # A, H, Y
        ],
        names=["region", "year", "symbol"]
    )

    # Convert to a DataFrame
    typologies = mi.to_frame(index=False)

    # Map code to label and unit
    label_map = {code: label for code, label, unit in indicators}
    unit_map = {code: unit for code, label, unit in indicators}

    typologies['label'] = typologies['symbol'].map(label_map)
    typologies['unit'] = typologies['symbol'].map(unit_map)
    typologies['value'] = np.nan

    # Add the region name by mapping from regions['name']
    name_map = dict(zip(regions['NUTS_ID'], regions['name']))
    typologies.insert(
        loc=typologies.columns.get_loc('region') + 1,
        column='region name',
        value=typologies['region'].map(name_map)
    )

    #%% --- Typology conditions ---

    for region in regions['NUTS_ID']:
        print(f'Processing region {region}...')
        for year in years:
            # Define a helper to safely get scalar value or return None
            def get_value(symbol):
                mask = (
                    (metabolic['region'] == region) &
                    (metabolic['year'] == int(year)) &
                    (metabolic['symbol'] == symbol)
                )
                values = metabolic.loc[mask, 'value']
                return values.squeeze() if not values.empty else None

            # Retrieve values
            H = get_value('H_ingestion')  # Human ingestion
            C = get_value('Agri_total')  # Total agricultural production
            P = get_value('H_total')  # Total crop production
            A = get_value('total_A_ingestion')  # Total animal ingestion
            D = get_value('L_density')  # Livestock density
            I = get_value('F_import')  # Net import feed
            G = get_value('PG_H')  # Permanent grassland production
            F = get_value('F_ingestion')  # Effective fodder ingestion
            N = get_value('total_N_input')  # Soil input from manure

            # Mask for setting the typology
            mask_T = (
                (typologies['region'] == region) &
                (typologies['year'] == int(year)) &
                (typologies['symbol'] == 'typ')
            )

            mask_U = (
                (typologies['region'] == region) &
                (typologies['year'] == int(year)) &
                (typologies['symbol'] == 'urb_typ')
            )

            # Only assign if all necessary values are present
            if None in [H, C, P, A, D, I, G, F, N]:
                continue  # skip this iteration if any required value is missing

            # Apply classification logic
            if H > C:
                typologies.loc[mask_U, 'value'] = 'URB'
            if P > (1.5 * A):
                typologies.loc[mask_T, 'value'] = 'SCS'
            elif (D > 1) and (I > (0.33 * A)):
                typologies.loc[mask_T, 'value'] = 'LVK'
            elif G > (0.5 * A):
                typologies.loc[mask_T, 'value'] = 'MXG'
            elif (F > (0.25 * A)) and (N > 30):
                typologies.loc[mask_T, 'value'] = 'MXF'
            else:
                typologies.loc[mask_T, 'value'] = 'DSG'

    #%%
    return typologies



"""
Script name: c_yields.py
Description: Calculates and corrects agricultural yields for each territory.
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
import matplotlib.pyplot as plt
import seaborn as sns

#%% Code function
def run_yield():
    print("Running crop yields workflow...")

    #%% --- Import and merge production and surface data ---

    # Load the two DataFrames
    final_production = pd.read_csv('data/outputs/intermediate_datasets/final_production.csv')
    final_areas = pd.read_csv('data/outputs/intermediate_datasets/final_areas.csv')

    # Load region ids
    regions = pd.read_csv('data/regions.csv', sep=';')

    # Define years and crops
    years = np.arange(1990, 2020)
    final_crops = [
        'Wheat', 'Other cereals', 'Grain maize', 'Barley', 'Fodder crops',
        'Oilseeds', 'Potatoes', 'Pulses', 'Sugar beet', 'Temporary grassland',
        'Vegetables and other', 'Forage legumes', 'Olives', 'Grapes',
        'Other permanent crops'
    ]

    # Define the three indicators: code, label, and unit
    indicators = [
        ('A', 'harvested area', 'Mha'),
        ('H', 'harvested quantity', 'Gg N'),
        ('Y', 'yield', 'kg N / ha'),
    ]

    # Build a MultiIndex over region, crop, year, and indicator code
    mi = pd.MultiIndex.from_product(
        [
            sorted(regions['NUTS_ID']),  # all NUTS regions
            sorted(final_crops),         # all crops
            years,                       # all years
            [symbol for symbol, *_ in indicators]  # A, H, Y
        ],
        names=["region", "crop", "year", "symbol"]
    )

    # Convert to a DataFrame
    crop_production = mi.to_frame(index=False)

    # Map code to label and unit
    label_map = {code: label for code, label, unit in indicators}
    unit_map = {code: unit for code, label, unit in indicators}

    crop_production['label'] = crop_production['symbol'].map(label_map)
    crop_production['unit'] = crop_production['symbol'].map(unit_map)
    crop_production['value'] = np.nan
    crop_production['confidence'] = np.nan

    # Add the region name by mapping from regions['name']
    name_map = dict(zip(regions['NUTS_ID'], regions['name']))
    crop_production.insert(
        loc=crop_production.columns.get_loc('region') + 1,
        column='region name',
        value=crop_production['region'].map(name_map)
    )

    # Merge production and surface into the single crop production DataFrame
    final_areas = final_areas.copy()
    final_areas['symbol'] = 'A'
    final_production = final_production.copy()
    final_production['symbol'] = 'H'

    # Concatenate area and production into one DataFrame for merging
    data_to_merge = pd.concat(
        [final_areas[['region', 'crop', 'year', 'symbol', 'value', 'confidence']],
         final_production[['region', 'crop', 'year', 'symbol', 'value', 'confidence']]],
        ignore_index=True
    )

    # Merge crop_production with the data, filling in 'value' and 'confidence'
    crop_production = (
        crop_production
        .drop(columns=['value', 'confidence'])
        .merge(
            data_to_merge,
            on=['region', 'crop', 'year', 'symbol'],
            how='left'
        )
    )

    #%% --- Calculate yield values ---

    # Create a key to identify each (region, crop, year) tuple
    crop_production['_key'] = list(zip(crop_production['region'], crop_production['crop'], crop_production['year']))

    # Build Series for A and H values indexed by that key
    A = crop_production[crop_production['symbol'] == 'A'].set_index('_key')['value']
    H = crop_production[crop_production['symbol'] == 'H'].set_index('_key')['value']

    # For each Y-row, compute value = H / A
    def compute_yield(key):
        a = A.get(key, np.nan)
        h = H.get(key, np.nan)
        if pd.isna(a) or pd.isna(h):
            return np.nan
        if a == 0 and h == 0:
            return 0.0  # Set yield to 0 if both are zero
        if a == 0:
            return np.nan  # avoid division by zero
        return h / a

    mask_Y = crop_production['symbol'] == 'Y'
    crop_production.loc[mask_Y, 'value'] = [
        compute_yield(k) for k in crop_production.loc[mask_Y, '_key']
    ]

    # Assign 'calculated' confidence to valid Yields
    mask_Y_valid = (crop_production['symbol'] == 'Y') & (crop_production['value'].notna())
    crop_production.loc[mask_Y_valid, 'confidence'] = 'calculated'

    # Clean up helper column
    crop_production.drop(columns=['_key'], inplace=True)

    #%% --- Calculate production or surface values when production or surface and yield are available ---

    # Loop over each crop of interest
    for crop in final_crops:
        # Filter yield data for the given crop
        df_crop_yield = crop_production[(crop_production['crop'] == crop) & (crop_production['label'] == 'yield')]

        # Compute mean and standard deviation for yield
        mean_yield = df_crop_yield['value'].mean()
        std_yield = df_crop_yield['value'].std()

        # Define outlier thresholds
        lower_bound = mean_yield - 3 * std_yield
        upper_bound = mean_yield + 3 * std_yield

        # Identify outlier rows
        outlier_indices = df_crop_yield[
            (df_crop_yield['value'] < lower_bound) | (df_crop_yield['value'] > upper_bound)
            ].index

        # Exclude outliers to use for interpolation or averaging
        df_crop_yield_clean = df_crop_yield.drop(outlier_indices)

        # Loop through each outlier to correct it
        for idx in outlier_indices:
            row = df_crop_yield.loc[idx]
            region = row['region']
            year = row['year']
            old_value = row['value']

            # Try to interpolate yield using same crop and region over time
            df_region_yield = df_crop_yield_clean[df_crop_yield_clean['region'] == region].sort_values('year')

            if len(df_region_yield) >= 2:
                interpolated_value = np.interp(year, df_region_yield['year'], df_region_yield['value'])
                method = "interpolated"
            else:
                interpolated_value = df_crop_yield_clean['value'].mean()
                method = "mean crop value"

            # Apply the corrected yield value
            crop_production.loc[idx, 'value'] = interpolated_value
            crop_production.loc[idx, 'confidence'] = 'corrected (f)'

            # Print correction
            print(
                f"[{crop}] {region}-{year}: Yield corrected from {old_value:.2f} to {interpolated_value:.2f} ({method})")

            # Try to correct associated 'H' and 'A' via interpolation only
            for symbol in ['A', 'H']:  # A = harvested area, H = harvested quantity
                condition = (
                        (crop_production['region'] == region) &
                        (crop_production['year'] == year) &
                        (crop_production['crop'] == crop) &
                        (crop_production['symbol'] == symbol)
                )

                # Locate the row
                matching_rows = crop_production[condition]

                if not matching_rows.empty:
                    # Extract all valid rows for this crop/region/symbol (excluding current year)
                    df_symbol_series = crop_production[
                        (crop_production['crop'] == crop) &
                        (crop_production['region'] == region) &
                        (crop_production['symbol'] == symbol) &
                        (crop_production['year'] != year)
                        ].sort_values('year')

                    if len(df_symbol_series) >= 2:
                        # Interpolation is possible
                        interpolated_symbol_value = np.interp(
                            year,
                            df_symbol_series['year'],
                            df_symbol_series['value']
                        )
                        crop_production.loc[condition, 'value'] = interpolated_symbol_value
                        crop_production.loc[condition, 'confidence'] = 'corrected (f)'

                        label_name = matching_rows['label'].values[0]
                        print(f"  ↳ {label_name} corrected by interpolation to {interpolated_symbol_value:.5f}")

                    else:
                        # Cannot interpolate, mark as suspect
                        crop_production.loc[condition, 'confidence'] = 'suspect'

                        label_name = matching_rows['label'].values[0]
                        print(f"  ↳ {label_name} could not be interpolated → flagged as 'suspect'")

    #%%
    return crop_production


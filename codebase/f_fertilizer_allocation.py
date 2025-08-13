"""
Script name: f_fertilizer_allocation.py
Description: Estimates synthetic fertilizer allocation for each territory.
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
import warnings

#%% Code function
def run_fertilizer_allocation():
    print("Running fertilizer allocation workflow...")

    #%% --- Define a new dataset to store fertilizer quantities ---

    # Define the three indicators
    indicators = [
        ('Q', 'total quantity of synthetic N fertilizer applied', 'Gg N'),
        ('Q_AL', 'quantity applied on arable land', 'Gg N'),
        ('Q_PG', 'quantity applied on permanent grassland', 'Gg N'),
        ('Q_PC', 'quantity applied on permanent crops', 'Gg N'),
    ]

    # Define the period
    years = np.arange(1990, 2020)

    # Load region ids
    regions = pd.read_csv('data/regions.csv', sep=';')

    # Build a MultiIndex over region, year, and indicator code
    mi = pd.MultiIndex.from_product(
        [
            sorted(regions['NUTS_ID']),  # all NUTS regions
            years,                       # all years
            [symbol for symbol, *_ in indicators]  # indicator symbols
        ],
        names=["region", "year", "symbol"]
    )

    # Convert to a DataFrame
    fertilizer = mi.to_frame(index=False)

    # Map code to label and unit
    label_map = {code: label for code, label, unit in indicators}
    unit_map = {code: unit for code, label, unit in indicators}

    fertilizer['label'] = fertilizer['symbol'].map(label_map)
    fertilizer['unit'] = fertilizer['symbol'].map(unit_map)
    fertilizer['value'] = np.nan
    fertilizer['confidence'] = np.nan

    # Add the region name by mapping from regions['name'] if available
    name_map = dict(zip(regions['NUTS_ID'], regions['name']))
    fertilizer.insert(
        loc=fertilizer.columns.get_loc('region') + 1,
        column='region name',
        value=fertilizer['region'].map(name_map)
    )

    # Import fertilizer quantities
    final_fertilizer = pd.read_csv('data/outputs/intermediate_datasets/final_fertilizer.csv')

    # Keep only required columns for merging
    data_to_merge = final_fertilizer[['region', 'year', 'symbol', 'value', 'confidence']]

    # Merge into the fertilizer DataFrame
    fertilizer = (
        fertilizer
        .drop(columns=['value', 'confidence'])
        .merge(
            data_to_merge,
            on=['region', 'year', 'symbol'],
            how='left'
        )
    )

    #%% --- Import fertilizer rates ---

    # Load and process EuropeAgriDB
    path_euragri = 'data/EuropeAgriDB_v1.0/tables/synthetic_fertilizer.csv'
    rates = pd.read_csv(path_euragri, sep=';')
    rates_cropland = rates.loc[rates['Symbol'] == 'R_C']
    rates_grassland = rates.loc[rates['Symbol'] == 'R_PG']

    # Set region as index
    rates_cropland = rates_cropland.set_index('Region')
    rates_grassland = rates_grassland.set_index('Region')

    # Define the period
    years = np.arange(1990, 2020)

    # Load permanent/arable fertilizer rates from EFMA
    efma = pd.read_csv('data/literature_various/EFMA_fertlizer_rates.csv', sep=';')

    #%% --- Import and format cropland areas ---

    # Import data
    land_areas = pd.read_csv('data/outputs/land_areas.csv')

    # Cropland
    cropland_area_complete = (
        land_areas
        .loc[land_areas['symbol'] == 'C_sum', ['region', 'year', 'value']]
        .pivot(index='region', columns='year', values='value')
        .sort_index()
    )

    # Arable land
    arable_area_complete = (
        land_areas
        .loc[land_areas['symbol'] == 'AL_sum', ['region', 'year', 'value']]
        .pivot(index='region', columns='year', values='value')
        .sort_index()
    )

    # Arable land
    permanent_area_complete = (
        land_areas
        .loc[land_areas['symbol'] == 'PC_sum', ['region', 'year', 'value']]
        .pivot(index='region', columns='year', values='value')
        .sort_index()
    )

    # Permanent grassland
    pegrass = (
        land_areas
        .loc[land_areas['symbol'] == 'PG', ['region', 'year', 'value']]
        .pivot(index='region', columns='year', values='value')
        .sort_index()
    )

    #%% --- Calculate RATES ratios ---

    # Create empty DataFrames to store fertilizer ratios
    Rpg_Rc = pd.DataFrame(index=regions['NUTS_ID'].copy())
    Rpc_Ral = pd.DataFrame(index=regions['NUTS_ID'].copy())

    # Loop over each year
    for year in years:
        ratios_pg = []
        ratios_pc = []

        for region in regions['NUTS_ID']:
            country = region[:2]  # Extract 2-letter country code

            # --- Grassland rate ---
            pg_values = rates_grassland.loc[
                (rates_grassland.index == country) &
                (rates_grassland['Year'] == year),
                'Value'
            ].values

            # --- Cropland rate ---
            c_values = rates_cropland.loc[
                (rates_cropland.index == country) &
                (rates_cropland['Year'] == year),
                'Value'
            ].values

            # Extract values or assign NaN
            pg_val = pg_values[0] if len(pg_values) > 0 else np.nan
            c_val = c_values[0] if len(c_values) > 0 else np.nan

            # --- EFMA values for permanent/arable application rates ---
            efma_row = efma[efma['region'] == country]

            if not efma_row.empty:
                try:
                    al_val = efma_row['Arable land'].values[0]
                    pc_val = efma_row['Perm. crops'].values[0]
                except KeyError:
                    al_val = np.nan
                    pc_val = np.nan
            else:
                al_val = np.nan
                pc_val = np.nan

            # --- Compute ratios ---
            ratio_pg = pg_val / c_val if pd.notna(pg_val) and pd.notna(c_val) and c_val != 0 else np.nan
            ratio_pc = pc_val / al_val if pd.notna(pc_val) and pd.notna(al_val) and al_val != 0 else np.nan

            ratios_pg.append(ratio_pg)
            ratios_pc.append(ratio_pc)

        # Store results per year
        Rpg_Rc[year] = ratios_pg
        Rpc_Ral[year] = ratios_pc

    # Set index names
    Rpg_Rc.index.name = 'region'
    Rpc_Ral.index.name = 'region'

    # Fill CH and LI countries by 0.15
    Rpg_Rc.loc['CH'] = 0.15
    Rpg_Rc.loc['LI'] = 0.15

    # Fill NaNs in Rpc_Ral with the column (year) mean across all regions
    Rpc_Ral = Rpc_Ral.apply(lambda col: col.fillna(col.mean()), axis=0)


    # Apply linear interpolation across years for each region
    Rpg_Rc = (
        Rpg_Rc.interpolate(axis=1, method='linear', limit_direction='both')
    )

    # Fill NaN regions with 0
    Rpg_Rc = Rpg_Rc.fillna(0)

    #%% --- Calculate AREAS ratios ---

    # Create an empty DataFrame to store fertilizer application areas ratios
    Apg_Ac = pd.DataFrame(index=regions['NUTS_ID'])
    Apc_Aal = pd.DataFrame(index=regions['NUTS_ID'])

    # Loop over each year to compute and store the PG/C ratio for each region
    for year in years:
        ratios_pg = []
        ratios_pc = []

        for region in regions['NUTS_ID']:

            # Get areas for the region and year
            pg_values = pegrass.loc[(region, year)]
            c_values = cropland_area_complete.loc[(region, year)]
            al_values = arable_area_complete.loc[(region, year)]
            pc_values = permanent_area_complete.loc[(region, year)]

            # Compute the PG ratio safely
            if pd.notna(pg_values) and pd.notna(c_values) and c_values != 0 and pg_values != 0:
                ratio_pg = pg_values / c_values
            else:
                ratio_pg = np.nan
            ratios_pg.append(ratio_pg)

            # Compute the PC ratio safely
            if pd.notna(al_values) and pd.notna(pc_values) and al_values != 0 and pc_values != 0:
                ratio_pc = pc_values / al_values
            else:
                ratio_pc = np.nan
            ratios_pc.append(ratio_pc)

        # Store the results in the DataFrame
        Apg_Ac[year] = ratios_pg
        Apc_Aal[year] = ratios_pc

    # Set index name
    Apg_Ac.index.name = 'region'
    Apc_Aal.index.name = 'region'

    # Apply linear interpolation/extrapolation across years for each region
    Apg_Ac = (Apg_Ac.interpolate(axis=1, method='linear', limit_direction='both'))
    Apc_Aal = (Apc_Aal.interpolate(axis=1, method='linear', limit_direction='both'))

    # Fill NaN regions with 0
    Apc_Aal = Apc_Aal.fillna(0)

    #%% --- Calculate fertilizer quantity applied in permanent grassland ---

    # Loop over each year and region
    for year in years:
        for region in regions['NUTS_ID']:

            # Select total fertilizer input (symbol 'Q') for the given year and region
            Q_series = fertilizer.loc[
                (fertilizer['year'] == year) &
                (fertilizer['region'] == region) &
                (fertilizer['symbol'] == 'Q'),
                'value'
            ]

            # Skip if no 'Q' value is available for this region and year
            if Q_series.empty:
                continue

            # Extract scalar value from the Series
            Q_value = Q_series.values[0]

            # Compute the cropland fraction
            frac_cropland = (1 + Rpg_Rc.loc[region, year] * Apg_Ac.loc[region, year])**(-1)

            # Compute the amount of fertilizer attributed to permanent grassland (Q_PG)
            Q_PG = Q_value * (1 - frac_cropland)

            # Assign the computed value to the row with symbol 'Q_PG'
            fertilizer.loc[
                (fertilizer['year'] == year) &
                (fertilizer['region'] == region) &
                (fertilizer['symbol'] == 'Q_PG'),
                ['value', 'confidence']
            ] = [Q_PG, 'Estimated']

            # Compute the quantity applied to cropland
            Q_C = Q_value - Q_PG

            # Compute the arable fraction
            frac_arable = (1 + Rpc_Ral.loc[region, year] * Apc_Aal.loc[region, year])**(-1)

            # Compute the amount of fertilizer attributed to permanent crops (Q_PC)
            Q_PC = Q_C * (1 - frac_arable)

            # Assign the computed value to the row with symbol 'Q_PC'
            fertilizer.loc[
                (fertilizer['year'] == year) &
                (fertilizer['region'] == region) &
                (fertilizer['symbol'] == 'Q_PC'),
                ['value', 'confidence']
            ] = [Q_PC, 'Estimated']

            # Compute the amount of fertilizer attributed to arable land (Q_AL)
            Q_AL = Q_C - Q_PC

            # Assign the computed value to the row with symbol 'Q_AL'
            fertilizer.loc[
                (fertilizer['year'] == year) &
                (fertilizer['region'] == region) &
                (fertilizer['symbol'] == 'Q_AL'),
                ['value', 'confidence']
            ] = [Q_AL, 'Estimated']

    #%% --- Correction outliers procedure (max 400 kgN/ha/year) ---

    # Import land use data
    land_use = pd.read_csv('data/outputs/land_areas.csv')

    # Mapping between fertilizer symbols and corresponding land use symbols
    symbol_surface_map = {
        'Q_AL': 'AL_sum',
        'Q_PG': 'PG',
        'Q_PC': 'PC_sum'
    }

    # Store (region, year) combinations where partial corrections occurred
    corrections_q = set()

    # Loop over each fertilizer symbol (partial applications)
    for fert_symbol, surf_symbol in symbol_surface_map.items():
        # Merge fertilizer data with corresponding land use (area in Mha)
        merged = fertilizer[fertilizer['symbol'] == fert_symbol].merge(
            land_use[land_use['symbol'] == surf_symbol][['region', 'year', 'value']],
            on=['region', 'year'],
            suffixes=('', '_area')
        )

        # Compute fertilization rate (Gg N / Mha = kg N / ha)
        merged['rate'] = merged['value'] / merged['value_area']

        # Identify outliers where rate > 400 kg N/ha
        over_limit = merged[merged['rate'] > 400]

        # Loop over each outlier and apply correction
        for _, row in over_limit.iterrows():
            region, year = row['region'], row['year']
            area = row['value_area']
            old_value = row['value']
            corrected_value = 400 * area  # Gg N

            # Update value and confidence flag in original DataFrame
            condition = (
                    (fertilizer['region'] == region) &
                    (fertilizer['year'] == year) &
                    (fertilizer['symbol'] == fert_symbol)
            )
            fertilizer.loc[condition, 'value'] = corrected_value
            fertilizer.loc[condition, 'confidence'] = 'corrected (f)'

            # Print correction details
            print(f"[{region}-{year}] {fert_symbol}: corrected from {old_value:.4f} to {corrected_value:.4f} Gg N")

            # Mark this (region, year) for total Q correction later
            corrections_q.add((region, year))

    # Update total synthetic N fertilizer applied (symbol == 'Q') for affected (region, year)
    for region, year in corrections_q:
        # Get corrected values of Q_AL, Q_PG, Q_PC
        q_parts = fertilizer[
            (fertilizer['region'] == region) &
            (fertilizer['year'] == year) &
            (fertilizer['symbol'].isin(['Q_AL', 'Q_PG', 'Q_PC']))
            ]['value']

        if len(q_parts) == 3:
            total_corrected = q_parts.sum()

            # Update Q value and flag
            condition = (
                    (fertilizer['region'] == region) &
                    (fertilizer['year'] == year) &
                    (fertilizer['symbol'] == 'Q')
            )
            fertilizer.loc[condition, 'value'] = total_corrected
            fertilizer.loc[condition, 'confidence'] = 'corrected (f)'

            # Print Q correction details
            print(f"[{region}-{year}] Q: updated to sum of components = {total_corrected:.4f} Gg N")

    #%%
    return fertilizer



"""
Script name: d_land_use.py
Description: Cleans and reformats agricultural land use for each territory.
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

#%% --- Code function ---
def run_land_use():
    print("Running land use workflow...")

    # %% --- Define a new dataset to store animal excretions ---

    # Import regions and years
    regions = pd.read_csv('data/regions.csv', sep=';')
    years = np.arange(1990, 2020)

    symbols = [
        'C_sum', 'AL_sum', 'PC_sum', 'TG', 'PG'
    ]

    label_map = {
        'C_sum': 'crop area sum',
        'AL_sum': 'arable land area sum',
        'PC_sum': 'permanent crop area sum',
        'TG': 'temporary grassland area',
        'PG': 'permanent grassland area',
    }

    # Cartesian product
    final_land_use = pd.MultiIndex.from_product(
        [sorted(regions['NUTS_ID']), symbols, sorted(years)],
        names=["region", "symbol", "year"]
    ).to_frame(index=False)

    # Region names
    region_name_map = regions.set_index('NUTS_ID')['name']
    final_land_use.insert(
        loc=1,
        column='region_name',
        value=final_land_use['region'].map(region_name_map)
    )

    # Metadata columns + labels
    final_land_use = final_land_use.assign(
        value = np.nan,
        label = final_land_use['symbol'].map(label_map),
        unit = "Mha",
        confidence = np.nan
    )

    # Enforce the per-(region, year) symbol ordering
    final_land_use['symbol'] = pd.Categorical(
        final_land_use['symbol'],
        categories=symbols,
        ordered=True
    )

    final_land_use = (
        final_land_use
        .sort_values(['region', 'year', 'symbol'])
        .reset_index(drop=True)
    )

    #%% --- Load data ---

    path_areas = 'data/outputs/crop_production_all_categories.csv'
    areas = pd.read_csv(path_areas)
    areas = areas.loc[areas['symbol'] == 'A']  # Only area data

    # Group by region and year
    grouped = areas.groupby(['region', 'year'])

    #%% --- Define crop category areas ---

    arable_crops = [
        'Wheat', 'Other cereals', 'Grain maize', 'Barley', 'Fodder crops',
        'Oilseeds', 'Potatoes', 'Pulses', 'Sugar beet',
        'Temporary grassland', 'Vegetables and other', 'Forage legumes'
    ]

    permanent_crops = [
        'Olives', 'Grapes', 'Other permanent crops'
    ]

    #%% --- Temporary grassland areas ---

    # Filter data for 'Temporary grassland'
    tg_data = areas.loc[areas['crop'] == 'Temporary grassland']

    # Group by region and year, sum the area and retain confidence
    tg_summary = (
        tg_data
        .groupby(['region', 'year'])
        .agg(
            tg_area=('value', 'sum'),
            tg_confidence=('confidence', 'first')  # or use 'mode', 'max', etc. as appropriate
        )
        .reset_index()
    )

    # Create a mask for the 'TG' rows in final_land_use
    mask = final_land_use['symbol'] == 'TG'
    tg_rows = final_land_use[mask]

    # Merge to bring in both the aggregated area and confidence
    merged = tg_rows.merge(
        tg_summary,
        on=['region', 'year'],
        how='left'
    )

    # Update the 'value' column with the aggregated area
    final_land_use.loc[mask, 'value'] = merged['tg_area'].values

    # Update the 'confidence' column with the extracted confidence
    final_land_use.loc[mask, 'confidence'] = merged['tg_confidence'].values

    # Correct AL, CH and NO, TG from Eurostat
    for year in years:
        final_land_use.loc[
            (mask) & (final_land_use['region'] == 'AL') & (final_land_use['year'] == year), 'value'] = 0.1437
        final_land_use.loc[
            (mask) & (final_land_use['region'] == 'AL') & (final_land_use['year'] == year), 'confidence'] = 'interpolated'
        final_land_use.loc[
            (mask) & (final_land_use['region'] == 'CH') & (final_land_use['year'] == year), 'value'] = 0.125
        final_land_use.loc[
            (mask) & (final_land_use['region'] == 'CH') & (final_land_use['year'] == year), 'confidence'] = 'interpolated'
        final_land_use.loc[
            (mask) & (final_land_use['region'] == 'NO') & (final_land_use['year'] == year), 'value'] = 0.66
        final_land_use.loc[
            (mask) & (final_land_use['region'] == 'NO') & (final_land_use['year'] == year), 'confidence'] = 'interpolated'

    #%% --- Permanent grassland areas ---

    # Load Eurostat dataset
    path_pegrass = 'data/Eurostat/surface/ef_lus_pegrass__custom_15260684_spreadsheet.xlsx'
    pegrass_data = pd.read_excel(path_pegrass, sheet_name='Sheet 1', header=None, na_values=':')

    # Rename columns: region and selected years
    pegrass_data = pegrass_data.rename(columns={
        0: 'region',
        2: 2010,
        3: 2013,
        4: 2016,
        5: 2020
    })
    pegrass_data['region'] = pegrass_data['region'].fillna('')

    # List of years to process
    years_pegrass = [2010, 2013, 2016, 2020]

    # Initialize output DataFrame with years as index
    pegrass = pd.DataFrame(index=years_pegrass)

    # Loop over all target NUTS regions
    for region in regions['NUTS_ID']:
        data = []

        for year in years_pegrass:
            if region in pegrass_data['region'].values:
                # Case 1: exact match found in dataset
                value = pegrass_data.loc[pegrass_data['region'] == region, year].values
                data.append(value[0] if len(value) > 0 else np.nan)
            elif pegrass_data['region'].str.startswith(region).any():
                # Case 2: sum over all subregions matching the prefix
                subset = pegrass_data[pegrass_data['region'].str.startswith(region)]
                data.append(np.nansum(subset[year].values))
            else:
                # Case 3: region not found, assign NaN
                data.append(np.nan)

        # Convert to a NumPy array (or pandas Series) before dividing
        arr = np.array(data, dtype=float)  # stays NaN if all entries are NaN
        pegrass[region] = arr / 1000000

    # Calculate permanent grassland mean areas
    pegrass.replace(0, np.nan, inplace=True)
    pegrass.loc['mean'] = np.nanmean(pegrass.values, axis=0)

    # Assign a value in Albania (from Billen et al., 2024)
    pegrass.loc['mean', 'AL'] = 0.4781
    pegrass.loc['mean', 'MT'] = 0

    # Compute regional coefficients of permanent grassland share relative to country total
    region_codes = pegrass.columns  # exclude the 'mean' row
    coefficients = pd.Series(index=region_codes, dtype=float)

    # Step 1: compute raw regional share of national mean
    for region in region_codes:
        country_code = region[:2]
        # get all regions belonging to this country
        country_regions = [r for r in region_codes if r.startswith(country_code)]

        if region == country_code:
            # region is a country: assign coefficient = 1
            coefficients[region] = 1.0
        else:
            total_country_mean = pegrass.loc['mean', country_regions].sum()
            region_mean = pegrass.loc['mean', region]
            coefficients[region] = region_mean / total_country_mean if total_country_mean != 0 else 0

    # Step 2: normalize coefficients so that their sum per country equals 1
    normalized_coefficients = coefficients.copy()

    for region in region_codes:
        country_code = region[:2]
        country_regions = [r for r in region_codes if r.startswith(country_code)]
        country_sum = coefficients[country_regions].sum()
        if country_sum != 0:
            normalized_coefficients[region] = coefficients[region] / country_sum
        else:
            normalized_coefficients[region] = 0  # fallback

    # Store normalized coefficients in a new DataFrame
    pegrass_coeff = normalized_coefficients.to_frame(name='normalized_coefficient')


    # Import Euragri data
    euragri = pd.read_csv('data/EuropeAgriDB_v1.0/tables/land_areas.csv', sep=';')
    euragri = euragri[euragri['Symbol'] == 'PG']

    # Filter for PG rows
    pg_rows = final_land_use[final_land_use['symbol'] == 'PG'].copy()

    # Loop through regions and years
    for region in regions['NUTS_ID']:
        country = region[:2]

        for year in years:
            mask = (pg_rows['region'] == region) & (pg_rows['year'] == year)

            # Check if Euragri has value for this country and year
            match = euragri[(euragri['Region'] == country) & (euragri['Year'] == year)]

            if not match.empty:
                value = match['Value'].values[0] * pegrass_coeff.loc[region, 'normalized_coefficient']
                pg_rows.loc[mask, 'value'] = value
                pg_rows.loc[mask, 'confidence'] = 'EuropeAgri'

    r = ['AL', 'CH', 'ME', 'MK', 'MT', 'NO', 'RS']
    # Replace missing values from Eurostat
    for region in r:
        for year in years:
            mask = (pg_rows['region'] == region) & (pg_rows['year'] == year)
            value = pegrass.loc['mean', region]
            pg_rows.loc[mask, 'value'] = value
            pg_rows.loc[mask, 'confidence'] = 'Eurostat'

    # Replace the original PG rows in final_land_use
    final_land_use.update(pg_rows)

    #%% --- Arable land area sum ---

    # Define conditional sum function
    def conditional_sum(df, crop_names):
        return df.loc[df['crop'].isin(crop_names), 'value'].sum()

    # Keep only relevant crops
    filtered_areas = areas[areas['crop'].isin(arable_crops)]

    # Group by region and year and compute total arable area
    arable_area = (
        filtered_areas
        .groupby(['region', 'year'])
        .apply(lambda g: conditional_sum(g, arable_crops))
        .reset_index(name='arable_area')
    )

    # Create a mask for the 'AL_sum' rows in final_land_use
    mask = final_land_use['symbol'] == 'AL_sum'
    al_rows = final_land_use[mask]

    # Merge to bring in both the aggregated area and confidence
    merged = al_rows.merge(
        arable_area,
        on=['region', 'year'],
        how='left'
    )

    # Update the 'value' column with the aggregated area
    final_land_use.loc[mask, 'value'] = merged['arable_area'].values

    # Update the 'confidence' column with the extracted confidence
    final_land_use.loc[mask, 'confidence'] = 'calculated'

    #%% --- Permanent crop areas sum ---

    # Keep only relevant crops
    filtered_areas = areas[areas['crop'].isin(permanent_crops)]

    # Group by region and year and compute total arable area
    pc_area = (
        filtered_areas
        .groupby(['region', 'year'])
        .apply(lambda g: conditional_sum(g, permanent_crops))
        .reset_index(name='perm_area')
    )

    # Create a mask for the 'PC_sum' rows in final_land_use
    mask = final_land_use['symbol'] == 'PC_sum'
    pc_rows = final_land_use[mask]

    # Merge to bring in both the aggregated area and confidence
    merged = pc_rows.merge(
        pc_area,
        on=['region', 'year'],
        how='left'
    )

    # Update the 'value' column with the aggregated area
    final_land_use.loc[mask, 'value'] = merged['perm_area'].values

    # Update the 'confidence' column with the extracted confidence
    final_land_use.loc[mask, 'confidence'] = 'calculated'

    #%% --- Cropland in use ---

    for year in years:
        for region in regions['NUTS_ID']:

            AL = final_land_use.loc[
                (final_land_use['symbol'] == 'AL_sum') &
                (final_land_use['region'] == region) &
                (final_land_use['year'] == year),
                'value'].values[0]

            PC = final_land_use.loc[
                (final_land_use['symbol'] == 'PC_sum') &
                (final_land_use['region'] == region) &
                (final_land_use['year'] == year),
                'value'].values[0]

            PG = final_land_use.loc[
                (final_land_use['symbol'] == 'PG') &
                (final_land_use['region'] == region) &
                (final_land_use['year'] == year),
                'value'].values[0]

            value = np.nansum([AL, PC, PG])

            mask = (
                (final_land_use['symbol'] == 'C_sum') &
                (final_land_use['region'] == region) &
                (final_land_use['year'] == year)
            )

            final_land_use.loc[mask, 'value'] = value
            final_land_use.loc[mask, 'confidence'] = 'calculated'

    # %% --- Linear interpolation & extrapolation per region/symbol ---

    zero_mask = final_land_use['value'] == 0
    final_land_use.loc[zero_mask, 'confidence'] = np.nan
    final_land_use.loc[zero_mask, 'value'] = np.nan

    final_land_use = final_land_use.sort_values(['region', 'symbol', 'year'])

    def interp_and_flag(group):
        was_na = group['value'].isna()
        group['value'] = group['value'].interpolate(method='linear', limit_direction='both')
        filled = was_na & group['value'].notna()
        if filled.any():
            group.loc[filled, 'confidence'] = 'interpolated'
        return group

    final_land_use = (
        final_land_use
        .groupby(['region', 'symbol'], group_keys=False)
        .apply(interp_and_flag)
    )

    # %% --- Restore original ordering (region, year, symbol) ---
    final_land_use = final_land_use.sort_values(['region', 'year', 'symbol']).reset_index(drop=True)

    #%% --- Quality control Arable land areas ---

    # 1. Filtrer uniquement les données de type 'yield'
    df = final_land_use[final_land_use['symbol'] == 'AL_sum'].copy()

    # 2. Trier par région, culture et année
    df = df.sort_values(['region', 'year'])

    # 3. Calculer les valeurs des années précédente et suivante
    df['prev_value'] = df.groupby(['region'])['value'].shift(1)
    df['next_value'] = df.groupby(['region'])['value'].shift(-1)

    # 4. Fonction de changement relatif
    def rel_change(current, other):
        if pd.isna(other) or other == 0:
            return np.nan
        return (current - other) / other

    # 5. Calcul des changements relatifs
    df['change_prev'] = df.apply(
        lambda row: rel_change(row['value'], row['prev_value']), axis=1
    )
    df['change_next'] = df.apply(
        lambda row: rel_change(row['value'], row['next_value']), axis=1
    )

    # 6. Détection des anomalies : changement >250% par rapport à avant **et** après
    anomalies_yield = df[
        (df['change_prev'].abs() > 2.5) |
        (df['change_next'].abs() > 2.5)
        ]

    # 7. Affichage des anomalies détectées
    print("=== Yield values with >250% deviation compared to both previous and next year ===\n")
    print(
        anomalies_yield[
            [
                'region',
                'year',
                'value',
                'prev_value',
                'change_prev',
                'next_value',
                'change_next'
            ]
        ]
        .sort_values(['region', 'year'])
        .to_string(index=False)
    )

    #%% Correct ITC arable land value in 2014

    # List of symbols to correct for region 'ITC' in year 2014
    symbols_to_correct = ['AL_sum', 'C_sum']
    target_region = 'ITC'
    target_year = 2014

    for symbol in symbols_to_correct:
        # Define condition for the symbol and region
        condition = (final_land_use['symbol'] == symbol) & (final_land_use['region'] == target_region)

        # Get values for years before and after the target year
        val_before = final_land_use.loc[condition & (final_land_use['year'] == target_year - 1), 'value']
        val_after = final_land_use.loc[condition & (final_land_use['year'] == target_year + 1), 'value']

        # If both values exist, compute the average and update the 2014 row
        if not val_before.empty and not val_after.empty:
            mean_val = np.mean([val_before.values[0], val_after.values[0]])
            update_mask = condition & (final_land_use['year'] == target_year)
            final_land_use.loc[update_mask, 'value'] = mean_val
            final_land_use.loc[update_mask, 'confidence'] = 'corrected'

    #%%
    return final_land_use

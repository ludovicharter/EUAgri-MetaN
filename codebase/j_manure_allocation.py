"""
Script name: j_manure_allocation.py
Description: Estimates manure allocation for each territory.
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
import glob
import os

#%% Code function
def run_manure():
    print("Running manure allocation workflow...")

    #%% --- Import data ---

    # Excretion
    path_excretion = 'data/outputs/animal_excretion.csv'
    excretion = pd.read_csv(path_excretion)

    # Regions
    regions = pd.read_csv('data/regions.csv', sep=';')

    # Define years
    years = np.arange(1990, 2020)

    #%% --- (1) Fraction of ruminant manure excreted outdoor ---

    # Define countries for UNFCCC
    countries = ['AUT', 'BEL', 'BGR', 'HRV', 'CYP', 'CZE', 'DNM', 'EST', 'FIN', 'FRK', 'DEU', 'HUN', 'GRC', 'IRL', 'ITA',
                 'LVA', 'MLT', 'NLD', 'NOR', 'POL', 'PRT', 'ROU', 'SVK', 'SVN', 'ESP', 'SWE', 'CHE', 'GBR']
    country_codes = ['AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR', 'DE', 'HU', 'EL', 'IE', 'IT', 'LV', 'MT',
                     'NL', 'NO', 'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE', 'CH', 'UK']

    # Define a dict to store data
    fractions = {}
    fractions['frac_dairy_cows'] = pd.DataFrame(index=years)
    fractions['frac_other_cows'] = pd.DataFrame(index=years)
    fractions['frac_total_cows'] = pd.DataFrame(index=years)
    fractions['frac_sheep'] = pd.DataFrame(index=years)
    fractions['frac_goats'] = pd.DataFrame(index=years)

    # Loop through each country
    for country, code in zip(countries, country_codes):

        folder_path = f'data/UNFCCC_inventories/{country}-2023-crf'

        # Initialize lists to store results
        frac_dairy_cows = []
        frac_other_cows = []
        frac_total_cows = []
        frac_sheep = []
        frac_goats = []

        for year in years:
            print(f'Processing UNFCCC for {country}, {year} ...')

            file_path = glob.glob(os.path.join(folder_path, f'{country}_2023_{year}_*.xlsx'))[0]
            # Read data
            data = pd.read_excel(file_path, sheet_name='Table3.B(b)', na_values=['NO'])
            data = data.rename(columns={'TABLE 3.B(b) SECTORAL BACKGROUND DATA FOR AGRICULTURE': 0})

            # Calculate fractions of outdoor excretion for sheep
            if not pd.isna(data.loc[data[0] == '2.  Sheep', 'Unnamed: 8'].values[0]):
                frac_sheep.append(data.loc[data[0] == '2.  Sheep', 'Unnamed: 8'].values[0] / data.loc[data[0] == '2.  Sheep', 'Unnamed: 13'].values[0])
            else:
                frac_sheep.append(np.nan)

            # Calculate fractions of outdoor excretion for goats
            if not pd.isna(data.loc[data[0] == 'Goats', 'Unnamed: 8'].values[0]):
                frac_goats.append(data.loc[data[0] == 'Goats', 'Unnamed: 8'].values[0] / data.loc[data[0] == 'Goats', 'Unnamed: 13'].values[0])
            else:
                frac_goats.append(np.nan)

            # Check option A data
            if not pd.isna(data.loc[data[0] == 'Dairy cattle(4)', 'Unnamed: 8'].values[0]) and not pd.isna(data.loc[data[0] == 'Non-dairy cattle', 'Unnamed: 8'].values[0]):

                # Calculate fractions of outdoor excretion for dairy and other cows
                frac_dairy_cows.append(data.loc[data[0] == 'Dairy cattle(4)', 'Unnamed: 8'].values[0] / data.loc[data[0] == 'Dairy cattle(4)', 'Unnamed: 13'].values[0])
                frac_other_cows.append(data.loc[data[0] == 'Non-dairy cattle', 'Unnamed: 8'].values[0] / data.loc[data[0] == 'Non-dairy cattle', 'Unnamed: 13'].values[0])
                frac_total_cows.append(np.nan)

            # Check option B data
            elif not pd.isna(data.loc[data[0] == 'Mature dairy cattle', 'Unnamed: 8'].values[0]) and not pd.isna(data.loc[data[0] == 'Other mature cattle', 'Unnamed: 8'].values[0]):

                # Calculate fractions of outdoor excretion for dairy and other cows
                dairy_cows_pasture = np.nansum([data.loc[data[0] == 'Mature dairy cattle', 'Unnamed: 8'].values[0], data.loc[data[0] == 'Growing cattle', 'Unnamed: 8'].values[0]])
                dairy_cows_total = np.nansum([data.loc[data[0] == 'Mature dairy cattle', 'Unnamed: 13'].values[0], data.loc[data[0] == 'Mature dairy cattle', 'Unnamed: 13'].values[0]])

                frac_dairy_cows.append(dairy_cows_pasture / dairy_cows_total)
                frac_other_cows.append(data.loc[data[0] == 'Other mature cattle', 'Unnamed: 8'].values[0] / data.loc[data[0] == 'Other mature cattle', 'Unnamed: 13'].values[0])
                frac_total_cows.append(np.nan)

            # Check total cattle data
            elif not pd.isna(data.loc[data[0] == '1.  Cattle', 'Unnamed: 8'].values[0]):

                # Calculate fractions of outdoor excretion for total cows
                frac_dairy_cows.append(np.nan)
                frac_other_cows.append(np.nan)
                frac_total_cows.append(data.loc[data[0] == '1.  Cattle', 'Unnamed: 8'].values[0] / data.loc[data[0] == '1.  Cattle', 'Unnamed: 13'].values[0])

            else:
                frac_dairy_cows.append(np.nan)
                frac_other_cows.append(np.nan)
                frac_total_cows.append(np.nan)

        fractions['frac_dairy_cows'][code] = frac_dairy_cows
        fractions['frac_other_cows'][code] = frac_other_cows
        fractions['frac_total_cows'][code] = frac_total_cows
        fractions['frac_sheep'][code] = frac_sheep
        fractions['frac_goats'][code] = frac_goats

    #%% --- (2) & (5) Fraction of ruminant manure excreted at grazing between permanent and temporary grassland ---

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

        # Temporary grassland
        tg_areas = (
            land_areas
            .loc[land_areas['symbol'] == 'TG', ['region', 'year', 'value']]
            .pivot(index='region', columns='year', values='value')
            .sort_index()
        )

    #%% --- (3) Losses from houses and storages ---

    # Import data from EuragriDB
    losses_path = 'data/EuropeAgriDB_v1.0/tables/mms_loss_share.csv'
    losses = pd.read_csv(losses_path, sep=';')

    losses_storage = pd.DataFrame(index=years)
    # Process each region
    for region in regions['NUTS_ID']:

        country = region[:2]

        if not losses[losses['Region'] == country].empty:
            losses_storage[region] = losses.loc[losses['Region'] == country, 'Value'].values[0]

        # Use the EU average if no data in EurAgriDB
        else:
            losses_storage[region] = losses['Value'].mean()

    #%% --- (4) Stored manure destination shares ---

    # Import data from EurAgriDB
    path_destination = 'data/EuropeAgriDB_v1.0/tables/stored_manure_destination_shares.csv'
    destination = pd.read_csv(path_destination, sep=';')

    # Identify rows for our three target animals on grassland
    target_animals = ['Pigs', 'Poultry and rabbits', 'Ruminants and equines']
    destination['is_target_grass'] = (
        destination['Destination_simple'].eq('Grassland') &
        destination['Animal_simple'].isin(target_animals)
    )

    # Group by Region to compute sums
    grp = destination.groupby('Region')

    # Numerator: sum of Value for target animals on grassland in each Region
    num = grp.apply(lambda x: x.loc[x['is_target_grass'], 'Value'].sum()) \
             .rename('sum_target_grass')

    # Denominator: total Value for all destinations in each Region
    den = grp['Value'].sum().rename('sum_all')

    # Build a DataFrame of ratios
    ratios = pd.concat([num, den], axis=1).reset_index()
    ratios['grass_animals_ratio'] = ratios['sum_target_grass'] / ratios['sum_all']

    # Build a regional DataFrame
    stored_manure_to_grassland = regions[['NUTS_ID']]

    # Extract the country code (first two letters of the NUTS_ID)
    stored_manure_to_grassland['country'] = stored_manure_to_grassland['NUTS_ID'].str[:2]
    country_ratios = ratios[['Region', 'grass_animals_ratio']] \
                        .rename(columns={'Region': 'country'})

    # Merge the country_ratio onto every sub-national region
    stored_manure_to_grassland = stored_manure_to_grassland.merge(
        country_ratios,
        on='country',
        how='left'
    )

    # Calculate A_G/A_C

    # Create empty DataFrames for A_G and A_C
    AG = pd.DataFrame(index=tg_areas.index, columns=years, dtype=float)
    AC = pd.DataFrame(index=tg_areas.index, columns=years, dtype=float)

    # Loop through each region and year
    for region in AG.index:
        for year in years:
            # Get value from tg_areas for this region and year
            tg_val = tg_areas.at[region, year] if year in tg_areas.columns else 0.0

            # Get pasture excretion mean from pegrass['mean'] row
            pe_val = pegrass.loc['mean', region] if region in pegrass.columns else 0.0

            # Compute A_G = tg_area + pegrass mean
            AG.at[region, year] = tg_val + pe_val

            # Get cropland area for this region and year
            try:
                total_cropland = cropland_area_complete.at[region, year]
            except KeyError:
                total_cropland = 0.0

            # Compute A_C = cropland - A_G
            AC.at[region, year] = total_cropland + pe_val

    # Compute the ratio A_G / A_C
    AG_AC = AG / AC

    #%% --- Compile all coefficients for manure allocation ---

    # Initiate a new dict
    manure_allocation = {}
    for year in years:
        manure_allocation[year] = pd.DataFrame(index=regions['NUTS_ID'])

    # 1
    for year in years:

        values_dairy_cows = []
        values_other_cows = []
        values_sheep = []
        values_goats = []

        for region in regions['NUTS_ID']:
            country = region[:2]

            if country in fractions['frac_dairy_cows'].columns:

                if pd.notna(fractions['frac_dairy_cows'].loc[(year, country)]):
                    values_dairy_cows.append(fractions['frac_dairy_cows'].loc[(year, country)])
                    values_other_cows.append(fractions['frac_other_cows'].loc[(year, country)])
                elif pd.notna(fractions['frac_total_cows'].loc[(year, country)]):
                    values_dairy_cows.append(fractions['frac_total_cows'].loc[(year, country)])
                    values_other_cows.append(fractions['frac_total_cows'].loc[(year, country)])
                else:
                    values_dairy_cows.append(0)
                    values_other_cows.append(0)

                values_sheep.append(fractions['frac_sheep'].loc[(year, country)])
                values_goats.append(fractions['frac_goats'].loc[(year, country)])

            else:
                values_dairy_cows.append(0)
                values_other_cows.append(0)
                values_sheep.append(0)
                values_goats.append(0)

        manure_allocation[year]['1:dairy'] = values_dairy_cows
        manure_allocation[year]['1:bovine'] = values_other_cows
        manure_allocation[year]['1:sheep'] = values_sheep
        manure_allocation[year]['1:goats'] = values_goats

        # Correct some values (Billen et al., 2024)
        r = ['AL', 'CY', 'LT', 'LU', 'ME', 'MK']
        v = [0.796, 0.338, 0.177, 0.238, 0.229, 0.362]
        for i, region in enumerate(r):
            manure_allocation[year].loc[region, '1:dairy'] = v[i]
            manure_allocation[year].loc[region, '1:bovine'] = v[i]
            manure_allocation[year].loc[region, '1:sheep'] = v[i]
            manure_allocation[year].loc[region, '1:goats'] = v[i]

    # 2
    for year in years:
        values = []

        for region in regions['NUTS_ID']:
            PG = pegrass.loc[(region, year)]
            TG = tg_areas.loc[(region, year)]
            values.append(PG / (TG + PG))

        manure_allocation[year]['2: A_PG/A_G'] = values

    # 3
    for year in years:
        values = []

        for region in regions['NUTS_ID']:
            values.append(losses_storage.loc[(year, region)])

        manure_allocation[year]['3: losses at storage'] = values

    # 4
    for year in years:
        manure_allocation[year]['4: grassland storage destination'] = stored_manure_to_grassland['grass_animals_ratio'].values
        manure_allocation[year]['4: A_G/A'] = AG_AC[year]

        # Correct some values from Billen et al., 2024
        r = ['AL', 'CH', 'CY', 'ME', 'MK', 'NO']
        v = [0.1, 0.357, 0.06, 0.1, 0.1, 0.357]
        for i, region in enumerate(r):
            manure_allocation[year].loc[region, '4: grassland storage destination'] = v[i]

    # 5
    for year in years:
        values = []

        for region in regions['NUTS_ID']:
            PC = permanent_area_complete.loc[(region, year)]
            AC = arable_area_complete.loc[(region, year)]

            den = AC + PC
            if den != 0:
                values.append(PC / den)
            else:
                values.append(np.nan)

        manure_allocation[year]['5: A_PC/A_C'] = values

        # R_PC/R_C from EFMA
        r = [
            "AT", "BE", "DK", "DE", "ES", "FI", "FR", "UK", "EL", "IE", "IT", "NL", "PT", "SE",
            "EU 15", "BG", "CY", "CZ", "EE", "HU", "LT", "LV", "PL", "RO", "SI", "SK", "EU 12",
            "EU 27", "NO", "CH"
        ]

        v = [
            0.32, 0.32, 0.56, 0.24, 0.42, 0.36, 0.23, 0.22, 0.35, 0.00, 0.36, 0.36, 0.37, 0.00,
            0.32, 0.39, 0.47, 0.33, 0.35, 0.20, 0.28, 0.41, 0.48, 0.53, 0.19, 0.23, 0.38,
            0.35, 0.31, 0.35
        ]

        region_values = dict(zip(r, v))
        default_value = 0.32  # EFMA EU average

        for region in manure_allocation[year].index.get_level_values(0).unique():
            value = region_values.get(region, default_value)
            manure_allocation[year].loc[(region,), '5: R_PC/R_C'] = value

        # Fillna manure_allocation
        manure_allocation[year] = manure_allocation[year].fillna(0)

    # Concatenate along the row-axis, introducing a first-level index = year
    concatenated = pd.concat(manure_allocation, axis=0)

    # Group by the second-level index (regions) and take the mean of each column
    mean_allocation = concatenated.groupby(level=1).mean()
    mean_allocation.index.name = 'region'

    # %% --- Define a new dataset to store manure allocation ---

    symbols = [
        'L_storage', 'G', 'G_PG', 'G_TG', 'E_total',
        'E_house', 'A_PG', 'A_TG', 'A_PC', 'A_AL'
    ]
    label_map = {
        'L_storage': 'loss from houses and storage',
        'G':         'excreted grazing',
        'G_PG':      'excreted grazing on permanent grassland',
        'G_TG':      'excreted grazing on temporary grassland',
        'E_total':   'excreted total',
        'E_house':   'excreted in house',
        'A_PG':      'applied to permanent grassland',
        'A_TG':      'applied to temporary grassland',
        'A_PC':      'applied to permanent crops',
        'A_AL':      'applied to other arable land'
    }

    # Cartesian product
    final_excretion = pd.MultiIndex.from_product(
        [sorted(regions['NUTS_ID']), symbols, sorted(years)],
        names=["region", "symbol", "year"]
    ).to_frame(index=False)

    # Region names
    region_name_map = regions.set_index('NUTS_ID')['name']
    final_excretion.insert(
        loc=1,
        column='region_name',
        value=final_excretion['region'].map(region_name_map)
    )

    # Metadata columns + labels
    final_excretion = final_excretion.assign(
        value = np.nan,
        label = final_excretion['symbol'].map(label_map),
        unit = "Gg N",
        confidence = np.nan
    )

    # Enforce the per-(region, year) symbol ordering
    final_excretion['symbol'] = pd.Categorical(
        final_excretion['symbol'],
        categories=symbols,
        ordered=True
    )

    final_excretion = (
        final_excretion
        .sort_values(['region', 'year', 'symbol'])
        .reset_index(drop=True)
    )

    #%% --- Excreted total (E_total) ---

    print('Processing E_total...')

    # List of animal categories to include in the total excretion
    animals = ['turkeys', 'ducks', 'chickens', 'bovine', 'dairy', 'pigs', 'sheep', 'goats']

    # Filter only rows corresponding to animal excretion
    animal_data = excretion[excretion['animal'].isin(animals)]

    # Compute the sum of excretion by region and year
    animal_sums = (
        animal_data
        .groupby(['region', 'year'], as_index=False)['value']
        .sum()
        .rename(columns={'value': 'sum_excretion'})
    )

    # Create a mask for the 'E_total' rows to be updated
    mask = final_excretion['symbol'] == 'E_total'

    # Merge the computed sums only onto the 'E_total' rows
    e_total = final_excretion[mask].merge(
        animal_sums,
        on=['region', 'year'],
        how='left'
    )

    # Update the 'value' column in the original DataFrame
    final_excretion.loc[mask, 'value'] = e_total['sum_excretion'].values

    # Set 'confidence' column to 'estimated' for those filled values
    final_excretion.loc[mask, 'confidence'] = 'estimated'

    #%% --- Excreted grazing (G) ---

    print('Processing G...')

    ruminants = ['bovine', 'dairy', 'sheep', 'goats']
    pivot_ruminants = {}

    # Create a pivot table for each ruminant type
    for r in ruminants:
        df_r = excretion[excretion['animal'] == r]
        pivot = df_r.pivot(index='region', columns='year', values='value')
        pivot_ruminants[r] = pivot

    # Apply allocation coefficients
    for year in years:
        for r in ruminants:
            # Retrieve allocation coefficient safely
            try:
                coef = manure_allocation[year][f'1:{r}']
            except KeyError:
                coef = None

            # Handle NaN or missing values
            if coef is None or pd.isna(coef).all():
                coef = mean_allocation[f'1:{r}']

            # Apply coefficient
            pivot_ruminants[r][year] = pivot_ruminants[r][year] * coef

    # Sum ruminant outdoor excretion
    sum_ruminant_outdoor = pd.concat(pivot_ruminants, axis=0)
    sum_ruminant_outdoor = sum_ruminant_outdoor.groupby(level=1).sum()
    sum_ruminant_outdoor.index.name = 'region'

    # Apply values in final_excretion
    for year in years:
        for region in regions['NUTS_ID']:

            value = sum_ruminant_outdoor.loc[region, year]

            mask = (
                (final_excretion['symbol'] == 'G') &
                (final_excretion['region'] == region) &
                (final_excretion['year'] == year)
            )

            final_excretion.loc[mask, 'value'] = value
            final_excretion.loc[mask, 'confidence'] = 'estimated'

    #%% --- Excreted in house (E_house) ---

    print('Processing E_house...')

    # Calculate excreted total - excreted at grazing
    for year in years:
        for region in regions['NUTS_ID']:

            E_total = final_excretion.loc[
                (final_excretion['symbol'] == 'E_total') &
                (final_excretion['region'] == region) &
                (final_excretion['year'] == year),
                'value'].values[0]

            G = final_excretion.loc[
                (final_excretion['symbol'] == 'G') &
                (final_excretion['region'] == region) &
                (final_excretion['year'] == year),
                'value'].values[0]

            value = E_total - G

            if value == np.nan:
                value = 0

            mask = (
                (final_excretion['symbol'] == 'E_house') &
                (final_excretion['region'] == region) &
                (final_excretion['year'] == year)
            )

            final_excretion.loc[mask, 'value'] = value
            final_excretion.loc[mask, 'confidence'] = 'estimated'

    #%% --- Excreted grazing in permanent grassland (G_PG) ---

    print('Processing G_PG...')

    for year in years:
        for region in regions['NUTS_ID']:

            # Quantity excreted at grazing (G)
            G = final_excretion.loc[
                (final_excretion['symbol'] == 'G') &
                (final_excretion['region'] == region) &
                (final_excretion['year'] == year),
                'value'].values[0]

            # Extract A_PG/A_G coefficient
            try:
                coef = manure_allocation[year].loc[region, '2: A_PG/A_G']
            except KeyError:
                coef = None

            # Handle NaN or missing values
            if coef is None or pd.isna(coef):
                coef = mean_allocation.loc[region, '2: A_PG/A_G']

            value = G * coef

            if value == np.nan:
                value = 0

            mask = (
                (final_excretion['symbol'] == 'G_PG') &
                (final_excretion['region'] == region) &
                (final_excretion['year'] == year)
            )

            final_excretion.loc[mask, 'value'] = value
            final_excretion.loc[mask, 'confidence'] = 'estimated'

    #%% --- Excreted grazing in temporary grassland (G_TG) ---

    print('Processing G_TG...')

    # Calculate excreted grazing total - excreted at grazing in permanent grassland
    for year in years:
        for region in regions['NUTS_ID']:

            G = final_excretion.loc[
                (final_excretion['symbol'] == 'G') &
                (final_excretion['region'] == region) &
                (final_excretion['year'] == year),
                'value'].values[0]

            G_PG = final_excretion.loc[
                (final_excretion['symbol'] == 'G_PG') &
                (final_excretion['region'] == region) &
                (final_excretion['year'] == year),
                'value'].values[0]

            value = G - G_PG

            if value == np.nan:
                value = 0

            mask = (
                (final_excretion['symbol'] == 'G_TG') &
                (final_excretion['region'] == region) &
                (final_excretion['year'] == year)
            )

            final_excretion.loc[mask, 'value'] = value
            final_excretion.loc[mask, 'confidence'] = 'estimated'

    #%% --- Losses from houses and storage (L_storage) ---

    print('Processing L_storage...')

    for year in years:
        for region in regions['NUTS_ID']:

            # Quantity excreted in house (E_house)
            E_house = final_excretion.loc[
                (final_excretion['symbol'] == 'E_house') &
                (final_excretion['region'] == region) &
                (final_excretion['year'] == year),
                'value'].values[0]

            # Extract the loss at storage coefficient
            try:
                coef = manure_allocation[year].loc[region, '3: losses at storage']
            except KeyError:
                coef = None

            # Handle NaN or missing values
            if coef is None or pd.isna(coef):
                coef = mean_allocation.loc[region, '3: losses at storage']

            value = E_house * coef

            mask = (
                (final_excretion['symbol'] == 'L_storage') &
                (final_excretion['region'] == region) &
                (final_excretion['year'] == year)
            )

            final_excretion.loc[mask, 'value'] = value
            final_excretion.loc[mask, 'confidence'] = 'estimated'

    #%% --- Stored manure applied to permanent grassland, temporary grassland, permanent crops and arable land ---

    print('Processing A_PG, A_TG, A_PC, A_AL...')

    for year in years:
        for region in regions['NUTS_ID']:
            # === 1. Compute quantity applied to field: Excretion in house - Storage losses ===
            try:
                E_house = final_excretion.loc[
                    (final_excretion['symbol'] == 'E_house') &
                    (final_excretion['region'] == region) &
                    (final_excretion['year'] == year),
                    'value'].values[0]

                L_storage = final_excretion.loc[
                    (final_excretion['symbol'] == 'L_storage') &
                    (final_excretion['region'] == region) &
                    (final_excretion['year'] == year),
                    'value'].values[0]
            except IndexError:
                continue  # Skip if any required value is missing

            E_applied = E_house - L_storage

            # === 2. Retrieve allocation coefficients (with fallback to mean_allocation if missing or NaN) ===
            allocation = manure_allocation.get(year)

            def safe_coef(df, region, col, fallback_df=None):
                try:
                    val = df.loc[region, col]
                    if pd.isna(val) and fallback_df is not None:
                        return fallback_df.loc[region, col]
                    return val
                except KeyError:
                    if fallback_df is not None:
                        return fallback_df.loc[region, col]
                    return np.nan

            coef_g_dest = safe_coef(allocation, region, '4: grassland storage destination', mean_allocation)
            coef_g_area = safe_coef(allocation, region, '4: A_G/A', mean_allocation)
            coef_pg_area = safe_coef(allocation, region, '2: A_PG/A_G', mean_allocation)
            coef_rpc_rc = safe_coef(allocation, region, '5: R_PC/R_C', mean_allocation)
            coef_apc_ac = safe_coef(allocation, region, '5: A_PC/A_C', mean_allocation)

            # === 3. Compute manure allocation to partial grassland, total grassland, permanent cropland, and arable land ===
            value_pg = E_applied * coef_g_dest * coef_g_area * coef_pg_area
            value_tg = (E_applied * coef_g_dest * coef_g_area) - value_pg
            value_al = (E_applied - (value_pg + value_tg)) * (1 + coef_rpc_rc * coef_apc_ac) ** -1
            value_pc = (E_applied - (value_pg + value_tg)) - value_al

            # === 4. Update the final_excretion DataFrame with estimated values ===
            updates = {
                'A_PG': value_pg,
                'A_TG': value_tg,
                'A_PC': value_pc,
                'A_AL': value_al,
            }

            for symbol, value in updates.items():
                mask = (
                    (final_excretion['symbol'] == symbol) &
                    (final_excretion['region'] == region) &
                    (final_excretion['year'] == year)
                )
                final_excretion.loc[mask, 'value'] = value
                final_excretion.loc[mask, 'confidence'] = 'estimated'

    #%% return
    return final_excretion
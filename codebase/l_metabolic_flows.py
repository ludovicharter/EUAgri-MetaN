"""
Script name: l_metabolic_flows.py
Description: Cleans, corrects, fills and reformat agricultural metabolic data for each territory.
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

#%% --- Code function ---
def run_metabolic_data():
    print("Running metabolic data workflow...")

    #%% --- Ignore warnings related to xlsx files ---
    warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

    #%% --- Initialize the tidy format ---

    # Load region ids
    regions = pd.read_csv('data/regions.csv', sep=';')

    # Define the period
    years = np.arange(1990, 2020)

    # Define the indicators: code, label, and unit
    indicators = [
        ('H_ingestion', 'total human ingestion', 'Gg N'),
        ('H_ingestion_cereals', 'total human ingestion of cereal', 'Gg N'),
        ('E_total', 'total animal excretion', 'Gg N'),
        ('A_non_edible', 'non-edible animal production', 'Gg N'),
        ('Eggs', 'egg production', 'Gg N'),
        ('Milk', 'cows milk production', 'Gg N'),
        ('Meat_bovine', 'bovine meat production', 'Gg N'),
        ('Meat_sheep', 'sheep meat production', 'Gg N'),
        ('Meat_goats', 'goats meat production', 'Gg N'),
        ('Meat_pigs', 'pigs meat production', 'Gg N'),
        ('Meat_poultry', 'poultry meat production', 'Gg N'),
        ('Agri_total', 'total agricultural production', 'Gg N'),
        ('M_ingestion', 'monogastric ingestion', 'Gg N'),
        ('R_ingestion', 'ruminant ingestion', 'Gg N'),
        ('total_A_ingestion', 'total animal ingestion', 'Gg N'),
        ('L_density', 'livestock density', 'LU/haUAA'),
        ('F_import', 'net import feed', 'Gg N'),
        ('H_total', 'total crop production', 'Gg N'),
        ('C_total', 'total cereal production', 'Gg N'),
        ('PG_H', 'permanent grassland production', 'Gg N'),
        ('F_ingestion', 'effective fodder ingestion', 'Gg N'),
        ('total_N_input', 'soil N input from manure', '%'),
        ('NS', 'soil N surplus', 'kgN/ha'),
        ('NUE', 'NUE arable', 'no unit'),
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
    metabolic = mi.to_frame(index=False)

    # Map code to label and unit
    label_map = {code: label for code, label, unit in indicators}
    unit_map = {code: unit for code, label, unit in indicators}

    metabolic['label'] = metabolic['symbol'].map(label_map)
    metabolic['unit'] = metabolic['symbol'].map(unit_map)
    metabolic['value'] = np.nan

    # Add the region name by mapping from regions['name']
    name_map = dict(zip(regions['NUTS_ID'], regions['name']))
    metabolic.insert(
        loc=metabolic.columns.get_loc('region') + 1,
        column='region name',
        value=metabolic['region'].map(name_map)
    )

    #%% --- Arable N surplus ---

    # Import budget data
    budget = pd.read_csv('data/outputs/arable_budget.csv')

    manure_al_complete = (
        budget
        .loc[budget['symbol'] == 'M', ['region', 'year', 'value']]
        .pivot(index='region', columns='year', values='value')
        .sort_index()
    )

    fertilizer_al_complete = (
        budget
        .loc[budget['symbol'] == 'F', ['region', 'year', 'value']]
        .pivot(index='region', columns='year', values='value')
        .sort_index()
    )

    fixation_al_complete = (
        budget
        .loc[budget['symbol'] == 'B', ['region', 'year', 'value']]
        .pivot(index='region', columns='year', values='value')
        .sort_index()
    )

    atm_dep_complete = (
        budget
        .loc[budget['symbol'] == 'D', ['region', 'year', 'value']]
        .pivot(index='region', columns='year', values='value')
        .sort_index()
    )

    prod_arable_wide = (
        budget
        .loc[budget['symbol'] == 'H', ['region', 'year', 'value']]
        .pivot(index='region', columns='year', values='value')
        .sort_index()
    )

    arable_areas = (
        budget
        .loc[budget['symbol'] == 'A', ['region', 'year', 'value']]
        .pivot(index='region', columns='year', values='value')
        .sort_index()
    )

    N_balance = (
        budget
        .loc[budget['symbol'] == 'NS', ['region', 'year', 'value']]
        .pivot(index='region', columns='year', values='value')
        .sort_index()
    )

    # Save into the tidy df
    for region in regions['NUTS_ID']:
        for year in years:
            value = N_balance.at[region, year]
            mask = (
                (metabolic['region'] == region) &
                (metabolic['year'] == int(year)) &
                (metabolic['symbol'] == 'NS')
            )
            metabolic.loc[mask, 'value'] = value

    #%% --- NUE arable ---

    nue = (
        budget
        .loc[budget['symbol'] == 'NUE', ['region', 'year', 'value']]
        .pivot(index='region', columns='year', values='value')
        .sort_index()
    )

    # Save into the tidy df
    for region in regions['NUTS_ID']:
        for year in years:
            value = nue.at[region, year]
            mask = (
                (metabolic['region'] == region) &
                (metabolic['year'] == int(year)) &
                (metabolic['symbol'] == 'NUE')
            )
            metabolic.loc[mask, 'value'] = value

    #%% --- Population and diet ---

    # Import population data
    demo_path = 'data/Eurostat/humans/demo_r_pjangroup__custom_17258396_spreadsheet.xlsx'
    demo = utils.import_eurostat_sheets(demo_path, 1, header_row=9)

    # Regional
    demo_regions = pd.DataFrame({'region': regions['NUTS_ID']})

    for year in years:

        # Default to NaN
        demo_regions[year] = np.nan

        # Regions
        for region in regions['NUTS_ID']:
            match = demo.loc[demo['region'] == region, year]
            if not match.empty and pd.notna(match.values[0]):
                value = match.values[0]
            else:
                subregions = demo[
                    demo['region'].astype(str).str.startswith(region)
                ]
                if not subregions.empty and subregions[year].notna().all():
                    value = subregions[year].sum(skipna=True)
                else:
                    value = np.nan
            demo_regions.loc[demo_regions['region'] == region, year] = value

    # Set 'region' as index
    demo_regions = demo_regions.set_index('region')

    # Interpolate/extrapolate missing values
    demo_regions = demo_regions.interpolate(method='linear', axis=1, limit_direction='both')

    # Import human ingestion
    human_ingestion_data = pd.read_csv('data/literature_various/human_ingestion.csv', sep=';')
    human_ingestion_data = human_ingestion_data.set_index('region')
    human_ingestion = human_ingestion_data.loc[demo_regions.index, 'total per cap ingestion (kgN/cap/yr)']
    human_ingestion_cereals = human_ingestion_data.loc[demo_regions.index, 'cereal protein ingestion (kgN/cap/yr)']

    # Calcul the total human ingestion (Gg N)
    total_human_ingestion = (
        demo_regions[years]
        .multiply(human_ingestion, axis=0)
        / 1_000_000
    )

    total_cereal_ingestion = (
        demo_regions[years]
        .multiply(human_ingestion_cereals, axis=0)
        / 1_000_000
    )

    # Save into the tidy df
    for region in regions['NUTS_ID']:
        for year in years:
            value = total_human_ingestion.at[region, year]
            mask = (
                (metabolic['region'] == region) &
                (metabolic['year'] == int(year)) &
                (metabolic['symbol'] == 'H_ingestion')
            )
            metabolic.loc[mask, 'value'] = value

    # Save into the tidy df
    for region in regions['NUTS_ID']:
        for year in years:
            value = total_cereal_ingestion.at[region, year]
            mask = (
                (metabolic['region'] == region) &
                (metabolic['year'] == int(year)) &
                (metabolic['symbol'] == 'H_ingestion_cereals')
            )
            metabolic.loc[mask, 'value'] = value

    #%% --- Livestock density ---

    # Import data

    UAS = pd.read_csv('data/outputs/land_areas.csv')
    # Crop land in use areas (UAA)
    area_crops = (
        UAS
        .loc[UAS['symbol'] == 'C_sum', ['region', 'year', 'value']]
        .pivot(index='region', columns='year', values='value')
        .sort_index()
    )

    manure_allocation = pd.read_csv('data/outputs/manure_allocation.csv')
    # Crop land in use areas (UAA)
    total_manure = (
        manure_allocation
        .loc[manure_allocation['symbol'] == 'E_total', ['region', 'year', 'value']]
        .pivot(index='region', columns='year', values='value')
        .sort_index()
    )

    # Compute LU = 85 kgN/year
    LU = (total_manure) / 85

    # Compute density (LU/haUAA)
    density = LU / area_crops

    # Save into the tidy df
    for region in regions['NUTS_ID']:
        for year in years:
            value = density.at[region, year]
            mask = (
                (metabolic['region'] == region) &
                (metabolic['year'] == int(year)) &
                (metabolic['symbol'] == 'L_density')
            )
            metabolic.loc[mask, 'value'] = value

    #%% --- Animal production ---

    # Load country list
    countries = pd.read_csv('data/countries.csv', sep=';')

    # Import data
    path_pann = 'data/Eurostat/animal_production/apro_mt_pann__custom_17279498_spreadsheet.xlsx'
    path_sloth = 'data/Eurostat/animal_production/apro_mt_sloth__custom_17279527_spreadsheet.xlsx'
    path_milk = 'data/Eurostat/animal_production/agr_r_milkpr__custom_17279596_spreadsheet.xlsx'
    path_eggs = 'data/FAOSTAT/animal_productions/FAOSTAT_data_en_6-30-2025.csv'
    path_norway = 'data/FAOSTAT/animal_productions/FAOSTAT_data_en_Norway.csv'

    # Meat
    meat_indices = [1, 11, 13, 17, 19]
    other_meat_indices = [1, 11, 13, 17]
    meat_list = ['bovine', 'pigs', 'sheep', 'goats', 'poultry']
    other_meat_list = ['bovine', 'pigs', 'sheep', 'goats']

    meat = utils.import_eurostat_sheets(path_pann, meat_indices, header_row=9)
    other_meat = utils.import_eurostat_sheets(path_sloth, other_meat_indices, header_row=9)

    production_countries_ini = utils.build_eurostat_dict(np.arange(1990, 2020), meat_list, meat, countries)
    other_production_countries = utils.build_eurostat_dict(np.arange(1990, 2020), other_meat_list, other_meat, countries)

    # Calculate the pann + sloth
    production_countries = utils.sum_dicts_on_common_cols(production_countries_ini, other_production_countries, other_meat_list)

    # Milk
    milk = utils.import_eurostat_sheets(path_milk, 1, header_row=8)
    # Initialize the DataFrame with all regions and NaN for each year
    milk_regions = pd.DataFrame({'region': regions['NUTS_ID']})
    milk_regions.set_index('region', inplace=True)
    milk_regions[years] = np.nan  # pre-fill all years with NaN
    # Fill values
    for year in years:
        for region in milk_regions.index:
            if year in milk.columns:
                # Try direct match
                if region in milk['region'].values:
                    value = milk.loc[milk['region'] == region, year].values[0]
                    if pd.notna(value):
                        milk_regions.loc[region, year] = value
                        continue
                # Otherwise try aggregation of subregions
                subregions = milk[milk['region'].astype(str).str.startswith(region)]
                if not subregions.empty and subregions[year].notna().any():
                    value = subregions[year].sum(skipna=True)
                else:
                    value = np.nan
                milk_regions.loc[region, year] = value

    # Eggs
    eggs = pd.read_csv(path_eggs, sep=';')
    eggs = eggs.set_index(['region', 'Year'])
    eggs = eggs.loc[eggs['Unit'] == 't']

    # Calculate animal regional repartition coefficients
    with open("data/outputs/intermediate_datasets/animals_dict.pkl", "rb") as f:
        animals = pickle.load(f)  # Load animal dict

    # Set 'region' as index + add poultry prod
    for year in years:
        animals[year] = animals[year].set_index('region')
        production_countries[year] = production_countries[year].set_index('region')
        production_countries_ini[year] = production_countries_ini[year].set_index('region')
        production_countries[year]['poultry'] = production_countries_ini[year]['poultry']

    animal_names = ['bovine', 'pigs', 'sheep', 'goats', 'chickens']
    animal_coefficients = utils.compute_regional_distribution_coefficients(animals, regions, crops=animal_names, years=years)
    animal_coefficients['poultry'] = animal_coefficients.pop('chickens')

    # Set 'region' as index
    for animal in meat_list:
        animal_coefficients[animal] = animal_coefficients[animal].set_index('region')

    # Initialize a dict to store regional production
    production_regions: dict[int, pd.DataFrame] = {
        year: pd.DataFrame(np.nan, index=regions['NUTS_ID'], columns=meat_list)
        for year in years
    }

    # Calculate production for each region
    for year in years:
        for region in regions['NUTS_ID']:
            country = region[:2]

            for animal in meat_list:
                coeffs = animal_coefficients[animal]

                # Try to get the region-year coefficient using .get() if it's a Series
                try:
                    coef = coeffs.loc[(region, year)]
                except KeyError:
                    coef = None

                # Fallback to the regional mean if coef is missing or NaN
                if coef is None or pd.isna(coef):
                    coef = coeffs.loc[(region, 'mean')]

                # Compute regional production
                production = production_countries[year].loc[country, animal] * coef
                production_regions[year].loc[region, animal] = production

    # Calculate egg production for each region
    for year in years:
        # Initialize eggs column
        production_regions[year]['eggs'] = np.nan

        for region in regions['NUTS_ID']:
            country = region[:2]
            coeffs = animal_coefficients['poultry']

            # Try to get the region-year coefficient using .get() if it's a Series
            try:
                coef = coeffs.loc[(region, year)]
            except KeyError:
                coef = None
            # Fallback to the regional mean if coef is missing or NaN
            if coef is None or pd.isna(coef):
                coef = coeffs.loc[(region, 'mean')]

            # Compute regional production
            try:
                val = eggs.loc[(country, year), 'Value']
            except KeyError:
                val = np.nan

            if pd.notna(val):
                egg_val = val * coef
                production_regions[year].loc[region, 'eggs'] = egg_val / 1000  # unit: kt

    # Insert milk production into production_region
    for year in years:
        # Initialize milk column
        production_regions[year]['milk'] = np.nan

        for region in regions['NUTS_ID']:
            try:
                val = milk_regions.loc[(region, year)]
            except KeyError:
                val = np.nan

            if pd.notna(val):
                production_regions[year].loc[region, 'milk'] = val  # unit: kt

    # Fill gaps in NO using data from FAOSTAT
    no = pd.read_csv(path_norway, sep=',')
    no = no.set_index('Year')

    def safe_extract(df, year, code):
        result = df.loc[(df.index == year) & (df['Item Code (CPC)'] == code), 'Value']
        return result.squeeze() if not result.empty else np.nan

    for year in years:
        bovine = safe_extract(no, year, 21111.01)
        pigs = safe_extract(no, year, 21113.01)
        sheep = safe_extract(no, year, 21115.0)
        goats = safe_extract(no, year, 21116.0)
        poultry = safe_extract(no, year, 21121.0)
        milk = safe_extract(no, year, 2211.0)

        production_regions[year].loc['NO', 'bovine'] = bovine / 1000  # unit: kt
        production_regions[year].loc['NO', 'pigs'] = pigs / 1000  # unit: kt
        production_regions[year].loc['NO', 'sheep'] = sheep / 1000  # unit: kt
        production_regions[year].loc['NO', 'goats'] = goats / 1000  # unit: kt
        production_regions[year].loc['NO', 'poultry'] = poultry / 1000  # unit: kt
        production_regions[year].loc['NO', 'milk'] = milk / 1000  # unit: kt

    # Interpolation / extrapolation
    production_list = ['bovine', 'pigs', 'sheep', 'goats', 'poultry', 'eggs', 'milk']
    production_regions = utils.interpolate_dt(production_regions, production_list, extrapolate=True)

    #%% --- Edible N animal production ---

    # Import N convertion coefficients
    path_N_content = 'data/literature_various/N_content_animal_products.csv'
    N_content = pd.read_csv(path_N_content, sep=';')

    # Convert N-content percentages to fractions (scalars)
    N_frac_edible = {
        'bovine':   N_content['%N_content_edible_bovine'] / 100,
        'pigs':     N_content['%N_content_edible_pigs'] / 100,
        'sheep':    N_content['%N_content_edible_sheep'] / 100,
        'goats':    N_content['%N_content_edible_goats'] / 100,
        'poultry':  N_content['%N_content_edible_poultry'] / 100,
        'eggs':     N_content['%N_content_edible_eggs'] / 100,
        'milk':     N_content['%N_content_edible_milk'] / 100
    }

    N_frac_non_edible = {
        'bovine':   (N_content['%N_content_non-edible_bovine']) / 100,
        'pigs':     (N_content['%N_content_non-edible_pigs']) / 100,
        'sheep':    (N_content['%N_content_non-edible_sheep']) / 100,
        'goats':    (N_content['%N_content_non-edible_goats']) / 100,
        'poultry':  (N_content['%N_content_non-edible_poultry']) / 100,
    }

    N_frac_total = {
        'bovine':   (N_content['%N_content_edible_bovine'] + N_content['%N_content_non-edible_bovine']) / 100,
        'pigs':     (N_content['%N_content_edible_pigs'] + N_content['%N_content_non-edible_pigs']) / 100,
        'sheep':    (N_content['%N_content_edible_sheep'] + N_content['%N_content_non-edible_sheep']) / 100,
        'goats':    (N_content['%N_content_edible_goats'] + N_content['%N_content_non-edible_goats']) / 100,
        'poultry':  (N_content['%N_content_edible_poultry'] + N_content['%N_content_non-edible_poultry']) / 100,
        'eggs':     N_content['%N_content_edible_eggs'] / 100,
        'milk':     N_content['%N_content_edible_milk'] / 100
    }

    # Define ruminant and monogastric products
    ruminant = ['bovine', 'sheep', 'goats', 'milk']
    monogastric = ['poultry', 'pigs', 'eggs']
    meat = ['bovine', 'sheep', 'goats', 'poultry', 'pigs']

    for year in years:
        pr = production_regions[year]

        total_vals = []
        for animal in N_frac_total:
            vals = pr[animal] * float(N_frac_total[animal])
            total_vals.append(vals.values)  # numpy array

        pr['total_N_production'] = np.nansum(total_vals, axis=0)  # GgN

        edible_vals = []
        for animal in N_frac_edible:
            vals = pr[animal] * float(N_frac_edible[animal])
            edible_vals.append(vals.values)

        pr['total_N_edible_production'] = np.nansum(edible_vals, axis=0)  # GgN

        monogastric_total_vals = []
        for animal in monogastric:
            vals = pr[animal] * float(N_frac_total[animal])
            monogastric_total_vals.append(vals.values)

        pr['monogastric_total_N_production'] = np.nansum(monogastric_total_vals, axis=0)  # GgN

        ruminant_total_vals = []
        for animal in ruminant:
            vals = pr[animal] * float(N_frac_total[animal])
            ruminant_total_vals.append(vals.values)

        pr['ruminant_total_N_production'] = np.nansum(ruminant_total_vals, axis=0)  # GgN

        non_edible_vals = []
        for animal in N_frac_non_edible:
            vals = pr[animal] * float(N_frac_non_edible[animal])
            non_edible_vals.append(vals.values)

        pr['total_N_non_edible_production'] = np.nansum(non_edible_vals, axis=0)  # GgN

        # Animal production details
        for m in meat:
            pr[f'{m}_N_edible_production'] = pr[m] * float(N_frac_edible[m])  # GgN

        # Milk and eggs
        pr['egg_N_edible_production'] = pr['eggs'] * float(N_frac_edible['eggs'])  # GgN
        pr['milk_N_edible_production'] = pr['milk'] * float(N_frac_edible['milk'])  # GgN

    #%% --- Non-edible animal production and details ---

    # Save into the tidy df
    for region in regions['NUTS_ID']:
        for year in years:
            value = production_regions[year]['total_N_non_edible_production'].at[region]
            mask = (
                (metabolic['region'] == region) &
                (metabolic['year'] == int(year)) &
                (metabolic['symbol'] == 'A_non_edible')
            )
            metabolic.loc[mask, 'value'] = value

    # Save into the tidy df
    for region in regions['NUTS_ID']:
        for year in years:
            value = production_regions[year]['egg_N_edible_production'].at[region]
            mask = (
                (metabolic['region'] == region) &
                (metabolic['year'] == int(year)) &
                (metabolic['symbol'] == 'Eggs')
            )
            metabolic.loc[mask, 'value'] = value

    # Save into the tidy df
    for region in regions['NUTS_ID']:
        for year in years:
            value = production_regions[year]['milk_N_edible_production'].at[region]
            mask = (
                (metabolic['region'] == region) &
                (metabolic['year'] == int(year)) &
                (metabolic['symbol'] == 'Milk')
            )
            metabolic.loc[mask, 'value'] = value

    # Save into the tidy df
    for region in regions['NUTS_ID']:
        for year in years:
            value = production_regions[year]['bovine_N_edible_production'].at[region]
            mask = (
                (metabolic['region'] == region) &
                (metabolic['year'] == int(year)) &
                (metabolic['symbol'] == 'Meat_bovine')
            )
            metabolic.loc[mask, 'value'] = value

    # Save into the tidy df
    for region in regions['NUTS_ID']:
        for year in years:
            value = production_regions[year]['sheep_N_edible_production'].at[region]
            mask = (
                (metabolic['region'] == region) &
                (metabolic['year'] == int(year)) &
                (metabolic['symbol'] == 'Meat_sheep')
            )
            metabolic.loc[mask, 'value'] = value

    # Save into the tidy df
    for region in regions['NUTS_ID']:
        for year in years:
            value = production_regions[year]['goats_N_edible_production'].at[region]
            mask = (
                (metabolic['region'] == region) &
                (metabolic['year'] == int(year)) &
                (metabolic['symbol'] == 'Meat_goats')
            )
            metabolic.loc[mask, 'value'] = value

    # Save into the tidy df
    for region in regions['NUTS_ID']:
        for year in years:
            value = production_regions[year]['poultry_N_edible_production'].at[region]
            mask = (
                (metabolic['region'] == region) &
                (metabolic['year'] == int(year)) &
                (metabolic['symbol'] == 'Meat_poultry')
            )
            metabolic.loc[mask, 'value'] = value

    # Save into the tidy df
    for region in regions['NUTS_ID']:
        for year in years:
            value = production_regions[year]['pigs_N_edible_production'].at[region]
            mask = (
                (metabolic['region'] == region) &
                (metabolic['year'] == int(year)) &
                (metabolic['symbol'] == 'Meat_pigs')
            )
            metabolic.loc[mask, 'value'] = value

    #%% --- Permanent grassland harvested quantities ---

    # Import permanent grassland surface (PG symbol in UAS)
    perm_grass_surface = (
        UAS
        .loc[UAS['symbol'] == 'PG', ['region', 'year', 'value']]
        .pivot(index='region', columns='year', values='value')
        .sort_index()
    )

    # Import permanent grassland yields (constant value per region)
    perm_grass_yield = pd.read_csv('data/literature_various/perm_grassland_yields.csv', sep=';')

    # Align regions to match the surface DataFrame index (preserve NaNs)
    perm_grass_yield = (
        perm_grass_yield
        .set_index('region')
        .reindex(perm_grass_surface.index)
    )

    # Multiply surface (ha) by yield (kg N/ha) → total N harvested (kg N)
    perm_grass_harvest = perm_grass_surface.mul(perm_grass_yield['PG_N_yield'], axis=0)

    # Save into the tidy df (total agricultural production)
    for region in regions['NUTS_ID']:
        for year in years:
            value = perm_grass_harvest.at[region, year]
            mask = (
                (metabolic['region'] == region) &
                (metabolic['year'] == int(year)) &
                (metabolic['symbol'] == 'PG_H')
            )
            metabolic.loc[mask, 'value'] = value

    #%% --- Total agricultural production ---

    # Harvested quantities
    crops = ['Wheat', 'Other cereals', 'Grain maize', 'Barley', 'Fodder crops', 'Oilseeds', 'Potatoes', 'Pulses',
             'Sugar beet', 'Temporary grassland', 'Vegetables and other', 'Forage legumes', 'Olives', 'Grapes',
             'Other permanent crops']
    cereals = ['Wheat', 'Other cereals', 'Grain maize', 'Barley']
    crop_production = pd.read_csv('data/outputs/crop_production_all_categories.csv')
    crop_production = crop_production.loc[crop_production['symbol'] == 'H']
    harvested_quantities = (
        crop_production
        .loc[crop_production['crop'].isin(crops)]
        .groupby(['region', 'year'])['value'].sum()
        .unstack('year')
        .sort_index()
    )

    harvested_quantities_cereals = (
        crop_production
        .loc[crop_production['crop'].isin(cereals)]
        .groupby(['region', 'year'])['value'].sum()
        .unstack('year')
        .sort_index()
    )

    # Save into the tidy df (total agricultural production)
    for region in regions['NUTS_ID']:
        for year in years:
            value = harvested_quantities.at[region, year] + production_regions[year].at[region, 'total_N_edible_production']
            mask = (
                (metabolic['region'] == region) &
                (metabolic['year'] == int(year)) &
                (metabolic['symbol'] == 'Agri_total')
            )
            metabolic.loc[mask, 'value'] = value

    # Save into the tidy df (total crop production)
    for region in regions['NUTS_ID']:
        for year in years:
            value = harvested_quantities.at[region, year]
            mask = (
                (metabolic['region'] == region) &
                (metabolic['year'] == int(year)) &
                (metabolic['symbol'] == 'H_total')
            )
            metabolic.loc[mask, 'value'] = value

    # Save into the tidy df (total cereal production)
    for region in regions['NUTS_ID']:
        for year in years:
            value = harvested_quantities_cereals.at[region, year]
            mask = (
                (metabolic['region'] == region) &
                (metabolic['year'] == int(year)) &
                (metabolic['symbol'] == 'C_total')
            )
            metabolic.loc[mask, 'value'] = value

    #%% --- Animal ingestion ---

    # Import total excretion
    path_excretion = 'data/outputs/manure_allocation.csv'
    excretion = pd.read_csv(path_excretion)
    total_excretion = (
        excretion
        .loc[excretion['symbol'] == 'E_total', ['region', 'year', 'value']]
        .pivot(index='region', columns='year', values='value')
        .sort_index()
    )

    # Save into the tidy df
    for region in regions['NUTS_ID']:
        for year in years:
            value = total_excretion.at[region, year] + production_regions[year].at[region, 'total_N_production']
            mask = (
                (metabolic['region'] == region) &
                (metabolic['year'] == int(year)) &
                (metabolic['symbol'] == 'total_A_ingestion')
            )
            metabolic.loc[mask, 'value'] = value

    #%% --- Total N excretion ---

    # Save into the tidy df
    for region in regions['NUTS_ID']:
        for year in years:
            value = total_excretion.at[region, year]
            mask = (
                (metabolic['region'] == region) &
                (metabolic['year'] == int(year)) &
                (metabolic['symbol'] == 'E_total')
            )
            metabolic.loc[mask, 'value'] = value

    #%% --- Monogastric and ruminant ingestion ---

    # Excretion data
    ruminant = ['bovine', 'dairy', 'sheep', 'goats']
    monogastric = ['turkeys', 'ducks', 'chickens', 'pigs']

    livestock_excretion = pd.read_csv('data/outputs/animal_excretion.csv')

    ruminant_excretion = (
        livestock_excretion
        .loc[livestock_excretion['animal'].isin(ruminant)]
        .groupby(['region', 'year'])['value'].sum()
        .unstack('year')
        .sort_index()
    )

    monogastric_excretion = (
        livestock_excretion
        .loc[livestock_excretion['animal'].isin(monogastric)]
        .groupby(['region', 'year'])['value'].sum()
        .unstack('year')
        .sort_index()
    )

    # Save into the tidy df
    for region in regions['NUTS_ID']:
        for year in years:

            # Monogastric
            value = monogastric_excretion.at[region, year] + production_regions[year].at[region, 'monogastric_total_N_production']
            mask = (
                (metabolic['region'] == region) &
                (metabolic['year'] == int(year)) &
                (metabolic['symbol'] == 'M_ingestion')
            )
            metabolic.loc[mask, 'value'] = value

            # Ruminant
            value = ruminant_excretion.at[region, year] + production_regions[year].at[region, 'ruminant_total_N_production']
            mask = (
                (metabolic['region'] == region) &
                (metabolic['year'] == int(year)) &
                (metabolic['symbol'] == 'R_ingestion')
            )
            metabolic.loc[mask, 'value'] = value

    #%% --- Animal nutrition balance ---

    # Effective grass consumption

    ruminant_ingestion = (
        metabolic
        .loc[metabolic['symbol'] == 'R_ingestion', ['region', 'year', 'value']]
        .pivot(index='region', columns='year', values='value')
        .sort_index()
    )

    effective_grass_ingestion = np.minimum(perm_grass_harvest, ruminant_ingestion)

    # Effective fodder consumption

    fodder_crops = ['Fodder crops', 'Oilseeds', 'Pulses', 'Temporary grassland', 'Forage legumes']
    fodder_harvest = (
        crop_production
        .loc[crop_production['crop'].isin(fodder_crops)]
        .groupby(['region', 'year'])['value'].sum()
        .unstack('year')
        .sort_index()
    )

    total_ingestion = (
        metabolic
        .loc[metabolic['symbol'] == 'total_A_ingestion', ['region', 'year', 'value']]
        .pivot(index='region', columns='year', values='value')
        .sort_index()
    )

    effective_fodder_ingestion = np.minimum(total_ingestion - effective_grass_ingestion, fodder_harvest)

    # Save into the tidy df
    for region in regions['NUTS_ID']:
        for year in years:
            value = effective_fodder_ingestion.at[region, year]
            mask = (
                (metabolic['region'] == region) &
                (metabolic['year'] == int(year)) &
                (metabolic['symbol'] == 'F_ingestion')
            )
            metabolic.loc[mask, 'value'] = value

    human_cereal_ingestion = (
        metabolic
        .loc[metabolic['symbol'] == 'H_ingestion_cereals', ['region', 'year', 'value']]
        .pivot(index='region', columns='year', values='value')
        .sort_index()
    )
    cereal_production = (
        metabolic
        .loc[metabolic['symbol'] == 'C_total', ['region', 'year', 'value']]
        .pivot(index='region', columns='year', values='value')
        .sort_index()
    )

    # Subtract 1.5 times human cereal ingestion to estimate remaining cereals
    remaining_cereals = cereal_production - (1.5 * human_cereal_ingestion)

    # Keep only positive values (set negative values to 0)
    remaining_cereals_positive = remaining_cereals.where(remaining_cereals > 0, 0)

    # Compute effective cereal ingestion: limited by available cereals and remaining ingestion need
    effective_cereal_ingestion = np.minimum(
        remaining_cereals_positive,
        total_ingestion - effective_grass_ingestion - effective_fodder_ingestion
    )

    #%% --- Net import feed ---

    imported_feed = np.maximum(0, total_ingestion - effective_grass_ingestion - effective_fodder_ingestion - effective_cereal_ingestion)

    # Save into the tidy df
    for region in regions['NUTS_ID']:
        for year in years:
            value = imported_feed.at[region, year]
            mask = (
                (metabolic['region'] == region) &
                (metabolic['year'] == int(year)) &
                (metabolic['symbol'] == 'F_import')
            )
            metabolic.loc[mask, 'value'] = value

    #%% --- Cropland soil N input from manure ---

    # Manure to cropland
    manure = (
        budget
        .loc[budget['symbol'] == 'M', ['region', 'year', 'value']]
        .pivot(index='region', columns='year', values='value')
        .sort_index()
    )

    # Other N input to cropland
    others = ['B', 'D', 'F']
    other_N_input = (
        budget
        .loc[budget['symbol'].isin(others)]
        .groupby(['region', 'year'])['value'].sum()
        .unstack('year')
        .sort_index()
    )

    # Cropland soil N input from manure
    N_from_manure = manure / (manure + other_N_input)

    # Save into the tidy df
    for region in regions['NUTS_ID']:
        for year in years:
            value = N_from_manure.at[region, year] * 100
            mask = (
                (metabolic['region'] == region) &
                (metabolic['year'] == int(year)) &
                (metabolic['symbol'] == 'total_N_input')
            )
            metabolic.loc[mask, 'value'] = value

    return metabolic

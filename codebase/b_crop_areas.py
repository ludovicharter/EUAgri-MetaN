"""
Script name: b_crop_areas.py
Description: Cleans and reformats agricultural area data for each territory.
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
from collections import OrderedDict
import warnings
import os
import codebase.utils as utils

#%% Code function
def run_areas():
    print("Running crop areas workflow...")

    # %% --- Ignore warnings related to xlsx files ---
    warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

    # %% --- Load and process surface datasets ---

    # File paths
    path_cereals = 'data/Eurostat/surface/apro_cpshr__custom_16598049_cereals_areas_2000-2023.xlsx'
    path_others = 'data/Eurostat/surface/apro_cpshr__custom_16599299_other_areas_2000-2023.xlsx'
    path_historical = 'data/Eurostat/surface/apro_cpnhr_h__custom_16597439_areas_1990-1999.xlsx'
    path_euragri = 'data/EuropeAgriDB_v1.0/tables/crop_production_all_categories.csv'

    # Indices of sheets in Eurostat files (cereals and others)
    cereal_indices = [6, 9, 10, 13, 17, 18, 19, 20, 21, 23, 22, 3]
    other_indices = [13, 16, 17, 18, 20, 1, 9, 37, 7, 8, 40, 22, 23, 24, 33, 32, 47, 48, 46, 45, 49]
    historical_indices = [6, 9, 10, 13, 17, 18, 19, 20, 21, 23, 22, 3, 37, 41, 42, 43, 45, 26, 34, 62, 32, 33, 65, 47,
                          48, 49, 58, 57, 72, 73, 71, 70, 74]

    # List of crop identifiers (align with sheet indices)
    crops = ['wheatspelt1', 'wheatspelt2', 'rye', 'Barley', 'oats1', 'oats2', 'Grain maize', 'triticale', 'sorghum',
             'rice', 'othercereals', 'graincereals', 'oilrape', 'sunflower', 'soy', 'flax', 'otheroilseeds', 'Pulses',
             'otherroot', 'Green maize', 'Potatoes', 'Sugar beet', 'Vegetables and other', 'fibreflax', 'hemp',
             'fibrecotton', 'legharvgr', 'Temporary grassland', 'grapes', 'olives', 'citrus', 'fruits',
             'other permanent']

    # Load region ids
    regions = pd.read_csv('data/regions.csv', sep=';')
    countries = pd.read_csv('data/countries.csv', sep=';')

    # Load modern data (2000–2019)
    cereals_dfs = utils.import_eurostat_sheets(path_cereals, cereal_indices)
    others_dfs = utils.import_eurostat_sheets(path_others, other_indices)
    codes = cereals_dfs + others_dfs
    # Regional areas
    areas = utils.build_eurostat_dict(np.arange(2000, 2020), crops, codes, regions)
    # National areas
    areas_countries = utils.build_eurostat_dict(np.arange(2000, 2020), crops, codes, countries)

    # Load historical data (1990–1999)
    codes_hist = utils.import_eurostat_sheets(path_historical, historical_indices)
    # Regional areas
    areas_hist = utils.build_eurostat_dict(np.arange(1990, 2000), crops, codes_hist, regions)
    # National areas
    areas_hist_countries = utils.build_eurostat_dict(np.arange(1990, 2000), crops, codes_hist, countries)

    # Combine both periods into a single dictionary
    areas.update(areas_hist)
    areas_countries.update(areas_hist_countries)

    # Chronological order
    areas = OrderedDict(sorted(areas.items()))
    areas = dict(areas)
    areas_countries = OrderedDict(sorted(areas_countries.items()))
    areas_countries = dict(areas_countries)

    # %% Save areas (flag confidence: high)

    # Create a df template
    years = np.arange(1990, 2020)  # Define years
    final_crops = ['Wheat', 'Other cereals', 'Grain maize', 'Barley', 'Fodder crops', 'Oilseeds', 'Potatoes', 'Pulses',
                   'Sugar beet', 'Temporary grassland', 'Vegetables and other', 'Forage legumes', 'Olives', 'Grapes',
                   'Other permanent crops']

    # Create df product of region, crop, and year
    final_areas = pd.MultiIndex.from_product(
        [sorted(regions['NUTS_ID']), sorted(final_crops), years],
        names=["region", "crop", "year"]
    ).to_frame(index=False)

    # Add constant and NaN columns
    final_areas["value"] = np.nan
    final_areas["label"] = "harvested area"
    final_areas["unit"] = "Mha"
    final_areas["confidence"] = np.nan

    # Aggregate and convert areas values to Mha

    crop_combinations_1 = {
        'wheatspelt': {
            'cols_strict': ['wheatspelt1', 'wheatspelt2'],  # list of columns to sum with skipna=False
            'cols_flexible': []  # list of columns to sum with skipna=True
        },
        'Oats': {
            'cols_strict': ['oats1'],
            'cols_flexible': ['oats2']
        }
    }

    # Create a copy of production
    high_confidence_areas = {year: df.copy() for year, df in areas.items()}
    high_confidence_areas = utils.merge_crops(high_confidence_areas, crop_combinations_1)

    # Aggregated crop names after merging
    crops_intermediate = ['wheatspelt', 'rye', 'Barley', 'Oats', 'Grain maize', 'triticale', 'sorghum', 'rice',
                          'othercereals', 'graincereals', 'oilrape', 'sunflower', 'soy', 'flax', 'otheroilseeds',
                          'Pulses', 'otherroot', 'Green maize', 'Potatoes', 'Sugar beet', 'Vegetables and other',
                          'fibreflax', 'hemp', 'fibrecotton', 'legharvgr', 'Temporary grassland', 'grapes', 'olives',
                          'citrus', 'fruits', 'other permanent']

    # Convert areas values to Mha
    for year in high_confidence_areas:
        for crop in crops_intermediate:
            high_confidence_areas[year][crop] = high_confidence_areas[year][crop] / 1000  # Mha

    # Merge crops into aggregates
    crop_combinations_2 = {
        'Oilseeds': {
            'cols_strict': ['oilrape', 'sunflower'],
            'cols_flexible': ['otheroilseeds', 'soy', 'flax']
        },
        'Fodder crops': {
            'cols_strict': ['Green maize'],
            'cols_flexible': ['otherroot']
        },
        'Forage legumes': {
            'cols_strict': ['legharvgr'],
            'cols_flexible': []
        },
        'Vegetables and other': {
            'cols_strict': [],
            'cols_flexible': ['hemp', 'fibrecotton', 'fibreflax', 'Vegetables and other']
        },
        'Wheat': {
            'cols_strict': ['wheatspelt'],
            'cols_flexible': []
        },
        'Other cereals': {
            'cols_strict': ['Oats', 'rye', 'triticale', 'sorghum'],
            'cols_flexible': ['rice', 'othercereals']
        },
        'Grain cereals': {
            'cols_strict': ['graincereals'],
            'cols_flexible': []
        },
        'Olives': {
            'cols_strict': ['olives'],
            'cols_flexible': []
        },
        'Grapes': {
            'cols_strict': ['grapes'],
            'cols_flexible': []
        },
        'Other permanent crops': {
            'cols_strict': [],
            'cols_flexible': ['citrus', 'fruits', 'other permanent']
        },
    }
    high_confidence_areas = utils.merge_crops(high_confidence_areas, crop_combinations_2)

    # List of crops for which euragri data is prioritized
    prior_crops = ['Fodder crops', 'Vegetables and other', 'Forage legumes', 'Temporary grassland']
    # Delete these crops in the current surface dataset
    for year, df in high_confidence_areas.items():
        high_confidence_areas[year][prior_crops] = np.nan

    # fill values and set confidence to 'high'
    final_areas = utils.fill_template(final_areas, high_confidence_areas, confidence_label='unprocessed')

    # %% --- (a) Correct missing surface data using available Eurostat national surfaces ---

    # Set 'region' as production index
    for year in years:
        areas[year] = areas[year].set_index('region')
        areas_countries[year] = areas_countries[year].set_index('region')

    # Calculate distribution coefficients
    coefficients = utils.compute_regional_distribution_coefficients(areas, regions, crops, years)

    # Save coefficients
    for item, df in coefficients.items():
        df.to_csv(f'data/outputs/surface_regional_ratios/ratios_{item}.csv', index=False)

    # Load regional production distribution coefficients dictionary
    production_coefficients = {}
    # Directory containing the ratio CSV files
    coeff_dir = 'data/outputs/production_regional_ratios'
    # Iterate over all files in the directory
    for filename in sorted(os.listdir(coeff_dir)):
        # Only process files of the form "ratios_{crop}.csv" (exclude aggregated files)
        if not (
                filename.startswith('ratios_') and
                not filename.startswith('ratios_aggregated_') and
                filename.endswith('.csv')):
            continue
        # Extract the crop name between "ratios_" and ".csv"
        crop = filename[len('ratios_'):-len('.csv')]
        filepath = os.path.join(coeff_dir, filename)
        df = pd.read_csv(filepath)
        # Store the DataFrame in the dictionary under its crop name
        production_coefficients[crop] = df

    # Set 'region' as index for each crop or year
    for crop in crops:
        coefficients[crop].set_index("region", inplace=True)
        production_coefficients[crop].set_index("region", inplace=True)

    # Process each crop
    for crop in crops:
        for region in regions['NUTS_ID']:
            # Extract the regional coefficient for the current crop and region
            coef_val = coefficients[crop]['mean'].loc[region]

            # If the surface coefficient is missing, fall back to production coefficient
            if pd.isna(coef_val):
                coef_val = production_coefficients[crop].loc[region, 'mean']

                if pd.isna(coef_val):
                    # No valid coefficient found, skip this region
                    continue

            # Extract the country code from the region (first two characters)
            country = region[:2]

            # For each year, update surface data in case of a gap (NaN or zero)
            for year in years:
                # Retrieve the current value for the crop and region
                current_value = areas[year].at[region, crop]

                # If the value is valid (non-null and not zero), skip to the next year
                if pd.notna(current_value) and current_value != 0:
                    continue

                # Extract the national value
                try:
                    national_value = areas_countries[year].at[country, crop]
                except (KeyError, ValueError):
                    national_value = None

                # Skip if no valid national value is found
                if national_value is None or pd.isna(national_value):
                    continue

                # Compute and assign the new value
                new_value = national_value * coef_val  # Units: kt
                areas[year].at[region, crop] = new_value
                print(f"Corrected: Year {year}, Region {region}, Crop {crop} -> New value = {new_value:.2f}")

        #%% Save surface (flag confidence: "filled (a)")

        # Create a copy of areas
        filled_a_confidence_areas = {year: df.copy() for year, df in areas.items()}
        # Crop combinations 1
        filled_a_confidence_areas = utils.merge_crops(filled_a_confidence_areas, crop_combinations_1)

        # Convert areas values to Mha
        for year in filled_a_confidence_areas:
            for crop in crops_intermediate:
                filled_a_confidence_areas[year][crop] = filled_a_confidence_areas[year][crop] / 1000  # Mha

        # Crop combinations 2
        filled_a_confidence_areas = utils.merge_crops(filled_a_confidence_areas, crop_combinations_2)

        # Delete prior crops in the current surface dataset
        for year, df in filled_a_confidence_areas.items():
            filled_a_confidence_areas[year][prior_crops] = np.nan

        # fill values and set confidence to 'filled (a)'
        final_areas = utils.fill_template(final_areas, filled_a_confidence_areas, confidence_label='filled (a)')

    # %% --- (b) Correct some cereal surface using the Eurostat 'grain cereals' surface category ---

    crop_combinations = {
        'sumcereals': {
            'cols_strict': [],  # list of columns to sum with skipna=False
            'cols_flexible': ['wheatspelt1', 'wheatspelt2', 'rye', 'Barley', 'oats1', 'oats2', 'Grain maize',
                              'triticale', 'sorghum', 'rice', 'othercereals']  # list of columns to sum with skipna=True
        }}
    areas = utils.merge_crops(areas, crop_combinations)

    # list of cereal types
    cereals = ['wheatspelt1', 'wheatspelt2', 'rye', 'Barley', 'oats1', 'oats2', 'Grain maize', 'triticale', 'sorghum',
               'rice', 'othercereals']

    # First compute the per-year ratio_dfs as before:
    ratio_dfs: dict[int, pd.DataFrame] = {}
    for year, df in areas.items():
        present = [c for c in cereals if c in df.columns]
        ratio_dfs[year] = df[present].div(df['sumcereals'], axis=0)

    # Build the final dict: one DataFrame per cereal, rows=regions, cols=years
    cereal_dfs: dict[str, pd.DataFrame] = {}

    for cereal in cereals:
        # collect a Series for each year
        series_list = []
        for year in sorted(ratio_dfs.keys()):
            df_ratios = ratio_dfs[year]
            if cereal in df_ratios.columns:
                s = df_ratios[cereal].copy()
                s.name = year
                series_list.append(s)
        if series_list:
            # concat along columns
            cereal_df = pd.concat(series_list, axis=1)
            cereal_dfs[cereal] = cereal_df

        # Compute average and standard deviation across years for each region
        year_cols = [y for y in years if y in cereal_dfs[cereal].columns]
        # extract the sub-DataFrame of just those year‐columns
        df_years = cereal_dfs[cereal][year_cols]
        # mask out zeros (turn them into NaN so that mean/std skip them)
        filtered = df_years.mask(df_years == 0)
        # now compute mean and std dev across the masked years
        cereal_dfs[cereal]['mean'] = filtered.mean(axis=1, skipna=True)
        cereal_dfs[cereal]['std_dev'] = filtered.std(axis=1, skipna=True)

    # Correct cereal values using mean cereal-to-total ratios, and print each correction
    for cereal in cereals:
        mean_series = cereal_dfs[cereal]['mean']

        for year in years:
            df = areas[year]

            for region in df.index:
                # get the current cereal value and the total-grain value
                val = df.at[region, cereal]
                grain_total = df.at[region, 'graincereals']

                # only fill if cereal is missing and grain total & mean ratio are present
                if pd.isna(val) and pd.notna(grain_total) and pd.notna(mean_series.get(region, np.nan)):
                    corrected = grain_total * mean_series[region]
                    df.at[region, cereal] = corrected
                    print(
                        f"Year {year}, Region {region}, Cereal {cereal}: "
                        f"was NaN, corrected to {corrected:.4f} "
                        f"(graincereals={grain_total:.4f} × mean_ratio={mean_series[region]:.4f})"
                    )
        areas[year] = df

    #%% Save surface (flag confidence: "filled (b)")

    # Create a copy of areas
    filled_b_confidence_areas = {year: df.copy() for year, df in areas.items()}
    # Crop combinations 1
    filled_b_confidence_areas = utils.merge_crops(filled_b_confidence_areas, crop_combinations_1)

    # Convert areas values to Mha
    for year in filled_b_confidence_areas:
        for crop in crops_intermediate:
            filled_b_confidence_areas[year][crop] = filled_b_confidence_areas[year][crop] / 1000  # Mha

    # Crop combinations 2
    filled_b_confidence_areas = utils.merge_crops(filled_b_confidence_areas, crop_combinations_2)

    # Delete prior crops in the current surface dataset
    for year, df in filled_b_confidence_areas.items():
        filled_b_confidence_areas[year][prior_crops] = np.nan

    # fill values and set confidence to 'filled (a)'
    final_areas = utils.fill_template(final_areas, filled_b_confidence_areas, confidence_label='filled (b)')

    #%% Save data after processes a and b
    for year, df in areas.items():
        df.to_csv(f'data/outputs/intermediate_datasets/surface/surface_a_b_{year}.csv', index=True)

    #%% --- (c) Interpolate + Extrapolate linearly per region + crop ---

    # Convert dict of DataFrames to a single 3D panel-like structure: (region x crop x year)
    for crop in crops:
        for region in regions['NUTS_ID']:
            # Extract time series for this (region, crop) across years
            time_series = []
            for year in years:
                df = areas[year]
                if region in df.index and crop in df.columns:
                    val = df.loc[region, crop]
                    if pd.isna(val) or val == 0:
                        time_series.append(np.nan)
                    else:
                        time_series.append(val)
                else:
                    time_series.append(np.nan)

            # Create Series with years as index
            values_series = pd.Series(time_series, index=years, dtype=float)

            # Interpolate only if at least two valid (positive) values
            if values_series.notna().sum() >= 2:
                interpolated = values_series.interpolate(method='linear', limit_direction='both')
                interpolated[interpolated < 0] = 0  # Prevent negative results

                # Update only originally missing or zero values
                for year in years:
                    original_df = areas[year]
                    if region in original_df.index and crop in original_df.columns:
                        original_value = original_df.loc[region, crop]
                        if pd.isna(original_value) or original_value == 0:
                            original_df.loc[region, crop] = interpolated.loc[year]

    #%% Save surface (flag confidence: "filled (c)")

    # Create a copy of areas
    filled_c_confidence_areas = {year: df.copy() for year, df in areas.items()}
    # Crop combinations 1
    filled_c_confidence_areas = utils.merge_crops(filled_c_confidence_areas, crop_combinations_1)

    # Convert areas values to Mha
    for year in filled_c_confidence_areas:
        for crop in crops_intermediate:
            filled_c_confidence_areas[year][crop] = filled_c_confidence_areas[year][crop] / 1000  # Mha

    # Crop combinations 2
    filled_c_confidence_areas = utils.merge_crops(filled_c_confidence_areas, crop_combinations_2)

    # Delete prior crops in the current surface dataset
    for year, df in filled_c_confidence_areas.items():
        filled_c_confidence_areas[year][prior_crops] = np.nan

    # fill values and set confidence to 'filled (a)'
    final_areas = utils.fill_template(final_areas, filled_c_confidence_areas, confidence_label='filled (c)')

    #%% --- Aggregate surface and convert to Mha ---

    # Crop combinations 1
    areas = utils.merge_crops(areas, crop_combinations_1)
    areas_countries = utils.merge_crops(areas_countries, crop_combinations_1)

    # Convert areas values to Mha
    for year in years:
        for crop in crops_intermediate:
            areas[year][crop] = areas[year][crop] / 1000  # Mha
            areas_countries[year][crop] = areas_countries[year][crop] / 1000  # Mha

    #%% --- Load and process EuropeAgriDB ---

    euragri_surf = pd.read_csv(path_euragri, sep=';')
    euragri_surf = euragri_surf.loc[euragri_surf['Symbol'] == 'A']  # Harvested area (Mha)

    # Define crops in EurAgriDB
    eur_crops = ['Barley', 'Fodder roots', 'Forage legumes', 'Grain maize', 'Green maize', 'Oilseeds', 'Other cereals',
                 'Other forage crops', 'Potatoes', 'Pulses', 'Sugar beet', 'Temporary grassland', 'Vegetables and other',
                 'Wheat', 'Olives', 'Grapes', 'Other permanent crops']

    # Process EuropeAgriDB production data for the period
    euragri = utils.process_euragri_data(years, eur_crops, euragri_surf, regions)

    # Set 'region' as production index
    for year in years:
        euragri[year] = euragri[year].set_index('region')

    # %% --- Merge crop subtypes into aggregates ---

    # Eurostat
    areas = utils.merge_crops(areas, crop_combinations_2)
    areas_countries = utils.merge_crops(areas_countries, crop_combinations_2)

    # EuropeAgriDB
    crop_combinations_eur = {
        'Fodder crops': {
            'cols_strict': [],
            'cols_flexible': ['Fodder roots', 'Green maize', 'Other forage crops']
        },
        'Grain cereals': {
            'cols_strict': ['Wheat', 'Barley', 'Grain maize', 'Other cereals'],
            'cols_flexible': []
        },
    }
    euragri = utils.merge_crops(euragri, crop_combinations_eur)

    # Final crop list
    for year in years:
        areas[year] = areas[year][final_crops]
        areas_countries[year] = areas_countries[year][final_crops]
        euragri[year] = euragri[year][final_crops]

    #%% --- Calculate a regional aggregated distribution coefficient for each country ---

    coefficients = utils.compute_regional_distribution_coefficients(areas, regions, final_crops, years)

    # Save coefficients
    for item, df in coefficients.items():
        df.to_csv(f'data/outputs/surface_regional_ratios/ratios_aggregated_{item}.csv', index=False)

    # %% --- (d) Fill gaps by multiplying Euragri national values by regional distribution coefficients ---

    # Load regional production distribution coefficients dictionary
    production_coefficients = {}
    coeff_dir = 'data/outputs/production_regional_ratios'
    for filename in sorted(os.listdir(coeff_dir)):
        if filename.startswith('ratios_aggregated_') and filename.endswith('.csv'):
            crop = filename.replace('ratios_aggregated_', '').replace('.csv', '')
            df = pd.read_csv(os.path.join(coeff_dir, filename))
            production_coefficients[crop] = df

    # Set 'region' as index for each crop or year
    for crop in final_crops:
        coefficients[crop].set_index("region", inplace=True)
        production_coefficients[crop].set_index("region", inplace=True)

    # Process each crop
    for crop in final_crops:
        for region in regions['NUTS_ID']:
            # Extract the regional coefficient for the current crop and region
            coef_val = coefficients[crop]['mean'].loc[region]

            # If the production coefficient is missing, fall back to surface coefficient
            if pd.isna(coef_val):
                coef_val = production_coefficients[crop].loc[region, 'mean']

                if pd.isna(coef_val):
                    # No valid coefficient found, skip this region
                    continue

            # Extract the country code from the region (first two characters)
            country = region[:2]

            # For each year, update production data in case of a gap (NaN or zero)
            for year in years:
                # Retrieve the current value for the crop and region
                current_value = areas[year].at[region, crop]

                # If the value is valid (non-null and not zero), skip to the next year
                if pd.notna(current_value) and current_value != 0:
                    continue

                # Extract national value from Euragri
                try:
                    national_value = euragri[year].at[country, crop]
                except (KeyError, ValueError):
                    national_value = None

                # Skip if no valid national value is found
                if national_value is None or pd.isna(national_value):
                    continue

                # Compute and assign the new value
                new_value = national_value * coef_val  # Units: Mha
                areas[year].at[region, crop] = new_value
                print(f"Corrected: Year {year}, Region {region}, Crop {crop} -> New value = {new_value:.2f}")

    #%% Save surface (flag confidence: "filled (d)")

    # Delete prior crops in the current surface dataset
    for year, df in filled_c_confidence_areas.items():
        areas[year][prior_crops] = np.nan

    final_areas = utils.fill_template(final_areas, areas, confidence_label='filled (d)')

    #%% --- (e) Correct fodder crops from EuropeAgriDB ---

    # Process each crop
    for crop in prior_crops:
        for region in regions['NUTS_ID']:
            # Extract the regional coefficient for the current crop and region
            coef_val = coefficients[crop]['mean'].loc[region]

            # If the production coefficient is missing, fall back to surface coefficient
            if pd.isna(coef_val):
                coef_val = production_coefficients[crop].loc[region, 'mean']

                if pd.isna(coef_val):
                    # No valid coefficient found, skip this region
                    continue

            # Extract the country code from the region (first two characters)
            country = region[:2]

            # For each year, update production data in case of a gap (NaN or zero)
            for year in years:
                # Retrieve the current value for the crop and region
                current_value = areas[year].at[region, crop]

                # If the value is valid (non-null and not zero), skip to the next year
                if pd.notna(current_value) and current_value != 0:
                    continue

                # Extract national value from Euragri
                try:
                    national_value = euragri[year].at[country, crop]
                except (KeyError, ValueError):
                    national_value = None

                # Skip if no valid national value is found
                if national_value is None or pd.isna(national_value):
                    continue

                # Compute and assign the new value
                new_value = national_value * coef_val  # Units: Mha
                areas[year].at[region, crop] = new_value
                print(f"Corrected: Year {year}, Region {region}, Crop {crop} -> New value = {new_value:.2f}")

    #%% Save surface (flag confidence: "corrected (e)")

    final_areas = utils.fill_template(final_areas, areas, confidence_label='corrected (e)')

    # %% --- Interpolate the last missing values linearly over time for each region + category ---

    final_crops = ['Wheat', 'Other cereals', 'Grain maize', 'Barley', 'Fodder crops', 'Oilseeds', 'Potatoes', 'Pulses',
                   'Sugar beet', 'Temporary grassland', 'Vegetables and other', 'Forage legumes', 'Olives', 'Grapes',
                   'Other permanent crops']

    # Interpolate inside gaps for each crop
    areas = utils.interpolate_dt(areas, final_crops, extrapolate=True)

    # Fill values and set confidence to 'interpolated'
    final_areas = utils.fill_template(final_areas, areas, confidence_label='interpolated')

    #%% Save the final dataset

    final_areas.to_csv('data/outputs/intermediate_datasets/final_areas.csv')

    return final_areas












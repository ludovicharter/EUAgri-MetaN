"""
Script name: e_synthetic_fertilizer.py
Description: Cleans, fills and reformats synthetic fertilizer quantity used in each territory.
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
def run_fertilizer():
    print("Running fertilizer workflow...")

    #%% --- Ignore warnings related to xlsx files ---
    warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

    #%% --- Import Eurostat data ---
    path_eurostat = 'data/Eurostat/mineral_fertilizer/aei_fm_usefert__custom_14899610_spreadsheet.xlsx'
    eurostat_fertilizer = utils.import_eurostat_sheets(path_eurostat, 1)

    # Load region ids
    regions = pd.read_csv('data/regions.csv', sep=';')
    countries = pd.read_csv('data/countries.csv', sep=';')

    # Define the period
    years = np.arange(1990, 2020)

    # Regional and national fertilizer
    fertilizer = pd.DataFrame({'region': regions['NUTS_ID']})
    fertilizer_countries = pd.DataFrame({'region': countries['NUTS_ID']})

    for year in years:
        # Default to NaN
        fertilizer[year] = np.nan
        fertilizer_countries[year] = np.nan
        # Skip if the column doesn't exist
        if year not in eurostat_fertilizer.columns:
            continue

        # Regions
        for region in regions['NUTS_ID']:
            match = eurostat_fertilizer.loc[eurostat_fertilizer['region'] == region, year]
            if not match.empty and pd.notna(match.values[0]):
                value = match.values[0]
            else:
                subregions = eurostat_fertilizer[
                    eurostat_fertilizer['region'].astype(str).str.startswith(region)
                ]
                if not subregions.empty and subregions[year].notna().all():
                    value = subregions[year].sum(skipna=True)
                else:
                    value = np.nan
            fertilizer.loc[fertilizer['region'] == region, year] = value

        # Countries
        for country in countries['NUTS_ID']:
            match = eurostat_fertilizer.loc[eurostat_fertilizer['region'] == country, year]
            if not match.empty:
                fertilizer_countries.loc[fertilizer_countries['region'] == country, year] = match.values[0]

    # Set 'region' as index
    fertilizer = fertilizer.set_index('region')
    fertilizer_countries = fertilizer_countries.set_index('region')

    # Convert fertilizer quantities to Gg N
    fertilizer = fertilizer / 1000
    fertilizer_countries = fertilizer_countries / 1000

    #%% Save fertilizer (flag confidence: high)

    # Create df product of region, fertilizer values, and year
    final_fertilizer = pd.MultiIndex.from_product(
        [sorted(regions['NUTS_ID']), years],
        names=["region", "year"]
    ).to_frame(index=False)

    # Add constant and NaN columns
    final_fertilizer["symbol"] = "Q"
    final_fertilizer["value"] = None
    final_fertilizer["label"] = "total quantity of synthetic N fertilizer applied"
    final_fertilizer["unit"] = "Gg N"
    final_fertilizer["confidence"] = None

    # Create a copy of fertilizer
    high_confidence_fertilizer = fertilizer.copy(deep=True)

    # fill values and set confidence to 'high'
    for region in high_confidence_fertilizer.index:
        for year in high_confidence_fertilizer.columns:
            value = high_confidence_fertilizer.at[region, year]

            if not pd.isna(value):
                mask = (final_fertilizer['region'] == region) & (final_fertilizer['year'] == int(year))
                final_fertilizer.loc[mask, 'value'] = value
                final_fertilizer.loc[mask, 'confidence'] = 'unprocessed'

    #%% Calculate regional repartition coefficients for fertilizer quantities

    fertilizer_long = fertilizer.reset_index().melt(id_vars='region', var_name='year', value_name='value')
    fertilizer_long['year'] = fertilizer_long['year'].astype(int)

    dfs = {
        year: group.set_index('region')[['value']]
        for year, group in fertilizer_long.groupby('year')
    }

    coefficients = utils.compute_regional_distribution_coefficients(dfs, regions, crops=None, years=years)['value']
    coefficients = coefficients.set_index('region')

    # Import other coefficients from national statistic databases

    path_other = 'data/literature_various/fertilizer_regional_repartition_coefficients.xlsx'
    other = pd.read_excel(path_other)

    target_regions = [
        "BE1", "BE2", "BE3",
        "DE1", "DE2", "DE3", "DE4", "DE5", "DE6", "DE7", "DE8", "DE9",
        "DEA", "DEB", "DEC", "DED", "DEE", "DEF", "DEG",
        "HU1", "HU2", "HU3",
        "IE04", "IE05", "IE06",
        "ITG", "ITH", "ITI", "ITC", "ITF",
        "PT11", "PT15", "PT16", "PT17", "PT18",
        "UKC", "UKD", "UKE", "UKF", "UKG", "UKH", "UKI", "UKJ", "UKK", "UKL", "UKM", "UKN"
    ]

    for region in target_regions:
        # Extract the new coefficient value for the region
        new_value_series = other.loc[other['region'] == region, 'coefficient']
        new_value = new_value_series.iloc[0]
        coefficients.at[region, 'mean'] = new_value

    #%% --- (a) Correct missing fertilizer data using available Eurostat national values ---

    # Iterate through each region
    for region in regions['NUTS_ID']:
        region = str(region)  # Ensure the region code is a string

        # Try to retrieve the regional coefficient
        try:
            coefficient = coefficients['mean'].loc[region]
        except KeyError:
            # Region not found in the coefficient table
            continue

        # Skip if the coefficient is missing
        if pd.isna(coefficient):
            continue

        # Extract the country code (first two letters)
        country = region[:2]

        # Iterate over all years
        for year in years:
            try:
                current_value = fertilizer.loc[region, year]
            except KeyError:
                # Region-year combination not found in fertilizer table
                continue

            # If the value is already valid, skip to the next year
            if pd.notna(current_value) and current_value != 0:
                continue

            # Attempt to retrieve the national-level value
            try:
                national_value = fertilizer_countries.loc[country, year]
            except (KeyError, ValueError):
                national_value = None

            # If the national value is missing, skip
            if national_value is None or pd.isna(national_value):
                continue

            # Estimate and assign the corrected value using the regional coefficient
            new_value = national_value * coefficient
            fertilizer.loc[region, year] = new_value
            print(f"Corrected: Year {year}, Region {region} -> New value = {new_value:.2f}")

    #%% Save fertilizer (flag confidence: 'filled (a)')

    # Create a copy of fertilizer
    confidence_a_fertilizer = fertilizer.copy(deep=True)

    # Loop over each row of final_fertilizer
    for i, row in final_fertilizer.iterrows():
        region = row['region']
        year = row['year']

        try:
            new_value = confidence_a_fertilizer.at[region, year]
        except KeyError:
            continue  # Skip if region or year not in fertilizer

        # Skip if value is NaN
        if pd.isna(new_value):
            continue

        current_value = row['value']

        # Replace if value is None or significantly different
        if current_value is None or not np.isclose(current_value, new_value, equal_nan=False):
            final_fertilizer.at[i, 'value'] = new_value
            final_fertilizer.at[i, 'confidence'] = 'filled (a)'

    #%% --- (b) Correct missing fertilizer data using available EuropeAgriDB and FAOSTAT national values ---

    # Load and process EuropeAgriDB
    path_euragri = 'data/EuropeAgriDB_v1.0/tables/synthetic_fertilizer.csv'
    euragri_fertilizer = pd.read_csv(path_euragri, sep=';')
    euragri_fertilizer = euragri_fertilizer.loc[euragri_fertilizer['Symbol'] == 'Q']  # Total quantity of fertilizer applied (Gg N)
    euragri_fertilizer = euragri_fertilizer.set_index('Region')  # Set region as index

    # Load and process FAOSTAT
    path_fao = 'data/FAOSTAT/mineral_fertilizer/FAOSTAT_data_en_1-8-2025.csv'
    fao_fertilizer = pd.read_csv(path_fao, sep=';')
    fao_fertilizer = fao_fertilizer.set_index('Code')  # Set region as index

    # Iterate through each region
    for region in regions['NUTS_ID']:
        region = str(region)  # Ensure the region code is a string

        # Try to retrieve the regional coefficient
        try:
            coefficient = coefficients['mean'].loc[region]
        except KeyError:
            # Region not found in the coefficient table
            continue

        # Skip if the coefficient is missing
        if pd.isna(coefficient):
            continue

        # Extract the country code (first two letters)
        country = region[:2]

        # Iterate over all years
        for year in years:
            try:
                current_value = fertilizer.loc[region, year]
            except KeyError:
                # Region-year combination not found in fertilizer table
                continue

            # If the value is already valid, skip to the next year
            if pd.notna(current_value) and current_value != 0:
                continue

            # Attempt to retrieve the national-level value from EuropeAgriDB
            try:
                national_value = euragri_fertilizer.loc[(euragri_fertilizer.index == country) &
                                                        (euragri_fertilizer['Year'] == year), 'Value'].item()
            except (KeyError, ValueError):
                national_value = None
            # If the national value is missing, try in FAOSTAT
            try:
                national_value = fao_fertilizer.loc[(fao_fertilizer.index == country) &
                                                    (fao_fertilizer['Year'] == year), 'Value'].item() / 1000  # Gg N
            except (KeyError, ValueError):
                national_value = None

            # If the national value is missing, skip
            if national_value is None or pd.isna(national_value):
                continue

            # Estimate and assign the corrected value using the regional coefficient
            new_value = national_value * coefficient
            fertilizer.loc[region, year] = new_value
            print(f"Corrected: Year {year}, Region {region} -> New value = {new_value:.2f}")

    #%% Save fertilizer (flag confidence: 'filled (b)')

    # Create a copy of fertilizer
    confidence_b_fertilizer = fertilizer.copy(deep=True)

    # Loop over each row of final_fertilizer
    for i, row in final_fertilizer.iterrows():
        region = row['region']
        year = row['year']

        try:
            new_value = confidence_b_fertilizer.at[region, year]
        except KeyError:
            continue  # Skip if region or year not in fertilizer

        # Skip if value is NaN
        if pd.isna(new_value):
            continue

        current_value = row['value']

        # Replace if value is None or significantly different
        if current_value is None or not np.isclose(current_value, new_value, equal_nan=False):
            final_fertilizer.at[i, 'value'] = new_value
            final_fertilizer.at[i, 'confidence'] = 'filled (b)'

    #%% Save fertilizer data

    final_fertilizer.to_csv('data/outputs/intermediate_datasets/final_fertilizer.csv', index=False)

    return final_fertilizer



"""
Script name: i_livestock_excretion.py
Description: Cleans, fills and reformats animal excretion for each territory.
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
import warnings
import codebase.utils as utils
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
import pickle

#%% Code function
def run_excretion():
    print("Running animal excretion workflow...")

    # %% --- Ignore warnings related to xlsx files ---
    warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

    # %% --- Define the regions ---

    regions = pd.read_csv('data/regions.csv', sep=';')

    #%% --- Import datasets ---

    path_animals = 'data/Eurostat/animals/agr_r_animal__custom_16834420_spreadsheet.xlsx'
    path_poultry = 'data/Eurostat/animals/ef_lsk_poultry__custom_16801455_spreadsheet.xlsx'
    path_fao_poultry = 'data/FAOSTAT/animals/FAOSTAT_data_en_POULTRY.xlsx'
    path_fao_other = 'data/FAOSTAT/animals/FAOSTAT_data_en_AL_NO.xlsx'
    path_excretion_coefs = 'data/literature_various/Nexcr_coefficients.csv'

    #%% --- Clean and pre-process POULTRY datasets ---

    # Map poultry sheets to sheet names
    poultry_sheets = {
        'poultry': 'Sheet 1',
        'layinghens': 'Sheet 3',
        'broilers': 'Sheet 4',
        'turkeys': 'Sheet 7',
    }

    # Initialize an animal dict
    animal_dataframes = {}
    # Initialize for each year
    years = np.arange(1990, 2023)
    for year in years:
        animal_dataframes[year] = pd.DataFrame()

    # List of years to process
    years = [2010, 2013, 2016, 2020]

    # Loop through each animal in the poultry_sheets dictionary
    for animal, sheet_name in poultry_sheets.items():
        # Read each animal sheet
        df = pd.read_excel(path_poultry, sheet_name=sheet_name, header=None, na_values=':')

        # Rename column names
        df = df.rename(columns={0: 'region', 2: 2010, 4: 2013, 6: 2016, 8: 2020})

        # Loop through each year
        for year in years:

            # List to store animal data for each region
            animal_data = []

            # Loop through each region in nuts['NUTS_ID']
            for region in regions['NUTS_ID']:
                value = df.loc[df['region'] == region, year]

                if not value.empty and not pd.isna(value.values[0]):
                    animal_data.append(value.values[0])

                else:
                    df_filtered = df[df['region'].notna()]
                    startswith_mask = df_filtered['region'].str.startswith(region)

                    if startswith_mask.any():
                        d = df_filtered[startswith_mask]
                        animal_data.append(np.nansum(d[year]))

                    else:
                        animal_data.append(np.nan)

            # Add the animal data to the DataFrame for the year
            animal_dataframes[year][animal] = animal_data

    # Add region names to each DataFrame
    for year in animal_dataframes:
        animal_dataframes[year]['region'] = regions['NUTS_ID']

    for year in years:
        # Calculate 'otherpoultry' by subtracting the sum of 'layinghens', 'broilers', and 'turkeys' from 'poultry'
        animal_dataframes[year]['otherpoultry'] = (
                animal_dataframes[year]['poultry'] -
                animal_dataframes[year]['layinghens'].fillna(0) -
                animal_dataframes[year]['broilers'].fillna(0) -
                animal_dataframes[year]['turkeys'].fillna(0)
        )

        # Replace any unrealistic negative values in 'otherpoultry' with 0
        animal_dataframes[year].loc[animal_dataframes[year]['otherpoultry'] < 0, 'otherpoultry'] = 0

    #%% --- Calculation of POULTRY regional repartition coefficients ---

    # Initialize dictionary to store average ratios per animal
    average_ratios = {
        'layinghens': [],
        'broilers': [],
        'turkeys': [],
        'otherpoultry': []
    }

    # Loop through regions
    for region in regions['NUTS_ID']:
        country = region[:2]
        region_ratios = {animal: [] for animal in average_ratios.keys()}

        # Special case: if region is a country code (2 letters), assign 1 to all animals
        if len(region) == 2 and region.isalpha():
            for animal in average_ratios.keys():
                average_ratios[animal].append(1.0)
            continue  # Skip rest of loop

        for year in years:
            df_year = animal_dataframes[year]

            # Subset for regions of this country
            country_mask = df_year['region'].str.startswith(country)
            df_country = df_year.loc[country_mask].copy()
            expected_regions = [r for r in regions['NUTS_ID'] if r.startswith(country)]

            for animal in average_ratios.keys():
                animal_values = df_country.set_index('region')[animal]

                # 1st condition: skip year if any region has NaN
                if animal_values.isna().any():
                    ratio = np.nan
                else:
                    # 2nd condition: skip year if >50% of regions have 0
                    total_regions = len(expected_regions)
                    zero_count = (animal_values.loc[expected_regions] == 0).sum()
                    if zero_count > (total_regions / 2):
                        ratio = np.nan
                    else:
                        total_animal = animal_values.sum()
                        region_value = animal_values.get(region, np.nan)
                        if pd.notna(region_value) and total_animal != 0:
                            ratio = region_value / total_animal
                        else:
                            ratio = np.nan

                region_ratios[animal].append(ratio)

        # After looping through all years, compute average ratio
        for animal in average_ratios.keys():
            average_ratios[animal].append(np.nanmean(region_ratios[animal]))

    # Create final DataFrame
    average_ratios_df = pd.DataFrame(average_ratios)
    average_ratios_df['region'] = regions['NUTS_ID'].values
    average_ratios_df['country'] = average_ratios_df['region'].str[:2]

    # Normalize to ensure sum of ratios per country equals 1
    for animal in average_ratios.keys():
        average_ratios_df[animal] = (
            average_ratios_df
            .groupby('country')[animal]
            .transform(lambda grp: grp / grp.sum() if grp.sum() > 0 else grp)
        )

    #%% Fill gaps with FAOstat data

    # Read FAOstat poultry dataset
    fao = pd.read_excel(path_fao_poultry)

    # Initialize dict
    fao_dfs = {}

    # Initialize for each year
    years = np.arange(1990, 2023)
    for year in years:
        # Initialize DataFrame for the current year
        fao_dfs[year] = pd.DataFrame(index=regions['NUTS_ID'], columns=['turkeys', 'ducks', 'chickens'])

        # Loop through each region
        for region in regions['NUTS_ID']:
            country = region[:2]  # Extract country code from region

            # Turkeys
            condition = (fao['Year Code'] == year) & (fao['Country'] == country) & (fao['Item'] == 'Turkeys')
            ratio = average_ratios_df['turkeys'].loc[average_ratios_df['region'] == region].values[0]
            # Check if condition is met and assign value
            if not fao.loc[condition, 'Value'].empty:
                fao_dfs[year].loc[region, 'turkeys'] = fao.loc[condition, 'Value'].values[0] * ratio * 1000
            else:
                fao_dfs[year].loc[region, 'turkeys'] = np.nan

            # Ducks
            condition = (fao['Year Code'] == year) & (fao['Country'] == country) & (fao['Item'] == 'Ducks')
            ratio = average_ratios_df['otherpoultry'].loc[average_ratios_df['region'] == region].values[0]
            # Check if condition is met and assign value
            if not fao.loc[condition, 'Value'].empty:
                fao_dfs[year].loc[region, 'ducks'] = fao.loc[condition, 'Value'].values[0] * ratio * 1000
            else:
                fao_dfs[year].loc[region, 'ducks'] = np.nan

            # Chickens
            condition = (fao['Year Code'] == year) & (fao['Country'] == country) & (fao['Item'] == 'Chickens')
            ratio = average_ratios_df['layinghens'].loc[average_ratios_df['region'] == region].values[0]
            # Check if condition is met and assign value
            if not fao.loc[condition, 'Value'].empty:
                fao_dfs[year].loc[region, 'chickens'] = fao.loc[condition, 'Value'].values[0] * ratio * 1000
            else:
                fao_dfs[year].loc[region, 'chickens'] = np.nan

    # Combine FAO data to the main animal dict
    years = np.arange(1990, 2020)  # Define the period
    for year in years:
        animal_dataframes[year] = fao_dfs[year]
        # Change index name
        animal_dataframes[year].index.name = 'region'

    #%% --- Read other ANIMAL files from Eurostat ---

    animal_sheets = {'bovine': 'Sheet 1',  # Remove dairy cows
                     'dairycows': 'Sheet 19',
                     'pigs': 'Sheet 24',
                     'sheep': 'Sheet 38',
                     'goats': 'Sheet 41'
                     }

    # Clean each animal dataset
    cleaned_datasets = {}
    for animal, sheet in animal_sheets.items():
        df = pd.read_excel(path_animals, sheet_name=sheet, header=None, na_values=':')
        cleaned_datasets[animal] = utils.clean_animal_eurostat(df, regions)

    #%% --- Extract other ANIMAL Eurostat data by years ---

    # Initialize a dictionary to store DataFrames by year
    dataframes_by_year = {}

    # Define the period
    years = cleaned_datasets['bovine'].columns[1:]  # Skip the first column which is 'Region'

    # Iterate over each year and create a DataFrame for that year
    for year in years:
        df_year = pd.DataFrame({'region': cleaned_datasets['bovine'].iloc[:, 0]})

        # Add animal data for that year to the DataFrame
        for animal, df in cleaned_datasets.items():
            df_year[animal] = df.loc[:, year].values

        # Store the DataFrame in the dictionary with the year as the key
        dataframes_by_year[int(year)] = df_year

    #%% --- Merge all animals ---

    years = np.arange(1990, 2020)
    for year in years:
        animal_dataframes[year] = pd.merge(animal_dataframes[year], dataframes_by_year[year], on='region', how='outer')

        # Convert poultry data into 1000 units
        animal_dataframes[year]['chickens'] = animal_dataframes[year]['chickens'] / 1000
        animal_dataframes[year]['ducks'] = animal_dataframes[year]['ducks'] / 1000
        animal_dataframes[year]['turkeys'] = animal_dataframes[year]['turkeys'] / 1000

        # Remove dairy cows from bovine
        diff = animal_dataframes[year]['bovine'] - animal_dataframes[year]['dairycows']
        animal_dataframes[year]['bovine'] = diff.mask(diff < 0, 0)  # replaces negative entries with 0

    # Save animals
    with open("data/outputs/intermediate_datasets/animals_dict.pkl", "wb") as f:
        pickle.dump(animal_dataframes, f)

    # %% --- Define a new dataset to store animal excretions ---

    # Create a df template
    animals = ['turkeys', 'ducks', 'chickens', 'bovine', 'dairy', 'pigs', 'sheep', 'goats']

    # Create df product of region, crop, and year
    final_excretion = pd.MultiIndex.from_product(
        [sorted(regions['NUTS_ID']), sorted(animals), years],
        names=["region", "animal", "year"]
    ).to_frame(index=False)

    # Map region names from the regions DataFrame
    region_name_map = regions.set_index('NUTS_ID')['name']
    region_names = final_excretion['region'].map(region_name_map)
    # Insert 'region name' as the second column
    final_excretion.insert(
        loc=1,
        column='region name',
        value=region_names
    )

    # Add metadata columns with default values
    final_excretion = final_excretion.assign(
        value=None,
        label="excretion",
        unit="Gg N",
        confidence=None
    )

    #%% --- Save excretion (flag confidence: high) ---

    # Check if animal_dataframes isin the defined period
    for year in list(animal_dataframes.keys()):
        if year not in years:
            # Remove invalid year
            animal_dataframes.pop(year)

    # Calculate excretion
    high_confidence_excretion = utils.calculate_excretion(animal_dataframes, path_excretion_coefs)

    # fill values and set confidence to 'high'
    final_excretion = utils.fill_template(final_excretion, high_confidence_excretion, id_col='animal', confidence_label='unprocessed')

    #%% --- (a) Fill country gaps (AL and NO) with FAOSTAT ---

    # List of  missing countries
    missing_countries = ['NO', 'AL']

    # Read fao file
    fao_dfs = pd.read_excel(path_fao_other, skiprows=1)

    # Iterate over missing countries and years to update animal dataframes
    for country in missing_countries:
        for year in animal_dataframes:

            # Bovine and dairy cows
            value = fao_dfs.loc[(fao_dfs['NUTS_ID'] == country) &
                                (fao_dfs['Year'] == year) &
                                (fao_dfs['Item'] == 'Cattle'), 'Value']
            # Update animal dataframes
            animal_dataframes[year].loc[animal_dataframes[year]['region'] == country, 'bovine'] = (value.iloc[0] / 2) / 1000
            animal_dataframes[year].loc[animal_dataframes[year]['region'] == country, 'dairycows'] = (value.iloc[0] / 2) / 1000

            # Pigs
            value = fao_dfs.loc[(fao_dfs['NUTS_ID'] == country) &
                                (fao_dfs['Year'] == year) &
                                (fao_dfs['Item'] == 'Swine / pigs'), 'Value']
            # Update animal dataframes
            animal_dataframes[year].loc[animal_dataframes[year]['region'] == country, 'pigs'] = value.iloc[0] / 1000

            # Sheep
            value = fao_dfs.loc[(fao_dfs['NUTS_ID'] == country) &
                                (fao_dfs['Year'] == year) &
                                (fao_dfs['Item'] == 'Sheep'), 'Value']
            # Update animal dataframes
            animal_dataframes[year].loc[animal_dataframes[year]['region'] == country, 'sheep'] = value.iloc[0] / 1000

            # Goats
            value = fao_dfs.loc[(fao_dfs['NUTS_ID'] == country) &
                                (fao_dfs['Year'] == year) &
                                (fao_dfs['Item'] == 'Goats'), 'Value']
            # Update animal dataframes
            animal_dataframes[year].loc[animal_dataframes[year]['region'] == country, 'goats'] = value.iloc[0] / 1000

    #%% --- Save excretion (flag confidence: 'filled (a)') ---

    # Calculate excretion
    a_confidence_excretion = utils.calculate_excretion(animal_dataframes, path_excretion_coefs)

    # fill values and set confidence to 'filled (a)'
    final_excretion = utils.fill_template(final_excretion, a_confidence_excretion, id_col='animal', confidence_label='filled (a)')

    # Set confidence to 'filled (a)' for poultry corrected before using FAOSTAT
    target_animals = ['chickens', 'ducks', 'turkeys']
    # Apply condition
    mask = final_excretion['animal'].isin(target_animals) & final_excretion['value'].notna()
    final_excretion.loc[mask, 'confidence'] = 'filled (a)'

    # %% --- Interpolate + Extrapolate linearly per region + animal ---

    for animal in animals:
        for region in regions['NUTS_ID']:
            # Mask for the current region and animal
            mask_rows = (final_excretion['region'] == region) & (final_excretion['animal'] == animal)

            # Get a copy of the Series to interpolate
            values_series = final_excretion.loc[mask_rows, 'value'].copy()

            # Convert None to NaN explicitly if needed
            values_series = values_series.astype(float)

            # Count valid (non-NaN) entries
            if values_series.notna().sum() >= 2:
                # Interpolate linearly with extrapolation
                interpolated = values_series.interpolate(method='linear', limit_direction='both')

                # Ensure no negative values
                interpolated[interpolated < 0] = 0

                # Detect which values were originally NaN
                interpolated_indices = values_series.isna()

                # Update the DataFrame
                final_excretion.loc[mask_rows, 'value'] = interpolated

                # Update the 'confidence' column for interpolated values
                confidence = final_excretion.loc[mask_rows, 'confidence'].copy()
                confidence[interpolated_indices] = 'interpolated'
                final_excretion.loc[mask_rows, 'confidence'] = confidence
            else:
                # Not enough data to interpolate
                continue

    #%%
    return final_excretion













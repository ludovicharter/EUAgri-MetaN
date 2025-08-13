"""
Script name: k_arable_budget.py
Description: Compiles N budget in arable land for each territory.
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
import matplotlib.pyplot as plt

#%% --- Code function ---
def run_budget():
    print("Running cropland budget workflow...")

    #%% --- Import datasets ---

    land_areas = pd.read_csv('data/outputs/land_areas.csv')
    bnf = pd.read_csv('data/outputs/symbiotic_fixation.csv')
    atmospheric_deposition = pd.read_csv('data/outputs/atmospheric_deposition_rate.csv')
    synthetic_fertilization = pd.read_csv('data/outputs/synthetic_fertilizer.csv')
    crop_production = pd.read_csv('data/outputs/crop_production_all_categories.csv')
    manure = pd.read_csv('data/outputs/manure_allocation.csv')

    # Regions
    regions = pd.read_csv('data/regions.csv', sep=';')

    # Define the period
    years = np.arange(1990, 2020)

    # %% --- Create the final DF ---

    symbols = [
        'A', 'B', 'D', 'F', 'H', 'M', 'Vm', 'Vf', 'NUE', 'NS'
    ]

    label_map = {
        'A': 'arable area sum',
        'B': 'symbiotic fixation in arable',
        'D': 'atmospheric deposition to arable',
        'F': 'synthetic fertilizer to arable',
        'H': 'arable harvest',
        'M': 'manure to arable',
        'Vm': 'N volatilization at manure application',
        'Vf': 'N volatilization at fertilizer application',
        'NUE': 'N use efficiency',
        'NS': 'N surplus'
    }

    unit_map = {
        'A': 'Mha',
        'B': 'Gg N',
        'D': 'Gg N',
        'F': 'Gg N',
        'H': 'Gg N',
        'M': 'Gg N',
        'Vm': 'Gg N',
        'Vf': 'Gg N',
        'NUE': 'no unit',
        'NS': 'kgN/ha'
    }

    # Cartesian product of regions × symbols × years
    final_budget = pd.MultiIndex.from_product(
        [sorted(regions['NUTS_ID']), symbols, sorted(years)],
        names=["region", "symbol", "year"]
    ).to_frame(index=False)

    # Map region codes to region names
    region_name_map = regions.set_index('NUTS_ID')['name']
    final_budget.insert(
        loc=1,
        column='region_name',
        value=final_budget['region'].map(region_name_map)
    )

    # Initialize metadata columns
    final_budget = final_budget.assign(
        value=np.nan,
        label=final_budget['symbol'].map(label_map),
        unit=final_budget['symbol'].map(unit_map),
        confidence=np.nan  # new confidence column, to be filled later
    )

    # Enforce the per-(region, year) symbol ordering
    final_budget['symbol'] = pd.Categorical(
        final_budget['symbol'],
        categories=symbols,
        ordered=True
    )

    # Sort and reset index
    final_budget = (
        final_budget
        .sort_values(['region', 'year', 'symbol'])
        .reset_index(drop=True)
    )

    #%% ARABLE LAND
    #%% --- Process crop area sum (A) ---

    # Filter only the 2 symbols and keep region/year/value (AL, PC)
    crop_areas = (
        land_areas
        .loc[
            land_areas['symbol'].isin(['AL_sum']),
            ['region', 'year', 'value']
        ]
        # Group by region & year, summing their values
        .groupby(['region', 'year'])['value']
        .sum()
        .reset_index(name='cropland')
    )

    # Add to final_budget
    # Merge the summed values into final_budget
    final_budget = final_budget.merge(
        crop_areas,
        on=['region', 'year'],
        how='left'
    )

    # Assign to the new symbol 'A' (if that’s your target)
    mask_a = final_budget['symbol'] == 'A'
    final_budget.loc[mask_a, 'value'] = final_budget.loc[mask_a, 'cropland']

    # Drop the helper column
    final_budget.drop(columns='cropland', inplace=True)

    # Crop areas quality control

    # 1. Make a copy and sort by region and year
    df_qc = crop_areas.copy()
    df_qc = df_qc.sort_values(['region', 'year'])

    # 2. Compute the cropland value for the previous and next year within each region
    df_qc['prev_cropland'] = df_qc.groupby('region')['cropland'].shift(1)
    df_qc['next_cropland'] = df_qc.groupby('region')['cropland'].shift(-1)

    # 3. Define a function to calculate relative change (with protection against division by zero)
    def rel_change(current, other):
        if pd.isna(other) or other == 0:
            return np.nan
        return (current - other) / other

    # 4. Apply the relative change calculation compared to the previous and next year
    df_qc['change_prev'] = df_qc.apply(
        lambda row: rel_change(row['cropland'], row['prev_cropland']),
        axis=1
    )
    df_qc['change_next'] = df_qc.apply(
        lambda row: rel_change(row['cropland'], row['next_cropland']),
        axis=1
    )

    # 5. Filter anomalies: absolute change > 200% (i.e., |change| > 2)
    anomalies = df_qc[
        (df_qc['change_prev'].abs() > 1.1) |
        (df_qc['change_next'].abs() > 1.1)
        ]

    # 6. Print the problematic values with context
    print("=== Cropland values with >200% deviation compared to previous or next year ===\n")
    print(
        anomalies[
            [
                'region',
                'year',
                'cropland',
                'prev_cropland',
                'change_prev',
                'next_cropland',
                'change_next'
            ]
        ]
        .sort_values(['region', 'year'])
        .to_string(index=False)
    )

    #%% --- Process BNF (B) ---

    # Sum all bnf types by year and region
    fixation_sums = bnf.groupby(['region', 'year'])['value'].sum().reset_index()

    # Rename the 'value' column in fixation_sums
    fixation = fixation_sums.rename(columns={'value': 'B_value'})

    # Merge the B‐values into final_budget
    final_budget = final_budget.merge(
        fixation,
        on=['region', 'year'],
        how='left'
    )

    # Overwrite final_budget.value for symbol 'B' with the merged B_value
    mask_b = final_budget['symbol'] == 'B'
    final_budget.loc[mask_b, 'value'] = final_budget.loc[mask_b, 'B_value']

    # Drop the helper column
    final_budget.drop(columns='B_value', inplace=True)

    #%% --- Process atmospheric deposition (D) ---

    # Compute the N-deposition quantity D_value = value * A_value
    dep = (
        atmospheric_deposition
        # bring in A_value for each region/year
        .merge(crop_areas, on=['region', 'year'], how='left')
        # create the helper column D_value
        .assign(D_value=lambda df: df['value'] * df['cropland'])
        # keep only the keys + D_value
        [['region', 'year', 'D_value']]
    )

    # Merge D_value into final_budget
    final_budget = final_budget.merge(
        dep,
        on=['region', 'year'],
        how='left'
    )

    # Overwrite final_budget.value only for symbol 'D'
    is_d = final_budget['symbol'] == 'D'
    final_budget.loc[is_d, 'value'] = final_budget.loc[is_d, 'D_value']

    # Drop the temporary helper column
    final_budget.drop(columns='D_value', inplace=True)

    #%% --- Process synthetic fertilizer (F) ---

    # Filter only the 2 symbols and keep region/year/value (Q_AL, Q_PC)
    crop_fert = (
        synthetic_fertilization
        .loc[
            synthetic_fertilization['symbol'].isin(['Q_AL']),
            ['region', 'year', 'value']
        ]
        # Group by region & year, summing their values
        .groupby(['region', 'year'])['value']
        .sum()
        .reset_index(name='cropland')
    )

    # Add to final_budget
    # Merge the summed values into final_budget
    final_budget = final_budget.merge(
        crop_fert,
        on=['region', 'year'],
        how='left'
    )

    # Assign to the new symbol 'F' (if that’s your target)
    mask_f = final_budget['symbol'] == 'F'
    final_budget.loc[mask_f, 'value'] = final_budget.loc[mask_f, 'cropland']

    # Drop the helper column
    final_budget.drop(columns='cropland', inplace=True)

    #%% --- Process cropland harvest (H) ---

    crops = [
        'Wheat', 'Other cereals', 'Grain maize', 'Barley',
        'Fodder crops', 'Oilseeds', 'Potatoes', 'Pulses',
        'Sugar beet', 'Temporary grassland', 'Vegetables and other',
        'Forage legumes'
    ]

    # Filter only harvested in crop production file
    crop_harvest = crop_production.loc[crop_production['symbol'] == 'H']

    # Sum cropland harvested production
    crop_harvest = (
        crop_harvest
        .loc[
            crop_harvest['crop'].isin(crops),
            ['region', 'year', 'value']
        ]
        # Group by region & year, summing their values
        .groupby(['region', 'year'])['value']
        .sum()
        .reset_index(name='cropland')
    )

    # Add to final_budget
    # Merge the summed values into final_budget
    final_budget = final_budget.merge(
        crop_harvest,
        on=['region', 'year'],
        how='left'
    )

    # Assign to the new symbol 'H' (if that’s your target)
    mask_h = final_budget['symbol'] == 'H'
    final_budget.loc[mask_h, 'value'] = final_budget.loc[mask_h, 'cropland']

    # Drop the helper column
    final_budget.drop(columns='cropland', inplace=True)

    #%% --- Process manure to cropland (M) ---

    # Sum manure applied to cropland
    manure_to_cropland = (
        manure
        .loc[
            manure['symbol'].isin(['A_TG', 'A_AL']),
            ['region', 'year', 'value']
        ]
        # Group by region & year, summing their values
        .groupby(['region', 'year'])['value']
        .sum()
        .reset_index(name='cropland')
    )

    # Add to final_budget
    # Merge the summed values into final_budget
    final_budget = final_budget.merge(
        manure_to_cropland,
        on=['region', 'year'],
        how='left'
    )

    # Assign to the new symbol 'M' (if that’s your target)
    mask_m = final_budget['symbol'] == 'M'
    final_budget.loc[mask_m, 'value'] = final_budget.loc[mask_m, 'cropland']

    # Drop the helper column
    final_budget.drop(columns='cropland', inplace=True)

    #%% --- Synthetic fertilizer correction ---

    # Step 1 — Pivot final_budget to wide format
    budget_wide = (
        final_budget
        .pivot_table(index=['region', 'year'], columns='symbol', values='value')
        .copy()
    )

    # Step 2 — Compute total N input and NUE
    budget_wide['N_input'] = budget_wide[['B', 'D', 'F', 'M']].sum(axis=1)
    budget_wide['NUE'] = budget_wide['H'] / budget_wide['N_input']

    # Step 3 — Set target NUE
    target_NUE = 0.9

    # Step 4 — Identify region-years where NUE > 1
    to_correct = budget_wide['NUE'] > 1

    # Step 5 — Loop over rows to correct F and print each correction
    print("Corrections applied:\n")
    for idx in budget_wide[to_correct].index:
        region, year = idx
        row = budget_wide.loc[idx]

        current_NUE = row['NUE']
        original_F = row['F']
        other_inputs = row[['B', 'D', 'M']].sum()

        # Corrected F to reach target NUE
        corrected_F = (row['H'] / target_NUE) - other_inputs

        # Update the fertilizer value
        budget_wide.at[idx, 'F'] = corrected_F

        # Print the correction details
        print(f"Region: {region}, Year: {year}")
        print(f"  Original NUE: {current_NUE:.2f}")
        print(f"  Original F:   {original_F:.2f} Gg N")
        print(f"  Corrected F:  {corrected_F:.2f} Gg N (Target NUE: {target_NUE})\n")

    # Step 6 — Melt back to long format
    corrected_long = (
        budget_wide
        .drop(columns=['N_input', 'NUE'])
        .reset_index()
        .melt(id_vars=['region', 'year'], var_name='symbol', value_name='value')
    )

    # Step 7 — Merge corrected F values into final_budget
    final_budget = final_budget.merge(
        corrected_long[corrected_long['symbol'] == 'F'],
        on=['region', 'year', 'symbol'],
        how='left',
        suffixes=('', '_new')
    )

    # Step 8 — Replace original F with corrected F where it has changed
    value_changed = final_budget['value_new'].notna() & (final_budget['value'] != final_budget['value_new'])
    final_budget.loc[value_changed, 'value'] = final_budget.loc[value_changed, 'value_new']
    final_budget.drop(columns='value_new', inplace=True)

    # Step 9 — Update confidence only if value was actually corrected
    final_budget.loc[
        (final_budget['symbol'] == 'F') &
        (final_budget['confidence'].isna()) &
        value_changed,
        'confidence'
    ] = 'corrected (f)'

    #%% --- N volatilization at manure and fertilizer application ---

    final_budget.loc[final_budget['symbol'] == 'Vm', 'value'] = (
            final_budget.loc[final_budget['symbol'] == 'M', 'value'].values * 0.21
    )
    final_budget.loc[final_budget['symbol'] == 'Vf', 'value'] = (
            final_budget.loc[final_budget['symbol'] == 'F', 'value'].values * 0.11
    )

    #%% --- NUE and NS ---

    final_budget.loc[final_budget['symbol'] == 'NUE', 'value'] = (
            final_budget.loc[final_budget['symbol'] == 'H', 'value'].values /
            (
                    final_budget.loc[final_budget['symbol'] == 'F', 'value'].values +
                    final_budget.loc[final_budget['symbol'] == 'M', 'value'].values +
                    final_budget.loc[final_budget['symbol'] == 'B', 'value'].values +
                    final_budget.loc[final_budget['symbol'] == 'D', 'value'].values
            )
    )

    final_budget.loc[final_budget['symbol'] == 'NS', 'value'] = (
                                                                    (
                                                                        (final_budget.loc[final_budget[
                                                                                              'symbol'] == 'F', 'value'].values * 0.89) +
                                                                        (final_budget.loc[final_budget[
                                                                                              'symbol'] == 'M', 'value'].values * 0.79) +
                                                                        final_budget.loc[final_budget[
                                                                                             'symbol'] == 'B', 'value'].values +
                                                                        final_budget.loc[final_budget[
                                                                                             'symbol'] == 'D', 'value'].values
                                                                    ) - final_budget.loc[final_budget[
                                                                                            'symbol'] == 'H', 'value'].values
                                                                ) / final_budget.loc[
                                                                    final_budget['symbol'] == 'A', 'value'].values

    #%%
    return final_budget


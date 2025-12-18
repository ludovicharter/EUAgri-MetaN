"""
Script name: Figure_1_trends_in_Europe.py
Description: Figure 1 workflow
Author: Ludovic Harter
Created: 2025-12-18
Last modified: 2025-12-18
Version: 1.0
Project: Territorial nitrogen flows and metabolic typologies of EU Agri-Food Systems, 1990–2019
License: MIT
"""

#%% --- Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% --- Import data ---
# Import CSV files into DataFrames
crop_production = pd.read_csv('data/outputs/crop_production_all_categories.csv')  # All crop production categories
budget = pd.read_csv('data/outputs/arable_budget.csv')  # Cropland nitrogen budget
metabolic = pd.read_csv('data/outputs/metabolic_data.csv')  # Livestock typology and nitrogen flows
# Animal excretion and land area datasets
animal_excretion = pd.read_csv('data/outputs/animal_excretion.csv')
land_areas = pd.read_csv('data/outputs/land_areas.csv')

#%% --- Parameters ---
# Define the range of years for analysis
years = np.arange(1990, 2020, dtype=np.int16)
# List of final crop categories to include
final_crops = [
    'Wheat', 'Other cereals', 'Grain maize', 'Barley', 'Fodder crops', 'Oilseeds',
    'Potatoes', 'Pulses', 'Sugar beet', 'Temporary grassland', 'Vegetables and other', 'Forage legumes'
]
# Animal production symbols for filtering
animal_prod = [
    'E_total', 'A_non_edible', 'Eggs', 'Milk', 'Meat_bovine', 'Meat_sheep',
    'Meat_goats', 'Meat_pigs', 'Meat_poultry'
]
# Feed symbols: A = Domestic, I = Imported
feed_syms = ['A', 'I']
# Land use symbols to include
lands = ['AL_sum', 'PC_sum', 'PG']

#%% --- General pivot helper ---
def pivot_series(df, index='year', cols='symbol', values='value', filter_query=None):
    """
    Create a time series pivot table from the given DataFrame.

    Parameters:
    df : DataFrame to pivot
    index : column to use as the index (default 'year')
    cols : column to pivot into new columns (default 'symbol')
    values : column to aggregate (default 'value')
    filter_query : optional pandas query string to filter the DataFrame before pivot

    Returns:
    DataFrame reindexed to the full year range with missing values filled as 0
    """
    df_filtered = df.query(filter_query) if filter_query is not None else df
    table = (
        df_filtered
        .pivot_table(index=index, columns=cols, values=values, aggfunc='sum')
        .reindex(years, fill_value=0)
    )
    return table

#%% --- Prepare data series ---
# 1) Crop harvest quantities
harvest = crop_production.query("label == 'harvested quantity' and crop in @final_crops")
df_crops = pivot_series(harvest, cols='crop')
# 2) Crop harvest areas
area = crop_production.query("label == 'harvested area' and crop in @final_crops")
df_area = pivot_series(area, cols='crop')

# Fertilization budget categories
fert_labels = [
    'atmospheric deposition to arable', 'manure to arable', 'symbiotic fixation in arable', 'synthetic fertilizer to arable'
]
bud = budget.query('label in @fert_labels')
df_ferti = (
    bud
    .pivot_table(index='year', columns='label', values='value', aggfunc='sum')
    .reindex(years, fill_value=0)
)
# Apply volatilization during application factors for manure and synthetic fertilizer
df_ferti['manure to arable'] *= 0.79
df_ferti['synthetic fertilizer to arable'] *= 0.89

# Animal production series
df_animals = pivot_series(metabolic.query('symbol in @animal_prod'), cols='symbol')

# Land use evolution
df_land = pivot_series(land_areas.query('symbol in @lands'), cols='symbol')

# Yield calculation (kg N per ha)
yield_N = df_crops.div(df_area)

#%% --- Compute aggregated arable yield (sum H / sum A) ---

# Sum harvested quantities and areas per region/year
H_df = crop_production.query("symbol == 'H' and crop in @final_crops")
A_df = crop_production.query("symbol == 'A' and crop in @final_crops")

H_sum = H_df.groupby(['region', 'year'])['value'].sum()
A_sum = A_df.groupby(['region', 'year'])['value'].sum()

# Compute aggregated yield
aggregated_yield = (H_sum / A_sum).reset_index()
aggregated_yield.rename(columns={'value': 'aggregated_yield'}, inplace=True)

# Pivot to have regions as columns
df_agg_yield = aggregated_yield.pivot(index='year', columns='region', values='aggregated_yield')

#%% --- Feed imports (sum) at the EU scale ---

EU_metabolic = (
    metabolic
    .pivot_table(index='year', columns='label', values='value', aggfunc='sum')
    .reindex(years, fill_value=0)
)

imported_feed = EU_metabolic['net import feed']
domestic_feed = EU_metabolic['total animal ingestion'] - imported_feed

df_feed = pd.DataFrame({
    'domestic': domestic_feed,
    'imported': imported_feed
}, index=years)

#%% --- Plot generation ---
# Create a 2x2 subplot grid with wider horizontal size
fig, axs = plt.subplots(2, 2, figsize=(24, 12))
for ax in axs.flatten():
    ax.set_xlim(1990, 2019)

# 1) Arable harvests and fertilization plot - convert units to Tg (divide by 1000)
crop_colors = [
    "#66c2a5", "#f6ed3d", "#a89cc9", "#f46d43", "#73a8d3",
    "#fdae61", "#addd8e", "#f781bf", "#bdbdbd", "#b569b5",
    "#a6dba0", "#ffe34f"
]
fertilizer_darker_colors = [
    "#016953", "#6b3a0e", "#4a237d", "#7a2c71"
]

# Divide fertilization and harvest data by 1000 for Tg units
fert_data_tg = (df_ferti[['atmospheric deposition to arable','manure to arable',
                          'symbiotic fixation in arable','synthetic fertilizer to arable']] / 1000).T.values
crop_data_tg = -(df_crops / 1000).T.values  # negative values

fert_stack = axs[0,0].stackplot(
    df_ferti.index,
    fert_data_tg,
    labels=['Atmospheric deposition', 'Manure', 'Biologic fixation', 'Synthetic fertilizer'],
    colors=fertilizer_darker_colors
)

crop_stack = axs[0,0].stackplot(
    df_crops.index,
    crop_data_tg,
    labels=df_crops.columns,
    colors=crop_colors
)

axs[0,0].axhline(0, color='k')
axs[0,0].set_title('a) Arable harvests and fertilization (Europe)', fontsize=18, fontweight='bold')
axs[0,0].set_ylabel('Tg N', fontsize=18)  # Updated unit

legend1 = axs[0,0].legend(handles=crop_stack, loc='upper left', bbox_to_anchor=(1.05, 0.7), fontsize=16, title='Harvest')
legend1.get_title().set_fontsize(16)
legend1.get_title().set_fontweight('bold')
axs[0,0].add_artist(legend1)
legend2 = axs[0,0].legend(handles=fert_stack[::-1], loc='lower left', bbox_to_anchor=(1.05, 0.7), fontsize=16, title='Fertilization')
legend2.get_title().set_fontsize(16)
legend2.get_title().set_fontweight('bold')

# N surplus calculation also adjusted to Tg
total_fert_tg = df_ferti[['atmospheric deposition to arable', 'manure to arable',
                         'symbiotic fixation in arable', 'synthetic fertilizer to arable']].sum(axis=1) / 1000
total_harvest_tg = df_crops.sum(axis=1) / 1000
n_surplus_tg = total_fert_tg - total_harvest_tg

# Plot N surplus line in Tg
axs[0, 0].plot(df_ferti.index, total_fert_tg - n_surplus_tg, color='red', linewidth=2, label='N surplus')

x_pos = 1995
y_start = 11  # Adjusted values for Tg scale
y_end = 16.5

axs[0,0].annotate(
    '',
    xy=(x_pos, y_end), xytext=(x_pos, y_start),
    arrowprops=dict(arrowstyle='<->', color='red', lw=3)
)
axs[0,0].text(
    x_pos + 0.5, (y_start + y_end)/2, 'N surplus',
    color='red', fontsize=16, fontweight='bold', rotation=0, va='center'
)

# 2) Livestock production and feed ingestion - convert units to Tg (divide by 1000)
animal_colors = [
    "#b3b3b3", "#a67c52", "#f4d06f", "#7da0b1", "#8ab58a",
    "#e89cae", "#f2a65a", "#c0a6c1", "#b2d9e7"
]
feed_colors = [
    "#99c99c", "#a69acb"
]

prod_data_tg = -(df_animals / 1000).T.values
feed_data_tg = df_feed[['domestic', 'imported']].T.values / 1000

prod_stack = axs[0,1].stackplot(
    df_animals.index,
    prod_data_tg,
    labels=['Non-edible', 'Total excretion', 'Eggs', 'Bovine', 'Goats', 'Pigs', 'Poultry', 'Sheep', 'Cows milk'],
    colors=animal_colors
)

feed_stack = axs[0,1].stackplot(
    df_feed.index,
    feed_data_tg,
    labels=['Local feed', 'Imported feed'],
    colors=feed_colors
)

axs[0,1].axhline(0, color='k')
axs[0,1].set_title('b) Livestock production and ingestion (Europe)', fontsize=18, fontweight='bold')
axs[0,1].set_ylabel('Tg N', fontsize=18)  # Updated unit

legend1 = axs[0,1].legend(handles=prod_stack, loc='upper left', bbox_to_anchor=(1.05, 0.7), fontsize=16, title='Production')
legend1.get_title().set_fontsize(16)
legend1.get_title().set_fontweight('bold')
axs[0,1].add_artist(legend1)
legend2 = axs[0,1].legend(handles=feed_stack[::-1], loc='lower left', bbox_to_anchor=(1.05, 0.7), fontsize=16, title='Ingestion')
legend2.get_title().set_fontsize(16)
legend2.get_title().set_fontweight('bold')

# 3) Land use evolution plot (no unit change)
axs[1,0].stackplot(df_land.index, df_land.T.values, labels=['Arable land', 'Permanent crops', 'Permanent grassland'])
axs[1,0].set_title('c) Land use (Europe)', fontsize=18, fontweight='bold')
axs[1,0].set_ylabel('Mha', fontsize=18)
axs[1,0].legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=16)

# UAA visualization
x_pos = 2015
y_start = 5
y_end = 175

axs[1,0].annotate(
    '',
    xy=(x_pos, y_end), xytext=(x_pos, y_start),
    arrowprops=dict(arrowstyle='<->', color='k', lw=3)
)
axs[1,0].text(
    x_pos + 0.5, (y_start + y_end)/2, 'UAA',
    color='k', fontsize=16, fontweight='bold', rotation=0, va='center'
)

# 4) Yield evolution plot (no unit change)
for i, col in enumerate(yield_N.columns):
    axs[1,1].plot(yield_N.index, yield_N[col], label=col, color=crop_colors[i], linewidth=2)

# Plot aggregated arable yield on the same axis
axs[1,1].plot(
    df_agg_yield.index,
    df_agg_yield.mean(axis=1),
    label='Aggregated arable yield',
    color='black',
    linewidth=3
)

axs[1,1].set_title('d) Yield of main crops (Europe)', fontsize=18, fontweight='bold')
axs[1,1].set_ylabel('KgN/ha/year', fontsize=18)
axs[1,1].legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=16)

# Increase tick label size for all axes
tick_labelsize = 16
for ax in axs.flatten():
    ax.tick_params(axis='both', which='major', labelsize=tick_labelsize)

plt.tight_layout()
plt.savefig('figures/outputs/Figure_1_trends_in_Europe.png', dpi=400, bbox_inches='tight')
plt.close()
"""
Script name: Figure_2_territorial_maps.py
Description: Figure 2 workflow
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
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm
from matplotlib.colors import Normalize, TwoSlopeNorm

#%% --- Import data ---

budget = pd.read_csv('data/outputs/arable_budget.csv')
regions = pd.read_csv('data/regions.csv', sep=';')
UAS = pd.read_csv('data/outputs/land_areas.csv')
namure_all = pd.read_csv('data/outputs/manure_allocation.csv')
crops = pd.read_csv('data/outputs/crop_production_all_categories.csv')

# Manure losses from storage
losses_manure = (
    namure_all
    .loc[namure_all['symbol'] == 'L_storage', ['region', 'year', 'value']]
    .pivot(index='region', columns='year', values='value')
    .sort_index()
)

# Crop land in use areas
area_crops = (
    UAS
    .loc[UAS['symbol'] == 'C_sum', ['region', 'year', 'value']]
    .pivot(index='region', columns='year', values='value')
    .sort_index()
)

# Areas
area_arable = (
    budget
    .loc[budget['symbol'] == 'A', ['region', 'year', 'value']]
    .pivot(index='region', columns='year', values='value')
    .sort_index()
)

# Cropland
prod_arable_wide = (
    budget
    .loc[budget['symbol'] == 'H', ['region', 'year', 'value']]
    .pivot(index='region', columns='year', values='value')
    .sort_index()
)

# atm dep
atm_dep_complete = (
    budget
    .loc[budget['symbol'] == 'D', ['region', 'year', 'value']]
    .pivot(index='region', columns='year', values='value')
    .sort_index()
)

# fertilizer
fertilizer_al_complete = (
    budget
    .loc[budget['symbol'] == 'F', ['region', 'year', 'value']]
    .pivot(index='region', columns='year', values='value')
    .sort_index()
)

# manure
manure_al_complete = (
    budget
    .loc[budget['symbol'] == 'M', ['region', 'year', 'value']]
    .pivot(index='region', columns='year', values='value')
    .sort_index()
)

# BNF
fixation_wide = (
    budget
    .loc[budget['symbol'] == 'B', ['region', 'year', 'value']]
    .pivot(index='region', columns='year', values='value')
    .sort_index()
)

#%% --- Parameters ---

years = np.arange(1990, 2020)
shapefile_path = "data/NUTS_RG_01M_2021_4326/NUTS_RG_01M_2021_4326.shp"
output_dir = "figures"

#%% --- Prepare yields ---

# Load NUTS geometries
gdf = gpd.read_file(shapefile_path)

# Filter only selected regions
gdf = gdf[gdf['NUTS_ID'].isin(regions['NUTS_ID'])]

# Harmonize CRS and remove extreme northern regions
gdf = gdf.to_crs(epsg=4326)
gdf = gdf[gdf.geometry.centroid.y <= 73]

# Compute yield (kg/ha) per year
yield_all_years = (prod_arable_wide / area_arable).reset_index()
yield_all_years = yield_all_years.melt(id_vars='region', var_name='year', value_name='yield')
yield_all_years['year'] = yield_all_years['year'].astype(int)

# Melt area to filter based on area threshold
area_long = area_arable.reset_index().melt(id_vars='region', var_name='year', value_name='area')
area_long['year'] = area_long['year'].astype(int)

# Merge yield and area
yield_all_years = yield_all_years.merge(area_long, on=['region', 'year'])

# Yield in 2019
yield_2019 = yield_all_years[yield_all_years['year'] == 2019].copy()

# Compute means over periods
period_early = yield_all_years[yield_all_years['year'].between(1990, 1995)]
period_late = yield_all_years[yield_all_years['year'].between(2014, 2019)]

mean_yield_early = (
    period_early.groupby('region')
    .agg(mean_yield=('yield', 'mean'), mean_area=('area', 'mean'))
    .reset_index()
    .rename(columns={'mean_yield': 'yield_early', 'mean_area': 'area_early'})
)

mean_yield_late = (
    period_late.groupby('region')
    .agg(mean_yield=('yield', 'mean'), mean_area=('area', 'mean'))
    .reset_index()
    .rename(columns={'mean_yield': 'yield_late', 'mean_area': 'area_late'})
)

# Merge and apply area threshold filter
yield_diff = mean_yield_late.merge(mean_yield_early, on='region')
yield_diff = yield_diff[(yield_diff['area_early'] >= 0.05) & (yield_diff['area_late'] >= 0.05)]
yield_diff['yield_change'] = yield_diff['yield_late'] - yield_diff['yield_early']

# Merge with shapefiles
gdf_yield_2019 = gdf.merge(yield_2019[['region', 'yield']], left_on='NUTS_ID', right_on='region', how='left')
gdf_yield_change = gdf.merge(yield_diff[['region', 'yield_change']], left_on='NUTS_ID', right_on='region', how='left')
gdf_yield_late = gdf.merge(mean_yield_late[['region', 'yield_late']], left_on='NUTS_ID', right_on='region', how='left')

# Apply same CRS to yield dataframes for plotting
target_crs = "EPSG:3035"  # or choose EPSG:3857
gdf_yield_2019 = gdf_yield_2019.to_crs(target_crs)
gdf_yield_change = gdf_yield_change.to_crs(target_crs)
gdf_yield_late = gdf_yield_late.to_crs(target_crs)

#%% --- Prepare livestock density ---

# Compute mean UAA over all years par région
uaa_mean = area_crops.mean(axis=1)

# Compute livestock units (LU) per year from manure, then density per ha
# Step 1: LU = manure_al_complete / 0.000085
LU_wide = (manure_al_complete * 1_000_000) / 85

# Step 2: Melt LU_wide to long format
LU_long = LU_wide.reset_index().melt(
    id_vars='region',
    var_name='year',
    value_name='LU'
)
LU_long['year'] = LU_long['year'].astype(int)

# Step 3: Melt land_use to long format for UAA (ha)
land_use_long = area_crops.reset_index().melt(
    id_vars='region',
    var_name='year',
    value_name='UAA'
)
land_use_long['year'] = land_use_long['year'].astype(int)

# Step 4: Merge LU_long and land_use_long to compute density
density_all_years = LU_long.merge(
    land_use_long,
    on=['region', 'year'],
    how='left'
)
density_all_years['density'] = density_all_years['LU'] / (density_all_years['UAA'] * 1_000_000)

# Density in 2019
density_2019 = density_all_years[density_all_years['year'] == 2019][['region', 'density']].copy()

# Compute means over periods for density and UAA thresholds
period_early = density_all_years[density_all_years['year'].between(1990, 1994)]
period_late = density_all_years[density_all_years['year'].between(2015, 2019)]

mean_density_early = (
    period_early.groupby('region')
    .agg(mean_density=('density', 'mean'), mean_UAA=('UAA', 'mean'))
    .reset_index()
    .rename(columns={'mean_density': 'density_early', 'mean_UAA': 'UAA_early'})
)

mean_density_late = (
    period_late.groupby('region')
    .agg(mean_density=('density', 'mean'), mean_UAA=('UAA', 'mean'))
    .reset_index()
    .rename(columns={'mean_density': 'density_late', 'mean_UAA': 'UAA_late'})
)

# Merge
density_diff = mean_density_late.merge(mean_density_early, on='region')
density_diff['density_change'] = (
    density_diff['density_late'] - density_diff['density_early'])
#).clip(lower=-2, upper=2)

# Merge with shapefiles
gdf_density_2019 = gdf.merge(density_2019, left_on='NUTS_ID', right_on='region', how='left')
gdf_density_change = gdf.merge(density_diff[['region', 'density_change']],
                               left_on='NUTS_ID', right_on='region', how='left')
gdf_density_late = gdf.merge(mean_density_late[['region', 'density_late']], left_on='NUTS_ID', right_on='region', how='left')

# Apply same CRS to density dataframes for plotting
target_crs = "EPSG:3035"
gdf_density_2019 = gdf_density_2019.to_crs(target_crs)
gdf_density_change = gdf_density_change.to_crs(target_crs)
gdf_density_late = gdf_density_late.to_crs(target_crs)

#%% --- Prepare fertilization mode ---

# Compute the mineral fertilization ratio (in %)
fertilizer_fraction = fertilizer_al_complete / (
    fertilizer_al_complete + manure_al_complete + fixation_wide + atm_dep_complete
)
fertilizer_fraction_percent = fertilizer_fraction * 100

# Melt to long format
fertilizer_frac_long = (
    fertilizer_fraction_percent.reset_index()
    .melt(id_vars='region', var_name='year', value_name='mineral_frac')
    .replace(0, np.nan)
)
fertilizer_frac_long['year'] = fertilizer_frac_long['year'].astype(int)

# Maps: 2019 value and change (2015–2019 vs. 1990–1994)

# Get 2019 map
frac_2019 = fertilizer_frac_long[fertilizer_frac_long['year'] == 2019][['region', 'mineral_frac']].copy()

# Get mean over early and late periods
early_frac = (
    fertilizer_frac_long[fertilizer_frac_long['year'].between(1990, 1994)]
    .groupby('region')['mineral_frac']
    .mean()
    .reset_index(name='frac_early')
)
late_frac = (
    fertilizer_frac_long[fertilizer_frac_long['year'].between(2015, 2019)]
    .groupby('region')['mineral_frac']
    .mean()
    .reset_index(name='frac_late')
)

# Combine and compute change
frac_change = late_frac.merge(early_frac, on='region')
frac_change['frac_change'] = frac_change['frac_late'] - frac_change['frac_early']

# Merge with shapefiles
gdf_frac_2019 = gdf.merge(frac_2019, left_on='NUTS_ID', right_on='region', how='left')
gdf_frac_change = gdf.merge(frac_change[['region', 'frac_change']], left_on='NUTS_ID', right_on='region', how='left')
gdf_late_frac = gdf.merge(late_frac, left_on='NUTS_ID', right_on='region', how='left')

# Apply same CRS
gdf_frac_2019 = gdf_frac_2019.to_crs(target_crs)
gdf_late_frac = gdf_late_frac.to_crs(target_crs)
gdf_frac_change = gdf_frac_change.to_crs(target_crs)

#%% --- Prepare total N fertilization rates ---

# Total N input (kg) = sum of all sources
N_total_input = fertilizer_al_complete + manure_al_complete + fixation_wide + atm_dep_complete

# Total N input per ha of arable land (kg/ha)
N_input_per_ha = N_total_input / area_arable

N_input_long = (
    N_input_per_ha.reset_index()
    .melt(id_vars='region', var_name='year', value_name='N_input_kg_ha')
)
N_input_long['year'] = N_input_long['year'].astype(int)

# Remove values > 1000
N_input_long = N_input_long[N_input_long['N_input_kg_ha'] <= 1000]

# 2019
N_input_2019 = N_input_long[N_input_long['year'] == 2019][['region', 'N_input_kg_ha']].copy()

# Means
early_N = (
    N_input_long[N_input_long['year'].between(1990, 1994)]
    .groupby('region')['N_input_kg_ha']
    .mean()
    .reset_index(name='N_input_early')
)
late_N = (
    N_input_long[N_input_long['year'].between(2015, 2019)]
    .groupby('region')['N_input_kg_ha']
    .mean()
    .reset_index(name='N_input_late')
)

# Difference
N_input_change = late_N.merge(early_N, on='region')
N_input_change['N_input_change'] = N_input_change['N_input_late'] - N_input_change['N_input_early']

gdf_N_2019 = gdf.merge(N_input_2019, left_on='NUTS_ID', right_on='region', how='left')
gdf_late_N = gdf.merge(late_N, left_on='NUTS_ID', right_on='region', how='left')
gdf_N_change = gdf.merge(N_input_change[['region', 'N_input_change']], left_on='NUTS_ID', right_on='region', how='left')

gdf_N_2019 = gdf_N_2019.to_crs(target_crs)
gdf_N_change = gdf_N_change.to_crs(target_crs)
gdf_late_N = gdf_late_N.to_crs(target_crs)

#%% --- Prepare N balance ---

# Calcul N balance (kg/ha)
N_balance = (fertilizer_al_complete + manure_al_complete + fixation_wide + atm_dep_complete) - prod_arable_wide

# Format long
N_balance_long = (
    N_balance.reset_index()
    .melt(id_vars='region', var_name='year', value_name='N_balance_kg_ha')
)
N_balance_long['year'] = N_balance_long['year'].astype(int)

N_balance_long = N_balance_long[N_balance_long['N_balance_kg_ha'] >= 0]


# 2019
N_balance_2019 = N_balance_long[N_balance_long['year'] == 2019][['region', 'N_balance_kg_ha']].copy()

# Means
early_bal = (
    N_balance_long[N_balance_long['year'].between(1990, 1994)]
    .groupby('region')['N_balance_kg_ha']
    .mean()
    .reset_index(name='N_balance_early')
)
late_bal = (
    N_balance_long[N_balance_long['year'].between(2015, 2019)]
    .groupby('region')['N_balance_kg_ha']
    .mean()
    .reset_index(name='N_balance_late')
)

# Difference
N_balance_change = late_bal.merge(early_bal, on='region')
N_balance_change['N_balance_change'] = N_balance_change['N_balance_late'] - N_balance_change['N_balance_early']

# Merge with geo
gdf_bal_2019 = gdf.merge(N_balance_2019, left_on='NUTS_ID', right_on='region', how='left')
gdf_late_bal = gdf.merge(late_bal, left_on='NUTS_ID', right_on='region', how='left')
gdf_bal_change = gdf.merge(N_balance_change[['region', 'N_balance_change']], left_on='NUTS_ID', right_on='region', how='left')

# Reprojection
gdf_bal_2019 = gdf_bal_2019.to_crs(target_crs)
gdf_late_bal = gdf_late_bal.to_crs(target_crs)
gdf_bal_change = gdf_bal_change.to_crs(target_crs)

#%% --- Prepare NUE data ---

# Calcul NUE
NUE = (prod_arable_wide/area_arable) / ((fertilizer_al_complete + manure_al_complete + fixation_wide + atm_dep_complete)/area_arable)

# Long format
NUE_long = (
    NUE.reset_index()
    .melt(id_vars='region', var_name='year', value_name='NUE')
)
NUE_long['year'] = NUE_long['year'].astype(int)

# 2019
NUE_2019 = NUE_long[NUE_long['year'] == 2019][['region', 'NUE']].copy()

# Means
early_NUE = (
    NUE_long[NUE_long['year'].between(1990, 1994)]
    .groupby('region')['NUE']
    .mean()
    .reset_index(name='NUE_early')
)
late_NUE = (
    NUE_long[NUE_long['year'].between(2015, 2019)]
    .groupby('region')['NUE']
    .mean()
    .reset_index(name='NUE_late')
)

# Variation
NUE_change = late_NUE.merge(early_NUE, on='region')
NUE_change['NUE_change'] = NUE_change['NUE_late'] - NUE_change['NUE_early']

# Merge with geodata
gdf_NUE_2019 = gdf.merge(NUE_2019, left_on='NUTS_ID', right_on='region', how='left')
gdf_late_NUE = gdf.merge(late_NUE, left_on='NUTS_ID', right_on='region', how='left')
gdf_NUE_change = gdf.merge(NUE_change[['region', 'NUE_change']], left_on='NUTS_ID', right_on='region', how='left')

# Projection
gdf_NUE_2019 = gdf_NUE_2019.to_crs(target_crs)
gdf_late_NUE = gdf_late_NUE.to_crs(target_crs)
gdf_NUE_change = gdf_NUE_change.to_crs(target_crs)

# %% --- Create a 4x3 map panel: deltas between 1990-1995 and 2015-2019 ---

# Figure
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(14, 16))
axes = axes.flatten()

# Data
map_data_all = [
    (gdf_yield_late, 'yield_late', "a) Arable yield (2015–2019)", 'YlGn', 'kgN/ha/yr'),
    (gdf_density_late, 'density_late', "b) Livestock density (2015–2019)", 'Oranges', 'LU/haUAA'),
    (gdf_late_N, 'N_input_late', "c) Arable total N input (2015–2019)", 'Blues', 'kgN/ha'),
    (gdf_late_frac, 'frac_late', "d) Mineral fertilizer ratio \n in arable lands (2015–2019)", 'PuRd', '%'),
    (gdf_late_NUE, 'NUE_late', "e) Arable N use efficiency (2015–2019)", 'Greens', 'ratio'),
    (gdf_late_bal, 'N_balance_late', "f) Arable N surplus (2015–2019)", 'Reds', 'kgN/ha'),
    (gdf_yield_change, 'yield_change', "g) Δ Arable yield\n(2015–2019 vs 1990–1994)", 'coolwarm', 'Δ kgN/ha/yr'),
    (gdf_density_change, 'density_change', "h) Δ Livestock density\n(2015–2019 vs 1990–1994)", 'coolwarm', 'Δ LU/haUAA'),
    (gdf_N_change, 'N_input_change', "i) Δ Arable total N input\n(2015–2019 vs 1990–1994)", 'coolwarm', 'Δ kgN/ha'),
    (gdf_frac_change, 'frac_change', "j) Δ Mineral fertilizer ratio in arable \n lands (2015–2019 vs 1990–1994)", 'coolwarm', 'Δ %'),
    (gdf_NUE_change, 'NUE_change', "k) Δ Arable N use efficiency\n(2015–2019 vs 1990–1994)", 'coolwarm', 'Δ ratio'),
    (gdf_bal_change, 'N_balance_change', "l) Arable Δ N surplus\n(2015–2019 vs 1990–1994)", 'coolwarm', 'Δ kgN/ha'),
]

# Maps
for ax, (gdf, column, title, cmap_name, unit) in zip(axes, map_data_all):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.05)

    # Normalisation
    if "change" in column:
        vabs = max(abs(gdf[column].min()), abs(gdf[column].max()))
        norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
    else:
        norm = Normalize(vmin=gdf[column].min(), vmax=gdf[column].max())

    # Maps
    gdf.plot(
        column=column,
        cmap=cmap_name,
        linewidth=0.1,
        ax=ax,
        edgecolor='0.5',
        norm=norm,
        missing_kwds={
            "color": "lightgrey",
            "edgecolor": "white",
            "hatch": "///",
            "label": "Missing data"
        }
    )

    # Colorbar
    sm = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap(cmap_name))
    sm._A = []
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(unit, fontsize=13)
    cbar.ax.tick_params(labelsize=12)

    # Colorbar definition
    box = cax.get_position()
    shrink = 0.7
    new_height = box.height * shrink
    new_y = box.y0 + (box.height - new_height) / 2
    cax.set_position([box.x0, new_y, box.width, new_height])

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.axis('off')

plt.subplots_adjust(
    left=0.01, right=0.99, top=0.97, bottom=0.03,
    wspace=0.02,
    hspace=0.18
)

# Save
plt.savefig('figures/outputs/Figure_2_territorial_maps.png', dpi=400)
plt.close()
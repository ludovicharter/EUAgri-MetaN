"""
Script name: Figure_3_territorial_typology.py
Description: Figure 3 workflow
Author: Ludovic Harter
Created: 2025-12-18
Last modified: 2025-12-18
Version: 1.0
Project: Territorial nitrogen flows and metabolic typologies of EU Agri-Food Systems, 1990–2019
License: MIT
"""

#%% --- Libraries ---
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#%% --- Plot typologies ---

# Set shapefile path
shp_path = 'data/NUTS_RG_01M_2021_4326/NUTS_RG_01M_2021_4326.shp'

# Load shapes
nuts_shapes = gpd.read_file(shp_path)

# Prepare typology data
typ_df = pd.read_csv('data/outputs/typologies.csv')
typ_T = typ_df[typ_df['symbol'] == 'typ']

# Define typologies and custom colors
all_typologies = ['SCS', 'LVK', 'MXG', 'MXF', 'DSG']
typology_colors = {
    'DSG': 'lightyellow',
    'LVK': 'tab:red',
    'MXF': 'tab:orange',
    'MXG': 'tab:green',
    'SCS': 'yellow',
}

years_to_plot = [1990, 2019]

# Region groups
regions_est = [
    'BG', 'CZ', 'DE3', 'DE4', 'DE8', 'DED', 'DEE', 'DEG', 'EE', 'HU1', 'HU2', 'HU3',
    'LT', 'LV', 'PL2', 'PL4', 'PL5', 'PL6', 'PL7', 'PL8', 'PL9',
    'RO11', 'RO12', 'RO21', 'RO22', 'RO31', 'RO32', 'RO41', 'RO42',
    'RS', 'SI', 'SK'
]

regions_med = [
    'AL', 'CY', 'EL', 'ES11', 'ES12', 'ES13', 'ES21', 'ES22', 'ES23', 'ES24',
    'ES30', 'ES41', 'ES42', 'ES43', 'ES51', 'ES52', 'ES53', 'ES61', 'ES62',
    'FRJ1', 'FRJ2', 'FRL0', 'FRM0', 'HR', 'ITC', 'ITF', 'ITG', 'ITH', 'ITI',
    'MK', 'MT', 'PT11', 'PT15', 'PT16', 'PT17', 'PT18'
]

# Filter shapes for region groups
est_shapes = nuts_shapes[nuts_shapes['NUTS_ID'].isin(regions_est)]
med_shapes = nuts_shapes[nuts_shapes['NUTS_ID'].isin(regions_med)]

# Dissolve to get one polygon per group (for contour)
est_union = est_shapes.dissolve()
med_union = med_shapes.dissolve()

fig, axs = plt.subplots(1, 2, figsize=(16, 10))

for ax, yr in zip(axs, years_to_plot):
    # Extract typologies for the chosen year for symbol 'T'
    df_yr = (
        typ_T[typ_T['year'] == yr][['region', 'value']]
        .rename(columns={'value': 'typology'})
    )

    # Merge shapes with typologies (symbol 'T')
    merged = nuts_shapes.merge(
        df_yr,
        how='left',
        left_on='NUTS_ID',
        right_on='region'
    )

    # Assign color to each row based on typology
    merged['color'] = merged['typology'].map(typology_colors)

    # Plot the typologies
    for typ in all_typologies:
        merged[merged['typology'] == typ].plot(
            ax=ax,
            color=typology_colors[typ],
            edgecolor='grey',
            linewidth=0.2
        )

    # Add hatch on urban areas ('symbol' == 'U' and typology == 'URB') for the same year
    df_urb = typ_df[(typ_df['symbol'] == 'urb_typ') & (typ_df['year'] == yr) & (typ_df['value'] == 'URB')]
    df_urb = df_urb[['region']].drop_duplicates()

    merged_urb = nuts_shapes.merge(
        df_urb,
        how='inner',
        left_on='NUTS_ID',
        right_on='region'
    )

    # Plot hatching for urban regions
    merged_urb.plot(
        ax=ax,
        facecolor="none",
        edgecolor='black',
        linewidth=0.5,
        hatch='/////',
        zorder=10  # on top of colors
    )

    # Plot contours for region groups
    est_union.boundary.plot(ax=ax, edgecolor='red', linewidth=2, zorder=20)
    med_union.boundary.plot(ax=ax, edgecolor='blue', linewidth=2, zorder=20)

    ax.set_axis_off()
    ax.set_title(f"{yr}", fontsize=20, pad=15)
    ax.set_xlim(-25, 35)
    ax.set_ylim(34, 72)

# --- Legend construction once for both plots ---

# Patches for typologies (colors)
legend_handles = [mpatches.Patch(color=typology_colors[typ], label=typ) for typ in all_typologies]

# Patch for urban hatch
urban_patch = mpatches.Patch(facecolor='white', hatch='/////', edgecolor='black', label='URB')

# Patches for region groups contours
est_patch = mpatches.Patch(facecolor='none', edgecolor='red', linewidth=3, label='Eastern group')
med_patch = mpatches.Patch(facecolor='none', edgecolor='blue', linewidth=3, label='Mediterranean group')

#legend_handles.extend([urban_patch, est_patch, med_patch])
legend_handles.extend([urban_patch])

# Put legend on the right side of the figure
fig.legend(
    handles=legend_handles,
    title='Typology',
    loc='center left',
    fontsize=14,
    title_fontsize=16,
    frameon=True,
    edgecolor='black'
)

plt.subplots_adjust(right=0.85)  # Make space for legend
plt.tight_layout()
plt.savefig('figures/outputs/Figure_3_territorial_typology.png', bbox_inches='tight', dpi=400)
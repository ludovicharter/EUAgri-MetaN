"""
Script name: g_atmospheric_deposition.py
Description: Calculates atmospheric deposition rate for each territory.
Author: Ludovic Harter
Created: 2025-05-01
Last modified: 2025-08-13
Version: 1.0
Project: Territorial nitrogen flows and metabolic typologies of EU Agri-Food Systems, 1990–2019
License: MIT


Before running this script, make sure the following datasets are downloaded and available locally:

1. Corine Land Cover 2018
   Source: https://doi.org/10.2909/960998c1-1870-4e82-8051-6485205ebbac
   Format: GeoTIFF
   Recommended location: data/CORINE_Land_Cover2018/DATA/U2018_CLC2018_V2020_20u1.tif

2. EMEP Atmospheric Deposition Outputs
   Source: https://www.emep.int
   Format: NetCDF (.nc) files
   Recommended location: data/EMEP_deposition/
"""

#%% --- Libraries ---
import netCDF4 as nc
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import mapping
import rasterio
import rioxarray
import numpy as np
from rasterio.transform import from_bounds, rowcol

#%% --- Define the period and a dict to store deposition values ---
years = np.arange(1990, 2020)
deposition = {}

#%% --- Process each year ---
for year in years:

    # Print the processed year
    print(f'Processing deposition year {year} ...')

    # Import EMEP data
    path = 'data/EMEP_deposition/'
    emep = nc.Dataset(path + f'EMEP01_rv5.3_year.{year}met_{year}emis_rep2024.nc')

    # Extract data and coordinates
    dry_oxidized = emep.variables['DDEP_OXN_m2Grid'][:]  # Units: mg(N)/m2
    wet_oxidized = emep.variables['WDEP_OXN'][:]  # Units: mg(N)/m2
    dry_reduced = emep.variables['DDEP_RDN_m2Grid'][:]  # Units: mg(N)/m2
    wet_reduced = emep.variables['WDEP_RDN'][:]  # Units: mg(N)/m2
    lon = emep.variables['lon'][:]
    lat = emep.variables['lat'][:]

    # Convert to DataArray and assign coordinates
    dry_oxidized_da = xr.DataArray(dry_oxidized[0], coords=[lat, lon], name='dry_oxidized', dims=['lat', 'lon']).rename(
        {'lat': 'y', 'lon': 'x'})
    wet_oxidized_da = xr.DataArray(wet_oxidized[0], coords=[lat, lon], name='wet_oxidized', dims=['lat', 'lon']).rename(
        {'lat': 'y', 'lon': 'x'})
    dry_reduced_da = xr.DataArray(dry_reduced[0], coords=[lat, lon], name='dry_reduced', dims=['lat', 'lon']).rename(
        {'lat': 'y', 'lon': 'x'})
    wet_reduced_da = xr.DataArray(wet_reduced[0], coords=[lat, lon], name='wet_reduced', dims=['lat', 'lon']).rename(
        {'lat': 'y', 'lon': 'x'})

    # Write CRS
    for da in [dry_oxidized_da, dry_reduced_da, wet_reduced_da, wet_oxidized_da]:
        da.rio.write_crs("EPSG:4326", inplace=True)

    # Read the CORINE raster file
    corine_path = 'data/CORINE_Land_Cover2018/DATA/U2018_CLC2018_V2020_20u1.tif'
    with rasterio.open(corine_path) as src:
        corine = src.read(1)
        corine_transform = src.transform
        corine_crs = src.crs

    # Load NUTS shapefile
    shapes = 'data/NUTS_RG_01M_2021_4326/NUTS_RG_01M_2021_4326.shp'
    nuts = gpd.read_file(shapes)

    # Load region names
    regions = pd.read_csv('data/regions.csv', sep=';')

    # Merge selected regions with their shapes
    nuts = pd.merge(nuts, regions, on='NUTS_ID')

    # Precompute the constants
    res_y = (lat[1] - lat[0]).item()
    res_x = (lon[1] - lon[0]).item()

    # Initialize a matrix to store the count of code 12 (=non-irrigated cropland) within each climate data cell
    code_12_count = np.zeros((len(lat), len(lon)))

    # Loop through each EMEP cell and count raster code 12 (=non-irrigated cropland) occurrences
    for i, latitude in enumerate(lat):
        for j, longitude in enumerate(lon):
            left = longitude - res_x / 2
            right = longitude + res_x / 2
            bottom = latitude - res_y / 2
            top = latitude + res_y / 2

            # Convert bounds to raster coordinates
            row_start, col_start = rowcol(corine_transform, left, top)
            row_end, col_end = rowcol(corine_transform, right, bottom)

            # Ensure the bounds are within raster dimensions
            row_start = np.clip(row_start, 0, corine.shape[0] - 1)
            row_end = np.clip(row_end, 0, corine.shape[0] - 1)
            col_start = np.clip(col_start, 0, corine.shape[1] - 1)
            col_end = np.clip(col_end, 0, corine.shape[1] - 1)

            # Count cells with code 12 within these bounds
            row_start, row_end = sorted([int(row_start), int(row_end)])
            col_start, col_end = sorted([int(col_start), int(col_end)])
            code_12_count[i, j] = np.sum(corine[row_start:row_end + 1, col_start:col_end + 1] == 12)

    # Transform the counts into a DataArray with coordinates
    code_12_count_da = xr.DataArray(code_12_count, dims=("y", "x"), coords={"y": lat, "x": lon})
    code_12_count_da.rio.write_crs("EPSG:4326", inplace=True)

    # Initialize dict to store results
    results = {'NUTS_ID': [],
               'Mean_Dry_Oxidized': [],
               'Mean_Dry_Reduced': [],
               'Mean_Wet_Oxidized': [],
               'Mean_Wet_Reduced': [],
               }

    # Compute the weighted mean for each NUTS region
    for region in nuts['NUTS_ID']:
        n = nuts[nuts['NUTS_ID'] == region]
        geometries = [mapping(geom) for geom in n.geometry]

        dry_oxidized_clip = dry_oxidized_da.rio.clip(geometries, n.crs, drop=False, all_touched=True)
        dry_reduced_clip = dry_reduced_da.rio.clip(geometries, n.crs, drop=False, all_touched=True)
        wet_reduced_clip = wet_reduced_da.rio.clip(geometries, n.crs, drop=False, all_touched=True)
        wet_oxidized_clip = wet_oxidized_da.rio.clip(geometries, n.crs, drop=False, all_touched=True)

        clipped_count = code_12_count_da.rio.clip(geometries, n.crs, drop=False, all_touched=True)

        # Compute the weighted mean
        valid_mask = ~np.isnan(dry_oxidized_clip)
        total_count = clipped_count.where(valid_mask).sum().item()

        if total_count > 0:
            weighted_mean_dry_oxidized = (dry_oxidized_clip * clipped_count).where(valid_mask).sum().item() / total_count
            weighted_mean_dry_reduced = (dry_reduced_clip * clipped_count).where(valid_mask).sum().item() / total_count
            weighted_mean_wet_reduced = (wet_reduced_clip * clipped_count).where(valid_mask).sum().item() / total_count
            weighted_mean_wet_oxidized = (wet_oxidized_clip * clipped_count).where(valid_mask).sum().item() / total_count
        else:
            weighted_mean_dry_oxidized = np.nan
            weighted_mean_dry_reduced = np.nan
            weighted_mean_wet_reduced = np.nan
            weighted_mean_wet_oxidized = np.nan

        results['NUTS_ID'].append(region)
        results['Mean_Dry_Oxidized'].append(weighted_mean_dry_oxidized)
        results['Mean_Dry_Reduced'].append(weighted_mean_dry_reduced)
        results['Mean_Wet_Reduced'].append(weighted_mean_wet_reduced)
        results['Mean_Wet_Oxidized'].append(weighted_mean_wet_oxidized)

    # Convert the results dictionary to a DataFrame
    deposition[year] = pd.DataFrame(results)

#%% --- Define a new dataset to store deposition rates ---

# Create df product of region, year
final_deposition = pd.MultiIndex.from_product(
    [sorted(regions['NUTS_ID']), years],
    names=["region", "year"]
).to_frame(index=False)

# Add constant and NaN columns
final_deposition["value"] = None
final_deposition["label"] = "atmospheric deposition rate to cropland"
final_deposition["unit"] = "kg N / ha"
final_deposition["confidence"] = None

# Loop over each year
for year in years:

    # Work on a copy to avoid SettingWithCopyWarning
    dep = deposition[year].copy()

    # Calculate total deposition in kg N/ha
    dep['total'] = (
                    dep['Mean_Dry_Oxidized'] + dep['Mean_Dry_Reduced'] +
                    dep['Mean_Wet_Oxidized'] + dep['Mean_Wet_Reduced']
                   ) / 100

    # Loop over regions
    for region in regions['NUTS_ID']:
        # Extract total deposition for this region
        value_series = dep.loc[dep['NUTS_ID'] == region, 'total']

        if not value_series.empty:
            value = value_series.iloc[0]
            mask = (final_deposition['region'] == region) & (final_deposition['year'] == int(year))
            final_deposition.loc[mask, 'value'] = value
            final_deposition.loc[mask, 'confidence'] = 'high'

# Add the region name by mapping from regions['name']
name_map = dict(zip(regions['NUTS_ID'], regions['name']))
final_deposition.insert(
    loc=final_deposition.columns.get_loc('region') + 1,
    column='region name',
    value=final_deposition['region'].map(name_map)
)

#%% Save data

final_deposition.to_csv('data/outputs/atmospheric_deposition_rate.csv')


"""
Script name: utils.py
Description: functions util for the project.
Author: Ludovic Harter
Created: 2025-05-01
Last modified: 2025-08-13
Version: 1.0
Project: Territorial nitrogen flows and metabolic typologies of EU Agri-Food Systems, 1990–2019
License: MIT
"""

# Libraries
import pandas as pd
import numpy as np
from typing import Iterable, Mapping


def import_eurostat_sheets(path, sheet_indices, na_val=':', header_row=8):
    """
    Import and format one or multiple Eurostat Excel sheets.

    Parameters:
    - path: str, path to the Excel file.
    - sheet_indices: int or list of int, index or indices of sheets to import.
    - na_val: str, representing missing values (default is ':').
    - header_row: int, row index containing column headers (default is 8).

    Returns:
    - A single formatted DataFrame if one sheet index is provided,
      otherwise a list of formatted DataFrames.
    """

    # Handle single index case
    single = isinstance(sheet_indices, int)
    indices = [sheet_indices] if single else sheet_indices

    dfs = []
    for index in indices:
        df = pd.read_excel(path, sheet_name=f'Sheet {index}', header=None, na_values=na_val)
        df = format_eurostat_df(df, header_row)
        dfs.append(df)

    return dfs[0] if single else dfs


def format_eurostat_df(df, header_row=8):
    """
    Standardize Eurostat DataFrame format.
    Sets column names from the specified header row and renames the first column as 'region'.

    Parameters:
    - df: pd.DataFrame raw Eurostat data
    - header_row: int index of the row containing column headers (default is 8)

    Returns:
    - pd.DataFrame: formatted with standardized column names
    """

    # Extract new column names
    raw_columns = df.iloc[header_row].reset_index(drop=True)
    new_columns = []

    for col in raw_columns:
        try:
            new_columns.append(float(col))  # Convert year labels to float if possible
        except ValueError:
            new_columns.append(col)

    # Rename first column as 'region'
    new_columns[0] = 'region'

    # Slice the data starting from the row after header_row
    df_clean = df.iloc[header_row + 1:].copy()
    df_clean.columns = new_columns

    # Reset index if needed
    df_clean.reset_index(drop=True, inplace=True)

    return df_clean


def build_eurostat_dict(years, crops, codes, nuts_df):
    """
    Construct yearly crop production data by region.
    Assembles a dictionary of DataFrames indexed by year, with specific variable
    values per NUTS region extracted from harmonized Eurostat inputs.

    Parameters:
    - years: list of int target years
    - crops: list of str crop names
    - codes : list of pd.DataFrame Eurostat-formatted crop DataFrames
    - nuts_df : pd.DataFrame with NUTS regions

    Returns:
    - dict of pd.DataFrame indexed by year with regional specific variable
    """

    dt = {year: pd.DataFrame({'region': nuts_df['NUTS_ID']}) for year in years}
    for year in years:
        for crop_id, crop in enumerate(crops):
            values = []
            for region in nuts_df['NUTS_ID']:
                if year in codes[crop_id].columns and region in codes[crop_id]['region'].values:
                    # Extract production value if year and region are present
                    values.append(codes[crop_id].loc[codes[crop_id]['region'] == region, year].values[0])
                else:
                    values.append(np.nan)  # Missing data is set as NaN
            dt[year][crop] = values
    return dt


def merge_crops(dt: dict[int, pd.DataFrame], crop_combinations: dict[str, dict[str, list[str]]]) -> dict[int, pd.DataFrame]:
    """
    Aggregate multiple crop columns into unified categories, with per-column skipna control.

    Parameters:
    - dt: dict of yearly crop production data by region (DataFrames indexed by region)
    - crop_combinations : dict
        mapping new_crop_name → dict with keys:
          - 'cols_strict'   : list of columns to sum with skipna=False
          - 'cols_flexible' : list of columns to sum with skipna=True

    Returns:
    - same dict of DataFrames, each extended with the new aggregated columns
    """

    for year, df in dt.items():
        for new_crop, behavior in crop_combinations.items():
            cols_strict = behavior.get('cols_strict', [])
            cols_flexible = behavior.get('cols_flexible', [])

            # sum the “strict” columns: any NaN → result NaN
            if cols_strict:
                sum_strict = df[cols_strict].sum(axis=1, skipna=False)
            else:
                # nothing strict → zero for all rows
                sum_strict = 0

            # sum the “flexible” columns: NaNs ignored
            if cols_flexible:
                sum_flex = df[cols_flexible].sum(axis=1, skipna=True)
            else:
                sum_flex = 0

            # combined: strict-sum may be NaN, flex-sum is numeric
            df[new_crop] = sum_strict + sum_flex

    return dt


def process_euragri_data(years, categories, euragri_df, nuts_df):
    """
    Aggregate EuropeAgriDB data by country and year.

    Extracts a specific variable for selected crop categories, grouping by 2-letter country codes from NUTS regions.
    Returns one dataframe per year.

    Parameters:
    - years: list of int years to include
    - categories: list of str crop categories to extract
    - euragri_df: pd.DataFrame EuropeAgriDB data with columns: 'Region', 'Year', 'Category', 'Value'
    - nuts_df: pd.DataFrame NUTS regions with 'NUTS_ID' column

    Returns:
    - dict[int, pd.DataFrame]: yearly dataframes with specific variable per country
    """

    euragri = {}
    for year in years:
        euragri[year] = pd.DataFrame()
        for category in categories:
            values = []
            countries = []
            # Loop through each region in nuts and collect unique country codes
            for region in nuts_df['NUTS_ID']:
                country = region[:2]  # Extract the 2-letter country code
                if country not in countries:
                    countries.append(country)
                    # Extract value for the given country, year, and crop category
                    value = euragri_df['Value'].loc[
                        (euragri_df['Region'] == country) &
                        (euragri_df['Year'] == year) &
                        (euragri_df['Category'] == category)]
                    if value.empty:
                        values.append(np.nan)
                    else:
                        values.append(value.item())
            euragri[year]['region'] = countries
            euragri[year][category] = values
    return euragri


def compute_regional_distribution_coefficients(dfs, regions, crops=None, years=None):
    """
    Compute a regional distribution coefficient for each region, crop, and year,
    indicating the share of regional production in total national production.

    If a region is a country (i.e., 2-letter code), its coefficient is set to 1.0.
    For each crop-year, if the sum of coefficients for subregions of a country is not 1,
    they are rescaled so that they sum to 1.

    Parameters:
    - dfs: dict of DataFrames {year: DataFrame with region as index and crop columns}
    - regions: DataFrame with a 'NUTS_ID' column for all regions
    - crops: list of crop names
    - years: list of years (used as keys in `dfs`)

    Returns:
    - coefficients: dict of DataFrames {crop: DataFrame with columns ['region', year1, ..., 'mean', 'std_dev']}
    """

    if years is None:
        years = list(dfs.keys())
    if crops is None:
        crops = ['value']  # Default column name if no crops specified

    # Initialize a dict to hold one output DataFrame per crop
    coefficients = {
        crop: pd.DataFrame({'region': regions['NUTS_ID'].astype(str)})
        for crop in crops
    }

    # Compute year-by-year coefficients
    for crop in crops:
        for year in years:
            coefs = []
            df_year = dfs.get(year, pd.DataFrame())

            for region in regions['NUTS_ID']:
                r = str(region)            # <-- force string
                # Case 1: country-level region (exactly 2 letters)
                if len(r) == 2:
                    coefs.append(1.0)
                    continue

                country = r[:2]
                # restrict to same-country rows
                country_data = df_year[df_year.index.astype(str).str.startswith(country)]

                # If crop is missing in this year's country data
                if crop not in country_data.columns:
                    coefs.append(np.nan)
                    continue

                region_value = df_year.at[r, crop] if r in df_year.index else np.nan

                # Check if there are any NaNs in the crop column of country_data
                if not country_data[crop].isna().any():
                    total = country_data[crop].sum(skipna=True)
                else:
                    total = np.nan

                if pd.isna(region_value) or total == 0 or pd.isna(total):
                    coefs.append(np.nan)
                else:
                    coefs.append(region_value / total)

            # store this year’s vector
            coefficients[crop][year] = coefs

    # Compute mean & std‐dev, then normalize per country
    for crop in crops:
        df_coef = coefficients[crop]
        year_cols = [y for y in years if y in df_coef.columns]
        # mask zeros so they don't skew stats
        data = df_coef[year_cols].replace(0, np.nan)
        df_coef['mean'] = data.mean(axis=1, skipna=True)
        df_coef['std_dev'] = data.std(axis=1, skipna=True)

        # normalize means so that each country's sum = 1
        for country in {str(r)[:2] for r in df_coef['region']}:
            mask = df_coef['region'].astype(str).str.startswith(country)
            total = df_coef.loc[mask, 'mean'].sum(skipna=True)
            if total and not np.isnan(total):
                df_coef.loc[mask, 'mean'] = df_coef.loc[mask, 'mean'] / total
            else:
                df_coef.loc[mask, 'mean'] = np.nan

    return coefficients


def fill_input_prod_areas(
    dt1: dict,
    dt2: dict,
    years: list,
    crops: list,
    mode: str = 'production'
) -> list:
    """
    Impute missing or zero values in production or area DataFrames using yields.

    Two modes:
      - 'production': correct production (dt1) using area (dt2).
      - 'areas': correct area (dt1) using production (dt2).

    Strategy:
    1. Compute yields = production / area for all region-year-crop where both present.
    2. For each target entry (production or area) missing/zero and counterpart > 0:
       a. Find yield for same region in nearest year.
       b. Else use yield from neighboring regions (same 2-letter prefix).
       c. Compute new_value = yield * counterpart (or counterpart / yield if mode 'areas').
    3. Record corrections: (year, region, crop, new_value, reason).

    Parameters:
    - dt1 (dict): year -> DataFrame to correct (production or area).
    - dt2 (dict): year -> DataFrame of counterpart (area or production).
    - years (list): years to process.
    - crops (list): crop columns.
    - mode (str): 'production' or 'areas'.

    Returns:
      - corrections (list of tuples): (year, region, crop, new_value, reason)
    """

    # Build yields history from both dt1 & dt2
    yields = {}
    for year in years:
        # production vs area
        df_p = dt1[year] if mode == 'production' else dt2[year]
        df_a = dt2[year] if mode == 'production' else dt1[year]
        for region in df_p.index.intersection(df_a.index):
            for crop in crops:
                p = df_p.at[region, crop]
                a = df_a.at[region, crop]
                if pd.notna(p) and pd.notna(a) and a != 0:
                    yields.setdefault(region, {}).setdefault(year, {})[crop] = p / a

    def find_yield(region, year, crop):
        # same region
        if region in yields:
            yrs = [y for y in yields[region] if crop in yields[region][y]]
            if yrs:
                best = min(yrs, key=lambda y: abs(y-year))
                return yields[region][best][crop], 'impute_region'
        # neighbors
        prefix = region[:2]
        candidates = []
        for reg, yrs in yields.items():
            if reg.startswith(prefix) and reg != region:
                for y, crop_map in yrs.items():
                    if crop in crop_map:
                        candidates.append((y, crop_map[crop]))
        if candidates:
            y_n, val = min(candidates, key=lambda x: abs(x[0]-year))
            return val, 'impute_neighbor'
        return None, None

    corrections = []
    # Impute and print corrections
    for year in years:
        df_t = dt1[year]
        df_c = dt2[year]
        for region in df_t.index:
            for crop in crops:
                t = df_t.at[region, crop]
                c = df_c.at[region, crop] if region in df_c.index else np.nan
                if pd.notna(c) and c > 0 and (pd.isna(t) or t == 0):
                    yld, reason = find_yield(region, year, crop)
                    if yld is not None:
                        new_val = yld * c if mode == 'production' else c / yld
                        df_t.at[region, crop] = new_val
                        corrections.append((year, region, crop, new_val, reason))
                        # Print summary of the correction
                        print(f"[{mode.upper()}] Year={year}, Region={region}, Crop={crop}: set to {new_val:.2f} ({reason})")
        dt1[year] = df_t
    return corrections


def interpolate_dt(
    dt: dict[int, pd.DataFrame],
    interpolate_crops: list[str],
    extrapolate: bool = False
) -> dict[int, pd.DataFrame]:
    """
    Interpolate (and optionally extrapolate) specified crop categories over time
    in a dictionary of annual DataFrames.

    Parameters:
    - dt: dict mapping year (int) to DataFrame (regions as index, crop categories as columns)
    - interpolate_crops: list of categories to interpolate/extrapolate; others remain unchanged
    - extrapolate: if False, only interpolate inside existing data (no extrapolation);
                   if True, also extrapolate to fill leading/trailing NaNs

    Process:
        1. Melt wide to long format
        2. Sort and keep pre-interpolation copy
        3. Group by region and category, and apply interpolation/extrapolation as specified
        4. Print each filled value with region, crop, year, and value
        5. Pivot back to wide format per year

    Returns:
    - dict with same structure as dt, with specified crops filled accordingly
    """

    # 1) Melt into long format
    records = []
    for year, df in dt.items():
        df = df.rename_axis('region')
        melted = (
            df
            .reset_index()
            .melt(id_vars='region', var_name='category', value_name='value')
        )
        melted['year'] = int(year)
        records.append(melted)

    df_all = pd.concat(records, ignore_index=True)
    df_all['year'] = df_all['year'].astype(int)
    df_all.sort_values(['region', 'category', 'year'], inplace=True)

    # 2) Backup before interpolation for reporting
    df_before = df_all.copy()

    # 3) Conditional interpolation/extrapolation
    def _fill(series: pd.Series) -> pd.Series:
        """
        Apply interpolation or both interpolation and extrapolation on the series.
        """
        # Only process specified categories
        (_, category) = series.name
        if category not in interpolate_crops:
            return series

        if extrapolate:
            # fill both inside and outside (leading/trailing)
            return series.interpolate(
                method='linear',
                limit_direction='both'
            )
        else:
            # only inside gaps, do not fill leading/trailing NaNs
            return series.interpolate(
                method='linear',
                limit_direction='both',
                limit_area='inside'
            )

    df_all['value'] = (
        df_all
        .groupby(['region', 'category'])['value']
        .transform(_fill)
    )

    # 4) Report filled values
    mask = df_before['value'].isna() & df_all['value'].notna()
    for _, row in df_all[mask].iterrows():
        action = 'Extrapolated' if extrapolate and (
            row.year == df_all[df_all['category'] == row.category]['year'].min()
            or row.year == df_all[df_all['category'] == row.category]['year'].max()
        ) else 'Interpolated'
        print(
            f"{action} for region={row.region}, crop={row.category}, "
            f"year={row.year}, value={row.value:.2f}"
        )

    # 5) Pivot back into dict by year
    filled_dict = {
        year: group.pivot(index='region', columns='category', values='value')
        for year, group in df_all.groupby('year')
    }

    return filled_dict


def fill_template(
    template_df: pd.DataFrame,
    data_dict: dict[int, pd.DataFrame],
    id_col: str = 'crop',
    confidence_label: str = 'high'
) -> pd.DataFrame:
    """
    Fill a template DataFrame with values from a dictionary of yearly DataFrames,
    and assign a confidence label only when the template value is changed.

    Parameters
    ----------
    template_df : pd.DataFrame
        Must contain columns ['region', id_col, 'year', 'value', 'label', 'unit', 'confidence'].
        'value' and 'confidence' may be pre-filled.
    data_dict : dict[int, pd.DataFrame]
        Mapping year → DataFrame with a 'region' column and columns for each id_col (e.g. crops or animals).
    id_col : str, default 'crop'
        Name of the category column in both template_df and data_dict (e.g. 'crop', 'animal').
    confidence_label : str, default 'high'
        Value to write into 'confidence' when a template value is replaced.

    Returns
    -------
    pd.DataFrame
        Same structure as template_df, with 'value' updated from data_dict when different,
        and 'confidence' set to confidence_label for those replaced entries.
    """
    # Build a long table of actual values from data_dict
    long_records = []
    for year, df in data_dict.items():
        # Ensure 'region' is a column
        if 'region' not in df.columns:
            df = df.reset_index().rename(columns={df.index.name or 'index': 'region'})
        # Melt the id_col (crop/animal) into rows
        df_long = df.melt(
            id_vars=['region'],
            var_name=id_col,
            value_name='value'
        )
        df_long['year'] = int(year)
        long_records.append(df_long[['region', id_col, 'year', 'value']])

    if not long_records:
        return template_df.copy()

    actual_values = pd.concat(long_records, ignore_index=True)

    # Merge template with the new values
    merged = template_df.merge(
        actual_values,
        on=['region', id_col, 'year'],
        how='left',
        suffixes=('', '_new')
    )

    # Where new value exists and differs, update and flag confidence
    orig = merged['value']
    new = merged['value_new']
    mask = new.notna() & ~new.eq(orig)

    merged.loc[mask, 'value'] = merged.loc[mask, 'value_new']
    merged.loc[mask, 'confidence'] = confidence_label

    # Drop the helper column and return
    merged.drop(columns=['value_new'], inplace=True)
    return merged


def clean_animal_eurostat(df, nuts):
    """
    Clean and reshape a Eurostat data table for NUTS regions.

    Parameters:
    - df : pandas.DataFrame, the raw Eurostat dataset for animals
    - nuts : pandas.DataFrame containing a column 'NUTS_ID' listing valid NUTS region

    Returns:
    pandas.DataFrame
        A filtered and restructured DataFrame where:
        - Every second (NaN) column has been dropped.
        - Only rows corresponding to the header row ('TIME') or valid NUTS IDs are kept.
        - The first row of remaining data is promoted to column headers (years).
        - The index is reset to provide a clean, zero-based index.
    """

    # Drop every second column (typically empty in Eurostat exports)
    df = df.drop(columns=df.columns[1::2])
    # Keep only rows for 'TIME' header or valid NUTS region codes
    df_filtered = df[df[0].isin(['TIME'] + list(nuts['NUTS_ID']))]
    df_filtered = df_filtered.reset_index(drop=True)
    # Use the first row of filtered data as new column names
    df_filtered.columns = df_filtered.iloc[0]
    # Remove the header row from the data
    df_filtered = df_filtered.drop(df_filtered.index[0]).reset_index(drop=True)
    return df_filtered


def calculate_excretion(animal_dataframes, path_excretion_coef):
    """
    Calculate nitrogen excretion (in GgN) by region and animal category for each year.

    Parameters:
    - animal_dataframes : dict[int, pandas.DataFrame]
        A dict mapping each year (int or str) to a DataFrame that must contain columns
        ['region', 'chickens', 'ducks', 'turkeys', 'bovine', 'dairycows', 'pigs', 'sheep', 'goats'].
     - path_excretion_coef : str

    Returns:
    - dict[int, pandas.DataFrame where each value is the excretion in GgN]
    """

    # Read excretion coefficients and compute a combined 'chickens' rate
    ex_coef = pd.read_csv(path_excretion_coef, sep=';')
    ex_coef['chickens'] = np.mean([ex_coef['broilers'], ex_coef['layinghens']], axis=0)

    # Define regions
    regions_ref = animal_dataframes[1990]['region']

    # Reorder the coefficient table to match the regions
    ex_coef = (
        ex_coef
        .set_index('ID')
        .reindex(regions_ref)
        .reset_index(drop=True)
    )

    # Prepare output dict
    excretion_by_year = {}

    # Loop over each year and compute excretion per animal
    for year, df in animal_dataframes.items():
        ex = pd.DataFrame()

        # Poultry categories
        ex['chickens'] = (df['chickens'] * ex_coef['chickens']) / 1000  # GgN
        ex['ducks'] = (df['ducks'] * ex_coef['otherpoultry']) / 1000
        ex['turkeys'] = (df['turkeys'] * ex_coef['turkey']) / 1000

        # Cattle categories
        ex['dairy'] = (df['dairycows'] * ex_coef['dairycows']) / 1000
        ex['bovine'] = (df['bovine'] * ex_coef['othercattle']) / 1000

        # Other livestock
        ex['pigs'] = (df['pigs'] * ex_coef['pigs']) / 1000
        ex['sheep'] = (df['sheep'] * ex_coef['sheep']) / 1000
        ex['goats'] = (df['goats'] * ex_coef['goats']) / 1000

        # Preserve region labels
        ex['region'] = df['region']

        # Select and order final columns
        excretion_by_year[int(year)] = ex.loc[:, [
            'region', 'dairy', 'bovine', 'pigs',
            'sheep', 'goats', 'chickens', 'ducks', 'turkeys'
        ]]

    return excretion_by_year


def sum_dicts_on_common_cols(
    d1: Mapping[int, pd.DataFrame],
    d2: Mapping[int, pd.DataFrame],
    cols: Iterable[str] | None = None,
    index_col: str = "region"
) -> dict[int, pd.DataFrame]:
    """
    Sum two dicts of DataFrames year‑by‑year, keeping only *cols*
    (or all common columns) and add a copy of *index_col* from d1
    to every resulting DataFrame.

    Parameters
    ----------
    d1, d2 : Mapping[int, pd.DataFrame]
        One DataFrame per year.
    cols : Iterable[str] | None, optional
        If provided, restrict the sum to these columns (intersection
        with the truly common columns). Default = None = all common cols.
    index_col : str, optional
        Column name to copy from d1 into the output (default "region").

    Returns
    -------
    dict[int, pd.DataFrame]
        Year → summed DataFrame (with *index_col* inserted as the first column).
    """
    out: dict[int, pd.DataFrame] = {}

    for year in d1.keys() & d2.keys():
        df1 = d1[year]
        df2 = d2[year]

        # Determine columns to sum
        common_cols = df1.columns.intersection(df2.columns)
        if cols is not None:
            common_cols = common_cols.intersection(cols)

        # Element‑wise sum (alignment on index; fill_value only for missing labels)
        summed = df1[common_cols].add(df2[common_cols], fill_value=0)

        # Add the index column at the front, if it exists in d1
        if index_col in df1.columns:
            summed.insert(0, index_col, df1[index_col])

        out[year] = summed

    return out



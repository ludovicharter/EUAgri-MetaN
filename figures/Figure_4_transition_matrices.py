#%% --- Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%% --- Import data ---
regions = pd.read_csv('data/regions.csv', sep=';')
typologies = pd.read_csv('data/outputs/typologies.csv')
metabolic = pd.read_csv('data/outputs/metabolic_data.csv')

# %% --- Function to compute average for a given period ---
def compute_period_average(df, start_year, end_year, period_label):
    """
    Compute average values of the typology for a given time period.

    Parameters:
        df (pd.DataFrame): Input dataframe containing 'year', 'value', and grouping columns.
        start_year (int): Start year of the period.
        end_year (int): End year of the period.
        period_label (str): Label to assign to the period (e.g., "1990–1994").

    Returns:
        pd.DataFrame: Grouped dataframe with average values and period label.
    """
    mask = df["year"].between(start_year, end_year, inclusive="both")

    grouped = (
        df.loc[mask]
        .copy()
    )
    grouped["value"] = pd.to_numeric(grouped["value"], errors="coerce")

    averaged = (
        grouped
        .groupby(["region", "region name", "symbol", "label", "unit"], as_index=False)["value"]
        .mean()
    )
    averaged["period"] = period_label

    return averaged

# %% --- Compute period averages ---
df_9094 = compute_period_average(metabolic, 1990, 1994, "1990-1994")
df_1519 = compute_period_average(metabolic, 2015, 2019, "2015-2019")

# %% --- Merge both periods ---
period_means = pd.concat([df_9094, df_1519], ignore_index=True)

# Columns
base_cols = ['region', 'region name', 'period']
new_rows = period_means[base_cols].drop_duplicates().copy()

# Add columns
new_rows['symbol'] = 'typ'
new_rows['label'] = 'typology'
new_rows['unit'] = 'no unit'
new_rows['value'] = np.nan

# Rearrange columns as in period_means
new_rows = new_rows[period_means.columns]
period_means = pd.concat([period_means, new_rows], ignore_index=True)

#%% --- Compute typologies ---

def get_typology_values(region, period, data):
    """Retrieve all needed values for typology classification."""
    def get_value(symbol):
        mask = (
            (data['region'] == region) &
            (data['period'] == period) &
            (data['symbol'] == symbol)
        )
        values = data.loc[mask, 'value']
        return values.squeeze() if not values.empty else None

    symbols = ['H_ingestion', 'Agri_total', 'H_total', 'total_A_ingestion', 'L_density', 'F_import', 'PG_H',
               'F_ingestion', 'total_N_input']
    return {symbol: get_value(symbol) for symbol in symbols}

# List of periods to process
periods = ['1990-1994', '2015-2019']

for region in regions['NUTS_ID']:
    print(f'Processing region {region}...')

    for period in periods:
        values = get_typology_values(region, period, period_means)

        # Skip if any value is missing
        if any(v is None for v in values.values()):
            print(f"  Skipped {period} (incomplete data)")
            continue

        # Unpack values
        H, C, P, A, D, I, G, F, N = [values[k] for k in ['H_ingestion', 'Agri_total', 'H_total', 'total_A_ingestion',
                                                         'L_density', 'F_import', 'PG_H', 'F_ingestion', 'total_N_input']]

        # Determine typology
        if P > (1.5 * A):
            typology = 'SCS'
        elif (D > 1) and (I > (0.33 * A)):
            typology = 'LVK'
        elif G > (0.5 * A):
            typology = 'MXG'
        elif (F > (0.25 * A)) and (N > 30):
            typology = 'MXF'
        else:
            typology = 'DSG'

        # Apply typology classification
        idx = (
            (period_means['region'] == region) &
            (period_means['period'] == period) &
            (period_means['symbol'] == 'typ')
        )
        period_means.loc[idx, 'value'] = typology

#%% --- Typology transitions ---

# Filter only the typology rows ('T') from period_means
typologies_only = period_means[period_means['symbol'] == 'typ'].copy()

# Pivot the DataFrame to wide format
typology_transition = typologies_only.pivot(
    index='region',
    columns='period',
    values='value'
).reset_index()

# Optional: rename columns for clarity
typology_transition.columns.name = None  # Remove pandas' column name from pivot
typology_transition = typology_transition.rename(columns={
    '1990-1994': 'typology_1990_1994',
    '2015-2019': 'typology_2015_2019'
})

# %% --- Compute mean Surplus (S) and Nitrogen Use Efficiency (NUE) over full period (1990–2019) ---

# Keep only S and NUE
s_nue_all_years = period_means[period_means['symbol'].isin(['NS', 'NUE'])].copy()
s_nue_all_years['value'] = pd.to_numeric(s_nue_all_years['value'], errors='coerce')

# Compute mean per region and symbol over full period
s_nue_means = (
    s_nue_all_years
    .groupby(['region', 'symbol'], as_index=False)['value']
    .mean()
    .pivot(index='region', columns='symbol', values='value')
    .rename(columns={
        'NS': 'mean_surplus_1990_2019',
        'NUE': 'mean_NUE_1990_2019'
    })
    .reset_index()
)

# Merge with typology transition table
typology_transition = typology_transition.merge(s_nue_means, on='region', how='left')

# %% --- Compute deltas between 1990–1994 and 2015–2019 for S and NUE ---

# Filter S and NUE for the two target periods only
subset = period_means[
    period_means['symbol'].isin(['NS', 'NUE']) &
    period_means['period'].isin(periods)
].copy()
subset['value'] = pd.to_numeric(subset['value'], errors='coerce')

# Compute mean per region, symbol and period
pivot_diff = (
    subset
    .groupby(['region', 'symbol', 'period'], as_index=False)['value']
    .mean()
    .pivot(index='region', columns=['symbol', 'period'], values='value')
)

# Flatten MultiIndex columns
pivot_diff.columns = [f"{symbol}_{period}" for symbol, period in pivot_diff.columns]
pivot_diff = pivot_diff.reset_index()

# Compute delta between the two periods
pivot_diff['delta_NS'] = pivot_diff['NS_2015-2019'] - pivot_diff['NS_1990-1994']
pivot_diff['delta_NUE'] = pivot_diff['NUE_2015-2019'] - pivot_diff['NUE_1990-1994']

# Keep only relevant columns
deltas = pivot_diff[['region', 'delta_NS', 'delta_NUE']]

# Merge with main transition DataFrame
typology_transition = typology_transition.merge(deltas, on='region', how='left')

#%% Figure

# Typology categories
types = ["SCS", "LVK", "MXG", "MXF", "DSG"]

# Build transition frequency matrix
freq_matrix = pd.crosstab(
    typology_transition["typology_1990_1994"],
    typology_transition["typology_2015_2019"]
).reindex(index=types, columns=types).fillna(0)

# Build delta NUE matrix
delta_NUE_matrix = pd.DataFrame(0, index=types, columns=types, dtype=float)
for i in types:
    for j in types:
        subset = typology_transition[
            (typology_transition["typology_1990_1994"] == i) &
            (typology_transition["typology_2015_2019"] == j)
        ]
        delta_NUE_matrix.loc[i, j] = subset["delta_NUE"].mean() if len(subset) > 0 else np.nan

# Build delta NS matrix
delta_NS_matrix = pd.DataFrame(0, index=types, columns=types, dtype=float)
for i in types:
    for j in types:
        subset = typology_transition[
            (typology_transition["typology_1990_1994"] == i) &
            (typology_transition["typology_2015_2019"] == j)
        ]
        delta_NS_matrix.loc[i, j] = subset["delta_NS"].mean() if len(subset) > 0 else np.nan

# Build probabilities transition matrix
proba_matrix = freq_matrix.div(freq_matrix.sum(axis=1), axis=0)

# Compute standard deviations per transition
std_NUE_matrix = pd.DataFrame(np.nan, index=types, columns=types)
std_NS_matrix = pd.DataFrame(np.nan, index=types, columns=types)

for i in types:
    for j in types:
        subset = typology_transition[
            (typology_transition["typology_1990_1994"] == i) &
            (typology_transition["typology_2015_2019"] == j)
        ]
        if len(subset) > 0:
            std_NUE_matrix.loc[i, j] = subset["delta_NUE"].std()
            std_NS_matrix.loc[i, j] = subset["delta_NS"].std()

# Prepare annotation
def make_annot(mean_mat, std_mat, fmt_mean="{:.2f}", fmt_std="{:.2f}"):
    annot = mean_mat.copy().astype(str)
    for i in mean_mat.index:
        for j in mean_mat.columns:
            mean_val = mean_mat.loc[i, j]
            std_val = std_mat.loc[i, j]
            if pd.isna(mean_val):
                annot.loc[i, j] = ""
            elif pd.isna(std_val):
                annot.loc[i, j] = fmt_mean.format(mean_val)
            else:
                annot.loc[i, j] = f"{fmt_mean.format(mean_val)} \n ± {fmt_std.format(std_val)}"
    return annot

annot_NUE = make_annot(delta_NUE_matrix, std_NUE_matrix, fmt_mean="{:.2f}", fmt_std="{:.2f}")
annot_NS = make_annot(delta_NS_matrix, std_NS_matrix, fmt_mean="{:.1f}", fmt_std="{:.1f}")

#%% Plot
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# Transition frequencies
sns.heatmap(freq_matrix, annot=True, fmt=".0f", cmap="Blues", cbar=True, ax=axes[0, 0], mask=freq_matrix==0, annot_kws={"fontsize":14})
axes[0, 0].set_title("a) Transition frequencies (1990-1994 → 2015-2019)", fontsize=19, fontweight='bold')
axes[0, 0].set_xlabel("Typology 2015-2019", fontsize=18)
axes[0, 0].set_ylabel("Typology 1990-1994", fontsize=18)
cbar = axes[0, 0].collections[0].colorbar
cbar.set_label("Number of territories", fontsize=18)
cbar.ax.tick_params(labelsize=14)

# Prob. transition matrix
sns.heatmap(proba_matrix, annot=True, fmt=".2f", cmap="Greens", cbar=True, ax=axes[0, 1], mask=proba_matrix==0, annot_kws={"fontsize":14})
axes[0, 1].set_title("b) Transition probabilities (1990-1994 → 2015-2019)", fontsize=19, fontweight='bold')
axes[0, 1].set_xlabel("Typology 2015-2019", fontsize=18)
axes[0, 1].set_ylabel("Typology 1990-1994", fontsize=18)
cbar = axes[0, 1].collections[0].colorbar
cbar.set_label("Transition probability", fontsize=18)
cbar.ax.tick_params(labelsize=14)

# Delta NUE
sns.heatmap(delta_NUE_matrix, annot=annot_NUE, fmt="", cmap="coolwarm_r", center=0, ax=axes[1, 0], annot_kws={"fontsize":14})
axes[1, 0].set_title("c) Δ mean NUE per transition (Δ ratio)", fontsize=19, fontweight='bold')
axes[1, 0].set_xlabel("Typology 2015-2019", fontsize=18)
axes[1, 0].set_ylabel("Typology 1990-1994", fontsize=18)
cbar = axes[1, 0].collections[0].colorbar
cbar.set_label("Δ ratio", fontsize=18)
cbar.ax.tick_params(labelsize=14)

# Delta NS
sns.heatmap(delta_NS_matrix, annot=annot_NS, fmt="", cmap="coolwarm", center=0, ax=axes[1, 1], annot_kws={"fontsize":14})
axes[1, 1].set_title("d) Δ mean NS per transition (Δ kg N/ha)", fontsize=19, fontweight='bold')
axes[1, 1].set_xlabel("Typology 2015-2019", fontsize=18)
axes[1, 1].set_ylabel("Typology 1990-1994", fontsize=18)
cbar = axes[1, 1].collections[0].colorbar
cbar.set_label("Δ kg N/ha", fontsize=18)
cbar.ax.tick_params(labelsize=14)

# Common ticks
for ax in axes.flat:
    ax.tick_params(axis='both', labelsize=18)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

plt.subplots_adjust(
    left=0.08,
    right=0.95,
    top=0.93,
    bottom=0.08,
    wspace=0.20,
    hspace=0.25
)

#plt.tight_layout()
plt.savefig("figures/outputs/Figure_4_transition_matrices.png", dpi=400)

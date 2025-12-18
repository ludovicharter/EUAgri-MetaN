"""
Script name: Figure_5_data_processing.py
Description: Figure 5 workflow
Author: Ludovic Harter
Created: 2025-12-18
Last modified: 2025-12-18
Version: 1.0
Project: Territorial nitrogen flows and metabolic typologies of EU Agri-Food Systems, 1990–2019
License: MIT
"""

#%% --- Libraries ---
import pandas as pd
import matplotlib.pyplot as plt

# Style
plt.style.use("seaborn-v0_8-whitegrid")

#%% --- Parameters ---
custom_order = [
    "unprocessed", "filled (a)", "filled (b)", "filled (c)", "filled (d)",
    "corrected (e)", "corrected (e\')", "corrected (f)", "interpolated", "no data"
]

tab20 = plt.cm.tab20.colors
manual_colors = [
    tab20[0],   # dark blue
    tab20[1],   # medium blue
    tab20[2],   # dark orange
    tab20[3],   # medium orange
    tab20[4],   # dark green
    tab20[5],   # medium green
    tab20[9],   # light purple
    tab20[6],   # dark red
    tab20[7],   # light red
    tab20[8],   # dark purple
]

color_map = {
    cat: manual_colors[i] for i, cat in enumerate(custom_order)
}

#%% --- Import data ---
animal_excretion = pd.read_csv('data/outputs/animal_excretion.csv')
crop_production = pd.read_csv('data/outputs/crop_production_all_categories.csv')
synthetic_fertilizer = pd.read_csv('data/outputs/synthetic_fertilizer.csv')

# Global shares
for df in [animal_excretion, crop_production, synthetic_fertilizer]:
    df["confidence"] = df["confidence"].fillna("no data").astype(str)

datasets = {
    "All animal excretion": animal_excretion,
    "All harvested area": crop_production[crop_production["symbol"] == "A"],
    "All harvested quantities": crop_production[crop_production["symbol"] == "H"],
    "Synthetic fertilizer": synthetic_fertilizer[synthetic_fertilizer["symbol"] == "Q"]
}

ratios_global = pd.DataFrame({
    name: (df["confidence"].value_counts(normalize=True) * 100)
    .reindex(custom_order, fill_value=0)
    for name, df in datasets.items()
}).T

#%% --- 1. Harvested quantities ---

dataH = crop_production[crop_production["symbol"] == "H"].copy()
dataH["confidence"] = dataH["confidence"].fillna("no data").astype(str)
dataH["crop"] = dataH["crop"].fillna("Unknown crop").astype(str)

ratios_crop = (
    dataH.groupby(["crop", "confidence"]).size()
    .groupby(level=0)
    .apply(lambda x: 100 * x / x.sum())
    .unstack(fill_value=0)
    .reindex(columns=custom_order, fill_value=0)
)

#%% --- 2. Animal excretion ---

dataA = animal_excretion.copy()
if "symbol" in animal_excretion.columns:
    dataA = animal_excretion[animal_excretion["symbol"] == "A"].copy()

dataA["confidence"] = dataA["confidence"].fillna("no data").astype(str)
dataA["animal"] = dataA["animal"].fillna("Unknown animal").astype(str)

ratios_animal = (
    dataA.groupby(["animal", "confidence"]).size()
    .groupby(level=0)
    .apply(lambda x: 100 * x / x.sum())
    .unstack(fill_value=0)
    .reindex(columns=custom_order, fill_value=0)
)

#%% --- Fusion ---

def simplify_index(df, prefix):
    if isinstance(df.index, pd.MultiIndex):
        df.index = [prefix + str(i) for i in df.index.get_level_values(0)]
    else:
        df.index = [prefix + str(i) for i in df.index]
    return df

ratios_crop = simplify_index(ratios_crop, prefix="")
ratios_animal = simplify_index(ratios_animal, prefix="")

combined = pd.concat([ratios_global, ratios_crop, ratios_animal], axis=0)

# --- Calcul separation position ---
split1 = len(ratios_global)
split2 = split1 + len(ratios_crop)

#%% Plot

fig, ax = plt.subplots(figsize=(max(14, len(combined) * 0.3), 7))

x = range(len(combined))
bottom = [0.0] * len(combined)

for cat in custom_order:
    values = combined[cat].values if cat in combined.columns else [0.0] * len(combined)
    ax.bar(
        x,
        values,
        bottom=bottom,
        color=color_map[cat],
        edgecolor="white",
        width=0.8
    )
    bottom = [b + v for b, v in zip(bottom, values)]

ax.set_ylabel("Share of processing category (%)", fontsize=18)
ax.set_ylim(0, 100)
ax.set_xticks(x)
ax.set_title("Share of processing category across datasets", pad=15, fontsize=18)
ax.grid(axis="y", linestyle="--", alpha=0.7)
ax.tick_params(axis="both", labelsize=16)

# Separation lines
for pos in [split1 - 0.5, split2 - 0.5]:
    ax.axvline(x=pos, color="black", linestyle="--", linewidth=2, alpha=0.7)

xtick_labels = combined.index.tolist()

# Color text
colors = []
for i in range(len(xtick_labels)):
    if i < split1:
        colors.append("black")
    elif i < split2:
        colors.append("#1b9e77")
    else:
        colors.append("#d95f02")

for tick, label, color in zip(ax.get_xticks(), xtick_labels, colors):
    ax.text(
        tick, -5, label,
        rotation=90,
        ha="center", va="top",
        fontsize=16,
        color=color
    )

ax.set_xticklabels([])

# legend

handles = [
    plt.Line2D([0], [0], marker="s", color=color_map[cat], label=cat,
               markersize=14, linestyle="None")
    for cat in custom_order
]
ax.legend(
    handles=handles[::-1],
    bbox_to_anchor=(1.02, 0.5),
    loc="center left",
    frameon=False,
    fontsize=16
)

fig.tight_layout()
plt.savefig("figures/outputs/Figure_5_data_processing.png", dpi=400)

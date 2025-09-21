# app.py
# Visual 2: Overall Alternative Energy Adoption by Governorate (stacked; % math fixed; %-aware sorting)
# Visual 3: Adoption vs Power Grid State — now with energy-type multiselect + Top Towns list
# Town names cleaned from links/URIs. No upload UI — reads 325 data.csv.

import os
import re
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

st.set_page_config(
    page_title="Municipal Energy Explorer — Visuals 2 & 3",
    layout="wide",
    page_icon="🔌",
)

DATA_FILE = "325 data.csv"

# ----------------------------
# Helpers
# ----------------------------
def extract_name_from_link(val: str) -> str:
    """Turn a URL/DBpedia URI into a readable name; otherwise return trimmed text."""
    if pd.isna(val):
        return val
    s = str(val).strip()
    if s.startswith(("http://", "https://")) or "dbpedia.org" in s:
        seg = s.rsplit("/", 1)[-1]
        seg = seg.split("?")[0].split("#")[0].replace("_", " ")
        seg = re.sub(r"\s*\((?:[^)]*)\)\s*$", "", seg).strip()
        return seg
    return s

@st.cache_data(show_spinner=False)
def load_data():
    if not os.path.exists(DATA_FILE):
        st.error(f"CSV not found: {DATA_FILE}")
        st.stop()

    df = pd.read_csv(DATA_FILE)
    df.columns = [c.strip() for c in df.columns]

    # Clean Town column
    town_col = None
    for cand in ["Town", "town", "Municipality", "municipality"]:
        if cand in df.columns:
            town_col = cand
            break
    if town_col is None:
        st.error("Could not find a 'Town' column. Please ensure your CSV has one.")
        st.stop()
    df[town_col] = df[town_col].apply(extract_name_from_link)

    # Governorate
    if "refArea" in df.columns:
        def extract_gov(x):
            if pd.isna(x): return np.nan
            seg = str(x).rsplit("/", 1)[-1].replace("_", " ")
            seg = seg.replace("Governorate", "").strip()
            seg = re.sub(r"\s*\((?:[^)]*)\)\s*$", "", seg).strip()
            return seg
        df["Governorate"] = df["refArea"].apply(extract_gov)
    elif "Governorate" not in df.columns:
        df["Governorate"] = np.nan

    COLS = {
        "town": town_col,
        "solar": "Type of alternative energy used - solar energy",
        "wind": "Type of alternative energy used - wind energy",
        "hydro": "Type of alternative energy used - hydropower (water use)",
        "other": "Type of alternative energy used - other",
        "grid_good": "State of the power grid - good",
        "grid_ok": "State of the power grid - acceptable",
        "grid_bad": "State of the power grid - bad",
    }

    type_cols = [COLS["solar"], COLS["wind"], COLS["hydro"], COLS["other"]]
    grid_cols = [COLS["grid_good"], COLS["grid_ok"], COLS["grid_bad"]]
    for c in type_cols + grid_cols:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    return df, COLS

# ----------------------------
# Load data
# ----------------------------
df, COLS = load_data()

st.title("🔌 Municipal Energy Explorer")
st.caption("Visuals 2 (overall adoption) and 3 (adoption vs grid). Town names are cleaned from links.")

# Precompute helper columns for convenience
TYPE_COLS = [COLS["solar"], COLS["wind"], COLS["hydro"], COLS["other"]]

# ======================================================
# VISUAL 2 — Overall Alternative Energy Adoption by Governorate (fixed % math, %-aware sorting)
# ======================================================
st.markdown("### 2) Overall Alternative Energy Adoption — by Governorate")

energy_map_display = {
    "Solar": COLS["solar"],
    "Wind":  COLS["wind"],
    "Hydro": COLS["hydro"],
    "Other": COLS["other"],
}

left, mid = st.columns([2, 2])
with left:
    normalize = st.checkbox(
        "Show as % of towns using any alternative energy",
        value=True,
        help="Each adopting town contributes a total of 1, split across the types it uses."
    )
with mid:
    sort_by = st.selectbox(
        "Sort governorates by",
        ["Total adopting towns", "Solar", "Wind", "Hydro", "Other", "Alphabetical"],
        index=0
    )

# Raw counts per governorate
grp_counts = (
    df.groupby("Governorate", dropna=False)[list(energy_map_display.values())]
    .sum()
    .reset_index()
    .rename(columns={
        COLS["solar"]: "Solar",
        COLS["wind"]: "Wind",
        COLS["hydro"]: "Hydro",
        COLS["other"]: "Other",
    })
)

# # adopting towns (≥1 type across all)
df_any_all = df.assign(types_count=df[TYPE_COLS].sum(axis=1))
adopting_per_gov = (
    df_any_all.groupby("Governorate", dropna=False)["types_count"]
    .apply(lambda s: (s > 0).sum())
    .reset_index()
    .rename(columns={"types_count": "Total adopting towns"})
)
grp_counts = grp_counts.merge(adopting_per_gov, on="Governorate", how="left")

# Normalized contributions (each adopting town contributes 1 split across its present types)
if normalize:
    any_mask = df_any_all["types_count"] > 0
    df_norm = df_any_all.loc[any_mask, ["Governorate", *list(energy_map_display.values()), "types_count"]].copy()
    for c in list(energy_map_display.values()):
        df_norm[c] = df_norm[c] / df_norm["types_count"]

    grp_norm = (
        df_norm.groupby("Governorate", dropna=False)[list(energy_map_display.values())]
        .sum()
        .reset_index()
        .rename(columns={
            COLS["solar"]: "Solar",
            COLS["wind"]: "Wind",
            COLS["hydro"]: "Hydro",
            COLS["other"]: "Other",
        })
    ).merge(adopting_per_gov, on="Governorate", how="left")

    tall = grp_norm.melt(
        id_vars=["Governorate", "Total adopting towns"],
        value_vars=["Solar", "Wind", "Hydro", "Other"],
        var_name="Energy type",
        value_name="town_equivalents"
    )
    tall["pct"] = np.where(
        tall["Total adopting towns"] > 0,
        tall["town_equivalents"] / tall["Total adopting towns"],
        0.0
    )
else:
    tall = grp_counts.melt(
        id_vars=["Governorate", "Total adopting towns"],
        value_vars=["Solar", "Wind", "Hydro", "Other"],
        var_name="Energy type",
        value_name="count"
    )

# %-aware sorting
if sort_by == "Alphabetical":
    sort_order = sorted(grp_counts["Governorate"].fillna("Unknown").tolist())
elif sort_by == "Total adopting towns":
    sort_order = grp_counts.sort_values("Total adopting towns", ascending=False)["Governorate"].tolist()
else:
    metric_col = sort_by  # "Solar", "Wind", "Hydro", or "Other"
    if normalize:
        pct_table = tall[tall["Energy type"] == metric_col][["Governorate", "pct"]].copy()
        pct_table["pct"] = pct_table["pct"].fillna(0)
        sort_order = pct_table.sort_values("pct", ascending=False)["Governorate"].tolist()
    else:
        cnt_table = tall[tall["Energy type"] == metric_col][["Governorate", "count"]].copy()
        cnt_table["count"] = cnt_table["count"].fillna(0)
        sort_order = cnt_table.sort_values("count", ascending=False)["Governorate"].tolist()

# Chart
if normalize:
    y = alt.Y("pct:Q", axis=alt.Axis(format="%"), title="Share of adopting towns")
    tooltip = [
        "Governorate:N", "Energy type:N",
        alt.Tooltip("pct:Q", format=".0%"),
        "Total adopting towns:Q",
    ]
else:
    y = alt.Y("count:Q", title="Number of towns")
    tooltip = ["Governorate:N", "Energy type:N", "count:Q", "Total adopting towns:Q"]

chart2 = (
    alt.Chart(tall)
    .mark_bar()
    .encode(
        x=alt.X("Governorate:N", sort=sort_order, title="Governorate"),
        y=y,
        color=alt.Color("Energy type:N", legend=alt.Legend(title="Energy type")),
        tooltip=tooltip,
    )
    .properties(height=460)
)
st.altair_chart(chart2, use_container_width=True)

st.divider()

# ======================================================
# VISUAL 3 — Adoption vs. Power Grid State
# Added: Multiselect for energy types + "Top Towns" list
# ======================================================
st.markdown("### 3) Alternative Energy Adoption vs. Power Grid State")

# Controls for Visual 3
opt_col1, opt_col2, opt_col3 = st.columns([2, 2, 2])
with opt_col1:
    energy_choice = st.multiselect(
        "Energy types to include",
        ["Solar", "Wind", "Hydro", "Other"],
        default=["Solar", "Wind", "Hydro", "Other"]
    )
with opt_col2:
    grid_for_list = st.selectbox("Filter Top Towns by grid state", ["(All)", "good", "acceptable", "bad"], index=0)
with opt_col3:
    top_n = st.slider("How many top towns to show", min_value=5, max_value=50, value=15, step=5)

# Map choices to columns
etype_to_col = {"Solar": COLS["solar"], "Wind": COLS["wind"], "Hydro": COLS["hydro"], "Other": COLS["other"]}
selected_cols = [etype_to_col[e] for e in energy_choice] if energy_choice else []

# Adoption flag based on selected energy types
if selected_cols:
    adopt_sum = df[selected_cols].sum(axis=1)
else:
    # If nothing selected, treat as "no types" -> no one adopts
    adopt_sum = pd.Series(0, index=df.index)

adopt_flag = (adopt_sum > 0).astype(int)

# Build grid-state adoption rates
grid_map = {"good": COLS["grid_good"], "acceptable": COLS["grid_ok"], "bad": COLS["grid_bad"]}
rows = []
for label, col in grid_map.items():
    denom = int((df[col] == 1).sum())
    numer = int(((df[col] == 1) & (adopt_flag == 1)).sum())
    rate = (numer / denom) if denom else 0.0
    rows.append({"Power Grid State": label, "Adoption Rate": rate, "Towns in state": denom, "Adopting towns": numer})
v3 = pd.DataFrame(rows)
v3["Power Grid State"] = pd.Categorical(v3["Power Grid State"], ["good", "acceptable", "bad"], ordered=True)

# Chart (percent)
chart3 = (
    alt.Chart(v3)
    .mark_bar()
    .encode(
        x=alt.X("Power Grid State:N", sort=["good", "acceptable", "bad"], title="Power Grid State"),
        y=alt.Y("Adoption Rate:Q", axis=alt.Axis(format="%"), title="Adoption Rate"),
        tooltip=["Power Grid State:N",
                 alt.Tooltip("Adoption Rate:Q", format=".1%"),
                 "Adopting towns:Q", "Towns in state:Q"],
    )
    .properties(height=360)
)
st.altair_chart(chart3, use_container_width=True)

# ----------------------------
# Top Towns list (rank by number of selected energy types present)
# ----------------------------
def row_grid_label(row) -> str:
    if row[COLS["grid_good"]] == 1:
        return "good"
    if row[COLS["grid_ok"]] == 1:
        return "acceptable"
    if row[COLS["grid_bad"]] == 1:
        return "bad"
    return "unknown"

# Score: how many of the selected energy types a town has (0..len(selected))
score = adopt_sum if selected_cols else pd.Series(0, index=df.index)

town_table = pd.DataFrame({
    "Town": df[COLS["town"]],
    "Governorate": df.get("Governorate", pd.Series([""]*len(df))),
    "Grid State": df.apply(row_grid_label, axis=1),
    "Score (selected types present)": score.astype(int),
    "Solar": df[COLS["solar"]].astype(int),
    "Wind": df[COLS["wind"]].astype(int),
    "Hydro": df[COLS["hydro"]].astype(int),
    "Other": df[COLS["other"]].astype(int),
})

# Keep only adopting towns (based on selection)
town_table = town_table[town_table["Score (selected types present)"] > 0]

# Optional grid-state filter
if grid_for_list != "(All)":
    town_table = town_table[town_table["Grid State"] == grid_for_list]

# Rank & show top N
town_table = town_table.sort_values(
    by=["Score (selected types present)", "Town"],
    ascending=[False, True]
).head(top_n)

with st.expander("Top Towns (based on your selection)"):
    st.dataframe(town_table, use_container_width=True)

st.caption("Reads data from 325 data.csv in this folder. Town names are auto-cleaned from links.")

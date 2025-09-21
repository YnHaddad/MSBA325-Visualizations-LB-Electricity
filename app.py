# app.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path
from urllib.parse import unquote

st.set_page_config(page_title="Lebanon Energy — Visual 2 & 4", page_icon="⚡", layout="wide")
st.title("⚡ Lebanon Energy — Interactive Visuals (Visual 2 & Visual 4)")

# -------------------------------- Config --------------------------------
DATA_PATH = Path("325 data.csv")  # <- change this if your CSV has a different name

# ---------------- Helpers ----------------
def load_csv_local(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None

def coerce_bool(s: pd.Series) -> pd.Series:
    if s is None: return pd.Series([False] * 0, dtype=bool)
    if pd.api.types.is_bool_dtype(s): return s.fillna(False)
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().any(): return (s_num.fillna(0) != 0)
    return s.astype(str).str.strip().str.lower().isin({"yes","y","true","t","1"})

def clean_town_name(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    tail = s.str.replace(r'/*[?#].*$', '', regex=True).str.replace(r'/*$', '', regex=True)
    tail = tail.str.extract(r'([^/]+)$')[0].fillna(s)
    tail = tail.map(lambda x: unquote(x) if isinstance(x, str) else x)
    return (tail
            .str.replace('_',' ',regex=False)
            .str.replace('-',' ',regex=False)
            .str.replace(r'\s+',' ',regex=True)
            .str.strip())

def derive_onehot_label(row: pd.Series, mapping: dict, default="Unknown"):
    for label, col in mapping.items():
        if col and col in row and pd.notna(row[col]) and coerce_bool(pd.Series([row[col]])).iloc[0]:
            return label
    return default

# ---------------- Load data (local only; no uploader) ----------------
df = load_csv_local(DATA_PATH)
if df is None:
    st.error(f"Could not find **{DATA_PATH}**. Place your CSV next to `app.py` (or update `DATA_PATH`).")
    st.stop()
df = df.copy()

# ---------------- Cleaned town + governorate ----------------
if "Town_clean" in df.columns:
    df["_town_clean"] = df["Town_clean"].astype(str)
elif "Town" in df.columns:
    df["_town_clean"] = clean_town_name(df["Town"])
elif "Observation URI" in df.columns:
    df["_town_clean"] = clean_town_name(df["Observation URI"])
else:
    guess_col = next((c for c in df.columns if "town" in c.lower() or "uri" in c.lower()), df.columns[0])
    df["_town_clean"] = clean_town_name(df[guess_col])

if "refArea_clean" in df.columns:
    COL_GOV = "refArea_clean"
elif "refArea" in df.columns:
    df["_gov_clean"] = clean_town_name(df["refArea"])
    COL_GOV = "_gov_clean"
else:
    COL_GOV = df.columns[0]

# ---------------- Column constants ----------------
COL_EXISTS_YES = "Existence of alternative energy - exists"
COL_EXISTS_NO  = "Existence of alternative energy - does not exist"
COL_GRID_GOOD  = "State of the power grid - good"
COL_GRID_OK    = "State of the power grid - acceptable"
COL_GRID_BAD   = "State of the power grid - bad"
COL_LIGHT_GOOD = "State of the lighting network - good"
COL_LIGHT_OK   = "State of the lighting network - acceptable"
COL_LIGHT_BAD  = "State of the lighting network - bad"

ENERGY_COLS = [c for c in [
    "Type of alternative energy used - solar energy",
    "Type of alternative energy used - wind energy",
    "Type of alternative energy used - hydropower (water use)",
    "Type of alternative energy used - other",
] if c in df.columns]

# Guardrails
for req in [
    COL_GOV, COL_EXISTS_YES, COL_EXISTS_NO,
    COL_GRID_GOOD, COL_GRID_OK, COL_GRID_BAD,
    COL_LIGHT_GOOD, COL_LIGHT_OK, COL_LIGHT_BAD
]:
    if req not in df.columns:
        st.error(f"Missing required column: **{req}**")
        st.stop()
if not ENERGY_COLS:
    st.error("No energy-type columns found.")
    st.stop()

# ---------------- Derivations ----------------
exists_yes = coerce_bool(df[COL_EXISTS_YES])
exists_no  = coerce_bool(df[COL_EXISTS_NO])
df["_adoption_pct"] = np.where(exists_yes, 100.0, np.where(exists_no, 0.0, np.nan))
df["_grid_label"] = df.apply(
    lambda r: derive_onehot_label(r, {"Good": COL_GRID_GOOD, "Acceptable": COL_GRID_OK, "Bad": COL_GRID_BAD}), axis=1
)
df["_light_label"] = df.apply(
    lambda r: derive_onehot_label(r, {"Good": COL_LIGHT_GOOD, "Acceptable": COL_LIGHT_OK, "Bad": COL_LIGHT_BAD}), axis=1
)

# ---------------- Sidebar filters (with search) ----------------
govs_all = sorted(df[COL_GOV].dropna().astype(str).unique().tolist())
with st.sidebar:
    st.header("🔎 Filters")
    gov_search = st.text_input("Search governorate", placeholder="type to filter…").strip().lower()
    if gov_search:
        gov_options = [g for g in govs_all if gov_search in g.lower()]
        if not gov_options:
            st.caption("No matches — showing all.")
            gov_options = govs_all
    else:
        gov_options = govs_all
    sel_govs = st.multiselect("Filter by governorate", options=gov_options, default=gov_options)
    if not sel_govs:
        sel_govs = gov_options  # ensure at least something selected

    min_adopt = st.slider("Minimum adoption rate (%)", 0, 100, 0, 1)

    st.markdown("---")
    st.subheader("🔥 Visual 4 Options")
    norm_axis = st.radio("Normalize heatmap by", ["None", "Grid state", "Lighting state"], horizontal=True)

mask = df[COL_GOV].astype(str).isin(sel_govs) & (df["_adoption_pct"].fillna(-1) >= min_adopt)
df_filt = df.loc[mask].copy()

tab2, tab4 = st.tabs(["📊 Visual 2 — Stacked Bar by Governorate", "🧯 Visual 4 — Heatmap: Grid × Lighting"])

# ---------------- Visual 2 (Percent-only stacked) ----------------
with tab2:
    st.subheader("Visual 2 — Alternative Energy by Governorate (Share within governorate)")
    long_frames = []
    for c in ENERGY_COLS:
        flag = coerce_bool(df_filt[c])
        towns = (
            df_filt.loc[flag, [COL_GOV, "_town_clean"]]
            .groupby(COL_GOV, dropna=False)["_town_clean"]
            .agg(lambda s: list(s[:12]))  # sample towns for tooltip
            .reset_index()
            .rename(columns={"_town_clean": "Towns (sample)"})
        )
        counts = (
            df_filt.assign(_flag=flag)
                  .groupby(COL_GOV, dropna=False)["_flag"]
                  .sum()
                  .rename("Count")
                  .reset_index()
        )
        merged = counts.merge(towns, on=COL_GOV, how="left")
        merged["Energy Type"] = c.replace("Type of alternative energy used - ", "").title()
        merged["Towns (sample)"] = merged["Towns (sample)"].apply(
            lambda lst: ", ".join(lst) if isinstance(lst, list) and len(lst) else "—"
        )
        long_frames.append(merged)

    long_df = pd.concat(long_frames, ignore_index=True)
    totals = long_df.groupby(COL_GOV)["Count"].sum().sort_values(ascending=False).index.tolist()

    chart_v2 = (
        alt.Chart(long_df)
        .mark_bar()
        .encode(
            x=alt.X(f"{COL_GOV}:N", sort=totals, title="Governorate"),
            y=alt.Y("Count:Q", stack="normalize", title="Share of towns (%)"),
            color=alt.Color("Energy Type:N", legend=alt.Legend(title="Energy Type")),
            order=alt.Order("Energy Type:N"),
            tooltip=[
                alt.Tooltip(f"{COL_GOV}:N", title="Governorate"),
                alt.Tooltip("Energy Type:N"),
                alt.Tooltip("Count:Q", title="Towns (count)"),
                alt.Tooltip("Towns (sample):N", title="Sample towns"),
            ],
        )
        .properties(height=430)
        .interactive()
    )
    selection = alt.selection_point(fields=["Energy Type"], bind="legend")
    st.altair_chart(chart_v2.add_params(selection).transform_filter(selection), use_container_width=True)

# ---------------- Visual 4 (Heatmap) ----------------
with tab4:
    st.subheader("Visual 4 — Heatmap: Grid State × Lighting State")
    st.caption("Cells show the number of towns (or share) for each Grid × Lighting combination.")
    grp = df_filt.groupby(["_grid_label", "_light_label"], dropna=False).size().reset_index(name="Count")
    towns = (
        df_filt.groupby(["_grid_label", "_light_label"], dropna=False)["_town_clean"]
        .apply(lambda s: ", ".join(list(s[:15])) if len(s) else "—")
        .reset_index(name="Towns (sample)")
    )
    heat = grp.merge(towns, on=["_grid_label", "_light_label"], how="left")

    if norm_axis == "Grid state":
        totals = heat.groupby("_grid_label")["Count"].transform("sum").replace(0, np.nan)
        heat["Value"] = (heat["Count"] / totals) * 100
        value_title = "Share within grid state (%)"
        color_field = alt.Color("Value:Q", title=value_title, scale=alt.Scale(scheme="blues"))
        tooltip_val = alt.Tooltip("Value:Q", title=value_title, format=".1f")
    elif norm_axis == "Lighting state":
        totals = heat.groupby("_light_label")["Count"].transform("sum").replace(0, np.nan)
        heat["Value"] = (heat["Count"] / totals) * 100
        value_title = "Share within lighting state (%)"
        color_field = alt.Color("Value:Q", title=value_title, scale=alt.Scale(scheme="blues"))
        tooltip_val = alt.Tooltip("Value:Q", title=value_title, format=".1f")
    else:
        heat["Value"] = heat["Count"]
        color_field = alt.Color("Value:Q", title="Towns", scale=alt.Scale(scheme="blues"))
        tooltip_val = alt.Tooltip("Value:Q", title="Towns", format=",.0f")

    x_order = ["Bad", "Acceptable", "Good", "Unknown"]
    y_order = ["Bad", "Acceptable", "Good", "Unknown"]

    chart = (
        alt.Chart(heat)
        .mark_rect()
        .encode(
            x=alt.X("_grid_label:N", sort=x_order, title="Grid state"),
            y=alt.Y("_light_label:N", sort=y_order, title="Lighting state"),
            color=color_field,
            tooltip=[
                alt.Tooltip("_grid_label:N", title="Grid"),
                alt.Tooltip("_light_label:N", title="Lighting"),
                alt.Tooltip("Count:Q", title="Towns", format=",.0f"),
                tooltip_val,
                alt.Tooltip("Towns (sample):N", title="Sample towns"),
            ],
        )
        .properties(height=420)
    )
    labels = (
        alt.Chart(heat)
        .mark_text(fontWeight="bold")
        .encode(
            x=alt.X("_grid_label:N", sort=x_order),
            y=alt.Y("_light_label:N", sort=y_order),
            text=alt.Text("Value:Q", format=".0f"),
        )
    )
    st.altair_chart((chart + labels).interactive(), use_container_width=True)

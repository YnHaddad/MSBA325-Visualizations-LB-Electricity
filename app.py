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
DATA_PATH = Path("325 data.csv")  # change if your CSV has a different name

# ---------------- Helpers ----------------
def load_csv_local(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None

def coerce_bool(s: pd.Series) -> pd.Series:
    if s is None: return pd.Series([False] * 0, dtype=bool)
    if pd.api.types.is_bool_dtype(s): return s.fillna(False)
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().any(): return (s_num.fillna(0) != 0)
    return s.astype(str).str.strip().str.lower().isin({"yes","y","true","t","1"})

def coerce_percent(s: pd.Series) -> pd.Series:
    """
    Accepts 0–1, 0–100, or '37%' strings and returns a numeric 0–100 scale.
    """
    if pd.api.types.is_numeric_dtype(s):
        s = s.astype(float)
        if s.dropna().between(0, 1).mean() > 0.5:
            return s * 100.0
        return s
    s_clean = s.astype(str).str.strip().str.replace('%', '', regex=False)
    s_num = pd.to_numeric(s_clean, errors='coerce')
    if s_num.dropna().between(0, 1).mean() > 0.5:
        return s_num * 100.0
    return s_num

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

# ---------------- Load data ----------------
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
    COL_GOV,
    COL_GRID_GOOD, COL_GRID_OK, COL_GRID_BAD,
    COL_LIGHT_GOOD, COL_LIGHT_OK, COL_LIGHT_BAD
]:
    if req not in df.columns:
        st.error(f"Missing required column: **{req}**")
        st.stop()
if not ENERGY_COLS:
    st.error("No energy-type columns found.")
    st.stop()

# ---------------- Adoption % (detect real column; else fallback) ----------------
name_hits = [c for c in df.columns if ("adopt" in c.lower()) or c.strip().endswith("%")]
range_hits = [c for c in df.columns
              if pd.api.types.is_numeric_dtype(df[c])
              and df[c].dropna().between(0, 100).mean() > 0.95
              and df[c].nunique(dropna=True) >= 3]
candidates = list(dict.fromkeys(name_hits + range_hits))
adopt_choice = st.sidebar.selectbox(
    "Adoption column",
    options=["<Use existence flags (0/100)>"] + candidates,
    index=(0 if not candidates else 1),
    help="Pick your numeric adoption column if you have one."
)

if adopt_choice != "<Use existence flags (0/100)>":
    df["_adoption_pct"] = coerce_percent(df[adopt_choice]).clip(lower=0, upper=100)
else:
    if (COL_EXISTS_YES in df.columns) and (COL_EXISTS_NO in df.columns):
        exists_yes = coerce_bool(df[COL_EXISTS_YES])
        exists_no  = coerce_bool(df[COL_EXISTS_NO])
        df["_adoption_pct"] = np.where(exists_yes, 100.0, np.where(exists_no, 0.0, np.nan))
    else:
        df["_adoption_pct"] = np.nan
        st.warning("No adoption column selected and existence flags missing; adoption filter will be disabled.")

# Grid & lighting labels
df["_grid_label"] = df.apply(
    lambda r: derive_onehot_label(r, {"Good": COL_GRID_GOOD, "Acceptable": COL_GRID_OK, "Bad": COL_GRID_BAD}),
    axis=1
)
df["_light_label"] = df.apply(
    lambda r: derive_onehot_label(r, {"Good": COL_LIGHT_GOOD, "Acceptable": COL_LIGHT_OK, "Bad": COL_LIGHT_BAD}),
    axis=1
)

# ---------------- Sidebar filters (search + adoption range + toggles) ----------------
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
        sel_govs = gov_options

    # Adoption range (based on chosen column / fallback)
    if df["_adoption_pct"].notna().any():
        data_min = float(np.nanmin(df["_adoption_pct"]))
        data_max = float(np.nanmax(df["_adoption_pct"]))
        slider_max = max(100.0, round(data_max + 0.5, 1))
        adopt_range = st.slider(
            "Adoption rate filter (%)",
            min_value=0.0, max_value=slider_max,
            value=(0.0, min(slider_max, round(data_max, 1))),
            step=0.5
        )
    else:
        adopt_range = (0.0, 100.0)
        st.caption("Adoption filter disabled (no adoption data).")

    # Bring back Counts vs Percent for Visual 2
    st.markdown("---")
    st.subheader("📊 Visual 2 Options")
    stack_mode = st.radio("Y-axis mode", ["Counts", "Percent"], horizontal=True)
    town_sample_max = st.slider("Towns listed in tooltip (per segment)", 0, 50, 12)

    st.markdown("---")
    st.subheader("🔥 Visual 4 Options")
    norm_axis = st.radio("Normalize heatmap by", ["None", "Grid state", "Lighting state"], horizontal=True)

# Filter rows
mask_gov = df[COL_GOV].astype(str).isin(sel_govs)
if df["_adoption_pct"].notna().any():
    mask_adopt = df["_adoption_pct"].between(adopt_range[0], adopt_range[1], inclusive="both").fillna(False)
else:
    mask_adopt = True
mask = mask_gov & mask_adopt
df_filt = df.loc[mask].copy()

# Context caption
if df["_adoption_pct"].notna().any():
    st.caption(
        f"Using **{adopt_choice if adopt_choice != '<Use existence flags (0/100)>' else 'existence flags'}**, "
        f"showing adoption **{adopt_range[0]}–{adopt_range[1]}%**."
    )

# Tabs
tab2, tab4 = st.tabs(["📊 Visual 2 — Stacked Bar by Governorate", "🧯 Visual 4 — Heatmap: Grid × Lighting"])

# ---------------- Visual 2 (Counts or Percent) ----------------
with tab2:
    st.subheader("Visual 2 — Alternative Energy by Governorate")

    # Build long-form with per-segment count, mean adoption, and town samples with adoption %
    long_frames = []
    for c in ENERGY_COLS:
        flag = coerce_bool(df_filt[c])

        seg = df_filt.loc[flag, [COL_GOV, "_town_clean", "_adoption_pct"]].copy()
        # Prepare sample of "Town (12.3%)"
        seg["_town_label"] = seg.apply(
            lambda r: f"{r['_town_clean']} ({r['_adoption_pct']:.1f}%)" if pd.notna(r["_adoption_pct"]) else f"{r['_town_clean']}",
            axis=1
        )

        # aggregate
        towns = (
            seg.groupby(COL_GOV, dropna=False)["_town_label"]
               .agg(lambda s: ", ".join(list(s[:town_sample_max])) if len(s) else "—")
               .reset_index()
               .rename(columns={"_town_label": "Towns (sample + adoption)"})
        )
        counts = (
            seg.groupby(COL_GOV, dropna=False)["_town_clean"]
               .size()
               .reset_index(name="Count")
        )
        means = (
            seg.groupby(COL_GOV, dropna=False)["_adoption_pct"]
               .mean()
               .reset_index(name="Mean adoption (%)")
        )

        merged = counts.merge(means, on=COL_GOV, how="left").merge(towns, on=COL_GOV, how="left")
        merged["Energy Type"] = c.replace("Type of alternative energy used - ", "").title()
        long_frames.append(merged)

    long_df = pd.concat(long_frames, ignore_index=True)
    long_df["Mean adoption (%)"] = long_df["Mean adoption (%)"].round(1)

    # Order x by total Count descending (Counts mode) or by total share (Percent is normalized anyway)
    totals = long_df.groupby(COL_GOV)["Count"].sum().sort_values(ascending=False).index.tolist()

    if stack_mode == "Percent":
        y_enc = alt.Y("Count:Q", stack="normalize", title="Share of towns (%)")
    else:
        y_enc = alt.Y("Count:Q", stack=None, title="Number of towns")

    chart_v2 = (
        alt.Chart(long_df)
        .mark_bar()
        .encode(
            x=alt.X(f"{COL_GOV}:N", sort=totals, title="Governorate"),
            y=y_enc,
            color=alt.Color("Energy Type:N", legend=alt.Legend(title="Energy Type")),
            order=alt.Order("Energy Type:N"),
            tooltip=[
                alt.Tooltip(f"{COL_GOV}:N", title="Governorate"),
                alt.Tooltip("Energy Type:N"),
                alt.Tooltip("Count:Q"),
                alt.Tooltip("Mean adoption (%):Q", title="Mean adoption (%)", format=".1f"),
                alt.Tooltip("Towns (sample + adoption):N", title="Sample towns"),
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
        totals_h = heat.groupby("_grid_label")["Count"].transform("sum").replace(0, np.nan)
        heat["Value"] = (heat["Count"] / totals_h) * 100
        value_title = "Share within grid state (%)"
        color_field = alt.Color("Value:Q", title=value_title, scale=alt.Scale(scheme="blues"))
        tooltip_val = alt.Tooltip("Value:Q", title=value_title, format=".1f")
    elif norm_axis == "Lighting state":
        totals_h = heat.groupby("_light_label")["Count"].transform("sum").replace(0, np.nan)
        heat["Value"] = (heat["Count"] / totals_h) * 100
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

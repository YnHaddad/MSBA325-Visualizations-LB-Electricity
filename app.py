# app.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path

st.set_page_config(page_title="Lebanon Energy — Visual 2 & 3", page_icon="⚡", layout="wide")
st.title("⚡ Lebanon Energy — Interactive Visuals (Visual 2 & Visual 3)")

# ---------------- Helpers ----------------
@st.cache_data
def load_csv_auto(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    p = Path("325 data.csv")
    if p.exists():
        return pd.read_csv(p)
    return None

def coerce_bool(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series([False] * 0, dtype=bool)
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().any():
        return (s_num.fillna(0) != 0)
    s_str = s.astype(str).str.strip().str.lower()
    return s_str.isin({"yes", "y", "true", "t", "1"})

def clean_town_name(series: pd.Series) -> pd.Series:
    """
    If 'Town' sometimes contains a URL or a path-like string, return only the tail (after the last '/').
    Also strips query params/fragments if present.
    """
    s = series.astype(str)
    # take everything after last '/'
    tail = s.str.extract(r'([^/]+)$')[0].fillna(s)
    # drop URL noise like ?x=y or #anchor
    tail = tail.str.replace(r'[\?#].*$', '', regex=True)
    # trim underscores/dashes and spaces
    tail = tail.str.replace(r'[_\-]+', ' ', regex=True).str.strip()
    return tail

def derive_onehot_label(row: pd.Series, mapping: dict, default="Unknown"):
    for label, col in mapping.items():
        if col and col in row and pd.notna(row[col]) and coerce_bool(pd.Series([row[col]])).iloc[0]:
            return label
    return default

# ---------------- Load data ----------------
with st.sidebar:
    st.header("📁 Data")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], help="If omitted, the app will try to read '325 data.csv' next to this file.")

df = load_csv_auto(uploaded)
if df is None:
    st.info("⬆️ Upload your CSV (or place a file named **325 data.csv** next to app.py).")
    st.stop()

# Column names tailored to your provided CSV
COL_GOV   = "refArea_clean" if "refArea_clean" in df.columns else ("refArea" if "refArea" in df.columns else df.columns[0])
COL_TOWN  = "Town" if "Town" in df.columns else df.columns[0]
COL_EXISTS_YES = "Existence of alternative energy - exists"
COL_EXISTS_NO  = "Existence of alternative energy - does not exist"
COL_GRID_GOOD  = "State of the power grid - good"
COL_GRID_OK    = "State of the power grid - acceptable"
COL_GRID_BAD   = "State of the power grid - bad"

ENERGY_COLS = [c for c in [
    "Type of alternative energy used - solar energy",
    "Type of alternative energy used - wind energy",
    "Type of alternative energy used - hydropower (water use)",
    "Type of alternative energy used - other",
] if c in df.columns]

# Guardrails
for required in [COL_GOV, COL_TOWN, COL_EXISTS_YES, COL_EXISTS_NO, COL_GRID_GOOD, COL_GRID_OK, COL_GRID_BAD]:
    if required not in df.columns:
        st.error(f"Missing required column: **{required}**")
        st.stop()
if not ENERGY_COLS:
    st.error("No energy-type columns found. Expected any of Solar/Wind/Hydropower/Other columns from your CSV.")
    st.stop()

# ---------------- Derivations ----------------
df = df.copy()

# Adoption %: 100 if exists, 0 if explicitly does-not-exist, else NaN
exists_yes = coerce_bool(df[COL_EXISTS_YES])
exists_no  = coerce_bool(df[COL_EXISTS_NO])
df["_adoption_pct"] = np.where(exists_yes, 100.0, np.where(exists_no, 0.0, np.nan))

# Grid label from one-hot flags
grid_map = {"Good": COL_GRID_GOOD, "Acceptable": COL_GRID_OK, "Bad": COL_GRID_BAD}
df["_grid_label"] = df.apply(lambda r: derive_onehot_label(r, grid_map, default="Unknown"), axis=1)

# Cleaned town name for tooltips
df["_town_clean"] = clean_town_name(df[COL_TOWN])

# ---------------- Filters & options ----------------
govs = sorted(df[COL_GOV].dropna().astype(str).unique().tolist())
with st.sidebar:
    st.header("🔎 Filters")
    sel_govs = st.multiselect("Filter by governorate", options=govs, default=govs)
    min_adopt = st.slider("Minimum adoption rate (%)", 0, 100, 0, 1)
    st.markdown("---")
    st.subheader("📊 Visual 2 Options")
    stack_mode = st.radio("Stack mode", ["Counts", "Percent"], horizontal=True)
    town_sample_max = st.slider("Towns listed in tooltip (per segment)", 0, 20, 8)

mask = df[COL_GOV].astype(str).isin(sel_govs) & (df["_adoption_pct"].fillna(-1) >= min_adopt)
df_filt = df.loc[mask].copy()

tab2, tab3 = st.tabs(["📊 Visual 2 — Stacked Bar by Governorate", "📈 Visual 3 — Adoption vs Grid"])

# ---------------- Visual 2 ----------------
with tab2:
    st.subheader("Visual 2 — Alternative Energy by Governorate (Stacked)")
    # Build long-form table with a *sample of cleaned town names* per segment for tooltips
    long_frames = []
    for c in ENERGY_COLS:
        flag = coerce_bool(df_filt[c])
        grp = (
            df_filt.loc[flag, [COL_GOV, "_town_clean"]]
            .groupby(COL_GOV, dropna=False)["_town_clean"]
            .agg(lambda s: list(s[:town_sample_max]))  # sample first N towns
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
        merged = counts.merge(grp, on=COL_GOV, how="left")
        merged["Energy Type"] = c.replace("Type of alternative energy used - ", "").title()
        # stringify towns for Altair tooltip
        merged["Towns (sample)"] = merged["Towns (sample)"].apply(
            lambda lst: ", ".join(lst) if isinstance(lst, list) and len(lst) else "—"
        )
        long_frames.append(merged)

    long_df = pd.concat(long_frames, ignore_index=True)

    # Sort governorates by total count desc
    totals = long_df.groupby(COL_GOV)["Count"].sum().sort_values(ascending=False).index.tolist()

    y_field = alt.Y(
        "Count:Q",
        stack=("normalize" if stack_mode == "Percent" else None),
        title=("Share of towns (%)" if stack_mode == "Percent" else "Number of towns")
    )

    chart_v2 = (
        alt.Chart(long_df)
        .mark_bar()
        .encode(
            x=alt.X(f"{COL_GOV}:N", sort=totals, title="Governorate"),
            y=y_field,
            color=alt.Color("Energy Type:N", legend=alt.Legend(title="Energy Type")),
            order=alt.Order("Energy Type:N"),
            tooltip=[
                alt.Tooltip(f"{COL_GOV}:N", title="Governorate"),
                alt.Tooltip("Energy Type:N"),
                alt.Tooltip("Count:Q"),
                alt.Tooltip("Towns (sample):N", title="Towns"),
            ],
        )
        .properties(height=430)
        .interactive()
    )

    # Legend click to isolate
    selection = alt.selection_point(fields=["Energy Type"], bind="legend")
    chart_v2 = chart_v2.add_params(selection).transform_filter(selection)
    st.altair_chart(chart_v2, use_container_width=True)

# ---------------- Visual 3 ----------------
with tab3:
    st.subheader("Visual 3 — Connectivity vs Alternative Energy Adoption")
    st.caption("Adoption is derived from existence flags: 100% if exists, 0% if does not exist.")
    if df_filt.empty:
        st.warning("No data after filters. Try including more governorates or lowering the adoption threshold.")
    else:
        color_by = st.selectbox("Color by", options=["None", "Governorate"], index=1)
        enc = {
            "x": alt.X("_grid_label:N", title="Grid state"),
            "y": alt.Y("_adoption_pct:Q", title="Adoption rate (%)"),
            "tooltip": [
                alt.Tooltip("_town_clean:N", title="Town"),
                alt.Tooltip(COL_GOV + ":N", title="Governorate"),
                alt.Tooltip("_grid_label:N", title="Grid state"),
                alt.Tooltip("_adoption_pct:Q", title="Adoption (%)", format=".0f"),
            ],
        }
        if color_by == "Governorate":
            enc["color"] = alt.Color(COL_GOV + ":N", title="Governorate")

        points = alt.Chart(df_filt).mark_circle(size=80, opacity=0.7).encode(**enc)

        # Boxplot overlay to summarize distributions per grid state
        box = (
            alt.Chart(df_filt)
            .mark_boxplot(opacity=0.3)
            .encode(
                x=alt.X("_grid_label:N", title="Grid state"),
                y=alt.Y("_adoption_pct:Q", title="Adoption rate (%)"),
            )
        )

        st.altair_chart((points + box).properties(height=500).interactive(), use_container_width=True)

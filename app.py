# app.py
# Streamlit app showing ONLY:
# 2) (Interactive) Lighting network state by governorate
# 4) (Interactive) Town-level alternative energy profile + compare
#
# It also cleans Town names that come as links (URIs) into readable names.

import os
import re
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

# ----------------------------
# Config
# ----------------------------
st.set_page_config(
    page_title="Municipal Energy Explorer — Visuals 2 & 4",
    layout="wide",
    page_icon="🔌",
)

DATA_PATH_GUESS = "325 data.csv"

# ----------------------------
# Helpers
# ----------------------------
def extract_name_from_link(val: str) -> str:
    """
    If the value looks like a URL/DBpedia URI, extract the last path segment,
    replace underscores with spaces, strip trailing punctuation/fragments.
    Otherwise, return the original string, trimmed.
    """
    if pd.isna(val):
        return val
    s = str(val).strip()
    if s.startswith(("http://", "https://")):
        # Take last path segment
        m = re.search(r"/([^/?#]+)", s[::-1])  # reverse trick to catch from end
        if m:
            seg = m.group(1)[::-1]  # reverse back
        else:
            # Fallback: last chunk after last slash
            seg = s.rsplit("/", 1)[-1]
        seg = seg.replace("_", " ")
        # Remove common DBpedia suffixes like "(district)" if present at end of segment
        seg = re.sub(r"\s*\((?:[^)]*)\)\s*$", "", seg).strip()
        return seg
    # Handle dbpedia-style without protocol
    if "dbpedia.org" in s:
        seg = s.rsplit("/", 1)[-1].replace("_", " ")
        seg = re.sub(r"\s*\((?:[^)]*)\)\s*$", "", seg).strip()
        return seg
    return s

@st.cache_data(show_spinner=False)
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        if not os.path.exists(DATA_PATH_GUESS):
            st.error("CSV not found. Upload it from the sidebar or place '325 data.csv' next to app.py.")
            st.stop()
        df = pd.read_csv(DATA_PATH_GUESS)

    # Tidy columns
    df.columns = [c.strip() for c in df.columns]

    # Clean Town names from links → readable names
    town_col_guess = None
    for cand in ["Town", "town", "Municipality", "municipality"]:
        if cand in df.columns:
            town_col_guess = cand
            break
    if town_col_guess is None:
        st.error("Could not find a 'Town' column. Please ensure your CSV has a Town/municipality column.")
        st.stop()

    df[town_col_guess] = df[town_col_guess].apply(extract_name_from_link)

    # Extract Governorate from refArea URI if present
    if "refArea" in df.columns:
        def extract_gov(x: str):
            if pd.isna(x):
                return np.nan
            m = re.search(r'/([^/]+)$', str(x))
            if m:
                name = m.group(1).replace("_", " ")
                name = name.replace("Governorate", "").strip()
                name = re.sub(r"\s*\((?:[^)]*)\)\s*$", "", name).strip()
                return name
            return np.nan
        df["Governorate"] = df["refArea"].apply(extract_gov)
    elif "Governorate" not in df.columns:
        df["Governorate"] = np.nan  # keep column for grouping

    # Canonical column keys (update these if your header names differ)
    COLS = {
        "town": town_col_guess,
        "light_good": "State of the lighting network - good",
        "light_ok": "State of the lighting network - acceptable",
        "light_bad": "State of the lighting network - bad",
        "alt_exists": "Existence of alternative energy - exists",
        "alt_not": "Existence of alternative energy - does not exist",
        "solar": "Type of alternative energy used - solar energy",
        "wind": "Type of alternative energy used - wind energy",
        "hydro": "Type of alternative energy used - hydropower (water use)",
        "other": "Type of alternative energy used - other",
    }

    # Ensure missing columns exist as zeros (prevents crashes if a column is absent)
    needed = [
        COLS["light_good"], COLS["light_ok"], COLS["light_bad"],
        COLS["alt_exists"], COLS["alt_not"],
        COLS["solar"], COLS["wind"], COLS["hydro"], COLS["other"],
    ]
    for c in needed:
        if c not in df.columns:
            df[c] = 0

    # Force binary ints
    for c in needed:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    return df, COLS

# ----------------------------
# Sidebar: data source
# ----------------------------
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
df, COLS = load_data(uploaded)

st.title("🔌 Municipal Energy Explorer")
st.caption("Showing only Visual 2 & Visual 4, with town names cleaned from links.")

# ======================================================
# VISUAL 2 — Lighting Network State by Governorate (Interactive)
# ======================================================
st.markdown("### 2) Lighting Network State — by Governorate")

left, right = st.columns([2, 1])
with left:
    govs_all = ["(All)"] + sorted([g for g in df["Governorate"].dropna().unique().tolist()])
    chosen_gov = st.selectbox("Filter by Governorate", options=govs_all, index=0)
with right:
    normalize = st.checkbox("Show as % of towns", value=True)

light_cols = [COLS["light_good"], COLS["light_ok"], COLS["light_bad"]]
renamer = {
    COLS["light_good"]: "Good",
    COLS["light_ok"]: "Acceptable",
    COLS["light_bad"]: "Bad",
}

df2 = df.copy()
if chosen_gov != "(All)":
    df2 = df2[df2["Governorate"] == chosen_gov]

if chosen_gov == "(All)":
    grp = (
        df2.groupby("Governorate", dropna=False)[light_cols]
        .sum()
        .reset_index()
        .rename(columns=renamer)
    )
    tall = grp.melt(id_vars=["Governorate"], var_name="State", value_name="count")
    if normalize:
        totals = tall.groupby("Governorate")["count"].transform("sum")
        tall["pct"] = np.where(totals > 0, tall["count"] / totals, 0.0)
        y = alt.Y("pct:Q", axis=alt.Axis(format="%"), title="Share of towns")
    else:
        y = alt.Y("count:Q", title="Number of towns")

    chart2 = (
        alt.Chart(tall)
        .mark_bar()
        .encode(
            x=alt.X("Governorate:N", sort="-y"),
            y=y,
            color=alt.Color("State:N"),
            tooltip=["Governorate:N", "State:N", "count:Q"] + (["pct:Q"] if normalize else []),
        )
        .properties(height=420)
    )
else:
    counts = {
        "Good": int(df2[COLS["light_good"]].sum()),
        "Acceptable": int(df2[COLS["light_ok"]].sum()),
        "Bad": int(df2[COLS["light_bad"]].sum()),
    }
    tall = pd.DataFrame({"State": list(counts.keys()), "count": list(counts.values())})
    if normalize:
        total = tall["count"].sum()
        tall["pct"] = (tall["count"] / total) if total else 0.0
        y = alt.Y("pct:Q", axis=alt.Axis(format="%"), title="Share of towns")
    else:
        y = alt.Y("count:Q", title="Number of towns")

    chart2 = (
        alt.Chart(tall)
        .mark_bar()
        .encode(
            x=alt.X("State:N", sort="-y"),
            y=y,
            color=alt.Color("State:N"),
            tooltip=["State:N", "count:Q"] + (["pct:Q"] if normalize else []),
        )
        .properties(height=360)
    )

st.altair_chart(chart2, use_container_width=True)
st.divider()

# ======================================================
# VISUAL 4 — Town-level Alternative Energy Profile (Interactive)
# ======================================================
st.markdown("### 4) Town Alternative Energy Profile")

energy_map = {
    "Solar": COLS["solar"],
    "Wind": COLS["wind"],
    "Hydro": COLS["hydro"],
    "Other": COLS["other"],
}

towns = sorted(df[COLS["town"]].dropna().unique().tolist())

c1, c2, c3 = st.columns([2, 2, 2])
with c1:
    selected_town = st.selectbox("Choose a town", options=towns)
with c2:
    compare_town = st.selectbox("Compare with (optional)", options=["(None)"] + towns, index=0)
with c3:
    types_chosen = st.multiselect(
        "Energy types to display",
        options=list(energy_map.keys()),
        default=list(energy_map.keys()),
    )

def build_town_profile(frame, town_name, label):
    row = frame[frame[COLS["town"]] == town_name].head(1)
    if row.empty:
        return pd.DataFrame(columns=["Town", "Type", "Present"])
    records = []
    for t_label, colname in energy_map.items():
        present = int(row[colname].iloc[0])
        records.append({"Town": label, "Type": t_label, "Present": present})
    return pd.DataFrame(records)

profile_main = build_town_profile(df, selected_town, selected_town)
if compare_town != "(None)":
    profile_cmp = build_town_profile(df, compare_town, compare_town)
    prof_all = pd.concat([profile_main, profile_cmp], ignore_index=True)
else:
    prof_all = profile_main.copy()

# Filter by energy types chosen
if types_chosen:
    prof_all = prof_all[prof_all["Type"].isin(types_chosen)]
else:
    st.info("Select at least one energy type to display.")
    prof_all = prof_all.iloc[0:0]

# Clustered bars (column facet per Town)
chart4 = (
    alt.Chart(prof_all)
    .mark_bar()
    .encode(
        x=alt.X("Type:N", title="Energy Type"),
        y=alt.Y("Present:Q", title="Presence (0/1)", scale=alt.Scale(domain=[0, 1])),
        column=alt.Column("Town:N", title=None),
        color=alt.Color("Type:N", legend=None),
        tooltip=["Town:N", "Type:N", "Present:Q"],
    )
    .properties(height=300)
)

st.altair_chart(chart4, use_container_width=True)

with st.expander("Summary"):
    if not prof_all.empty:
        by_town = (
            prof_all.groupby(["Town"])["Present"]
            .sum()
            .reset_index()
            .rename(columns={"Present": "Number of energy types present"})
        )
        st.dataframe(by_town, use_container_width=True)
    else:
        st.write("No types selected.")

# ----------------------------
# Footer
# ----------------------------
st.caption("Upload an updated CSV anytime from the sidebar. Town names are automatically cleaned from links to readable names.")

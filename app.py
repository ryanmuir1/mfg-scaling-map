
import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# -----------------------------------------------
# Config
# -----------------------------------------------
st.set_page_config(page_title="Scaling Surface Map", layout="wide")
st.title("Scaling Surface Map — Scale-Agnostic Manufacturing")

# DEFAULTS: adjust these to match your repo
DEFAULT_DATA_PATH = os.getenv("SCALING_DATA_PATH", "data/Scaling_Surface_Template.xlsx")
DEFAULT_DATA_URL  = os.getenv("SCALING_DATA_URL", "")  # optional raw GitHub URL

with st.expander("About this app"):
    st.markdown(
        """
        This app estimates how each process scales with lot size and visualises a **Scaling Surface Map**.
        By default, it loads your **repo-controlled template**. You can override by uploading a file.
        """
    )

# -----------------------------------------------
# Data Loaders
# -----------------------------------------------
@st.cache_data(show_spinner=False)
def load_from_path(path_str: str) -> pd.DataFrame:
    path_str = path_str.strip()
    if not path_str:
        raise FileNotFoundError("Empty path")
    if not os.path.exists(path_str):
        raise FileNotFoundError(f"Path not found: {path_str}")
    if path_str.lower().endswith(".csv"):
        df = pd.read_csv(path_str)
    else:
        xls = pd.ExcelFile(path_str)
        sheet = "ScalingData" if "ScalingData" in xls.sheet_names else xls.sheet_names[0]
        df = pd.read_excel(xls, sheet_name=sheet)
    return _clean(df)

@st.cache_data(show_spinner=False)
def load_from_url(url: str) -> pd.DataFrame:
    import urllib.request
    if not url:
        raise ValueError("Empty URL")
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    # Try Excel first, fallback CSV
    try:
        df = pd.read_excel(io.BytesIO(data), sheet_name="ScalingData")
    except Exception:
        try:
            df = pd.read_csv(io.BytesIO(data))
        except Exception as e:
            raise ValueError(f"Failed to parse content from URL: {e}")
    return _clean(df)

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    # Basic cleanup / typing
    text_cols = [
        "Process_ID","Process_Name","Process_Area","Metric_Type","Observation_Date","Lot_ID",
        "Owner","Subprocess","Current_State","Trigger_Type","Actor","Data_Coupling","State_Coupling",
        "Regulatory_Constraint","Target_State","Target_Intervention"
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    if "Lot_Size_Units" in df.columns:
        df["Lot_Size_Units"] = pd.to_numeric(df["Lot_Size_Units"], errors="coerce")
    if "Metric_Value" in df.columns:
        df["Metric_Value"] = pd.to_numeric(df["Metric_Value"], errors="coerce")
    return df

def compute_scaling(df: pd.DataFrame, metric_type: str) -> pd.DataFrame:
    dfm = df[df["Metric_Type"] == metric_type].copy()
    dfm = dfm.dropna(subset=["Process_ID","Lot_Size_Units","Metric_Value"])

    results = []
    for pid, g in dfm.groupby("Process_ID"):
        if g["Lot_Size_Units"].nunique() < 2:
            continue
        x = g["Lot_Size_Units"].values.astype(float)
        y = g["Metric_Value"].values.astype(float)

        A = np.vstack([x, np.ones_like(x)]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        yhat = intercept + slope * x
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2) if len(y) > 1 else 0.0
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

        meta_cols = [c for c in g.columns if c not in ["Lot_Size_Units","Metric_Value","Observation_Date","Lot_ID","Notes"]]
        meta = {col: g[col].iloc[0] for col in meta_cols}
        results.append({
            "Process_ID": pid,
            "Process_Name": meta.get("Process_Name", pid),
            "Process_Area": meta.get("Process_Area", ""),
            "Current_State": meta.get("Current_State", ""),
            "Trigger_Type": meta.get("Trigger_Type", ""),
            "Actor": meta.get("Actor", ""),
            "Data_Coupling": meta.get("Data_Coupling", ""),
            "State_Coupling": meta.get("State_Coupling", ""),
            "Automation_Readiness": pd.to_numeric(meta.get("Automation_Readiness", np.nan), errors="coerce"),
            "Regulatory_Constraint": meta.get("Regulatory_Constraint", ""),
            "Target_State": meta.get("Target_State", ""),
            "Target_Intervention": meta.get("Target_Intervention", ""),
            "Slope_per_Unit": float(slope),
            "Intercept": float(intercept),
            "R2": float(r2),
            "N_Obs": int(len(g))
        })
    res = pd.DataFrame(results)
    if res.empty:
        return res

    # Normalise slope into [0,1] via robust percentiles
    sl = res["Slope_per_Unit"].values
    p5, p95 = np.nanpercentile(sl, 5), np.nanpercentile(sl, 95)
    rng = (p95 - p5) if (p95 - p5) > 1e-9 else 1.0
    res["Scaling_Score_0to1"] = np.clip((sl - p5) / rng, 0, 1)

    bins = [0, 0.25, 0.5, 0.75, 1.01]
    labels = ["Agnostic","Event-leaning","Event-driven","Linear-ish"]
    res["Scaling_Class"] = pd.cut(res["Scaling_Score_0to1"], bins=bins, labels=labels, include_lowest=True)
    return res

def surface_grid(df_metrics: pd.DataFrame, df_raw: pd.DataFrame, metric_type: str):
    dfm = df_raw[df_raw["Metric_Type"] == metric_type].copy()
    dfm = dfm.dropna(subset=["Process_ID","Lot_Size_Units","Metric_Value"])
    processes = df_metrics["Process_ID"].tolist()
    lot_sizes = np.sort(dfm["Lot_Size_Units"].unique())

    Z = np.full((len(lot_sizes), len(processes)), np.nan)
    for i, ls in enumerate(lot_sizes):
        for j, pid in enumerate(processes):
            vals = dfm[(dfm["Process_ID"] == pid) & (dfm["Lot_Size_Units"] == ls)]["Metric_Value"].values
            if len(vals) > 0:
                Z[i, j] = float(np.mean(vals))
    return processes, lot_sizes, Z

# -----------------------------------------------
# Data Source Controls
# -----------------------------------------------
st.sidebar.header("Data Source")
source = st.sidebar.radio("Choose data source", ["Repo default", "Upload file"], index=0)

repo_path = st.sidebar.text_input("Repo path (relative or absolute)", value=DEFAULT_DATA_PATH)
repo_url  = st.sidebar.text_input("Optional raw GitHub URL", value=DEFAULT_DATA_URL)

uploaded = st.file_uploader("Upload a new dataset (.xlsx or .csv)", type=["xlsx","csv"]) if source == "Upload file" else None

# Load data
df = None
load_errors = []

if source == "Repo default":
    try:
        df = load_from_path(repo_path)
        st.success(f"Loaded repo data from: {repo_path}  (rows={len(df)})")
    except Exception as e:
        load_errors.append(str(e))
        if repo_url:
            try:
                df = load_from_url(repo_url)
                st.success(f"Loaded repo data from URL (rows={len(df)}).")
            except Exception as e2:
                load_errors.append(f"URL load failed: {e2}")
        else:
            st.info("No valid repo file found. Provide a URL in the sidebar or switch to 'Upload file'.")

elif uploaded is not None:
    try:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            xls = pd.ExcelFile(uploaded)
            sheet = "ScalingData" if "ScalingData" in xls.sheet_names else xls.sheet_names[0]
            df = pd.read_excel(xls, sheet_name=sheet)
        df = _clean(df)
        st.success(f"Loaded uploaded data (rows={len(df)}).")
    except Exception as e:
        load_errors.append(f"Upload load failed: {e}")

if load_errors and df is None:
    st.error(" | ".join(load_errors))

# -----------------------------------------------
# Main Analysis
# -----------------------------------------------
if df is not None and not df.empty:
    metric = st.selectbox("Choose Metric_Type to analyze", sorted(df["Metric_Type"].dropna().unique().tolist()))
    area_filter = st.multiselect("Filter by Process_Area (optional)", sorted(df["Process_Area"].dropna().unique().tolist()))
    df_f = df[df["Process_Area"].isin(area_filter)] if area_filter else df.copy()

    res = compute_scaling(df_f, metric)
    if res.empty:
        st.warning("Not enough data to compute scaling. Each Process_ID needs ≥ 2 distinct Lot_Size_Units.")
        st.stop()

    st.subheader("Per-Process Scaling Summary")
    st.dataframe(res.sort_values("Scaling_Score_0to1", ascending=False))

    st.subheader("Scaling Heatmap (0 = agnostic → 1 = linear-ish)")
    fig_hm, ax_hm = plt.subplots(figsize=(10, 1))
    hm = ax_hm.imshow(res[["Scaling_Score_0to1"]].T, aspect="auto")
    ax_hm.set_yticks([0]); ax_hm.set_yticklabels(["Scaling Score"])
    ax_hm.set_xticks(range(len(res))); ax_hm.set_xticklabels(res["Process_Name"], rotation=45, ha="right")
    ax_hm.set_title("Process Scaling Heatmap")
    plt.colorbar(hm, ax=ax_hm)
    st.pyplot(fig_hm)

    st.subheader("3D Surface — Relative Metric across Lot Sizes")
    procs, lot_sizes, Z = surface_grid(res, df_f, metric)
    if len(procs) > 0 and len(lot_sizes) > 0:
        X, Y = np.meshgrid(np.arange(len(procs)), np.arange(len(lot_sizes)))
        fig3d = plt.figure(figsize=(10, 6))
        ax3d = fig3d.add_subplot(111, projection="3d")
        surf = ax3d.plot_surface(X, Y, np.nan_to_num(Z, nan=np.nanmean(Z)), edgecolor="k")
        ax3d.set_xticks(np.arange(len(procs))); ax3d.set_xticklabels(procs, rotation=45, ha="right")
        ax3d.set_yticks(np.arange(len(lot_sizes))); ax3d.set_yticklabels(lot_sizes)
        ax3d.set_xlabel("Process_ID"); ax3d.set_ylabel("Lot Size"); ax3d.set_zlabel(metric)
        ax3d.set_title("Scaling Surface Map")
        st.pyplot(fig3d)
    else:
        st.info("Surface requires multiple lot sizes across multiple processes. Add more observations.")

    # Automated Insights
    st.subheader("Automated Insights")
    reg_weight = res["Regulatory_Constraint"].map({"Low":1.0,"Medium":1.3,"High":1.7}).fillna(1.2)
    readiness = pd.to_numeric(res["Automation_Readiness"], errors="coerce").fillna(3.0)
    res["Priority_Score"] = res["Scaling_Score_0to1"] * readiness / reg_weight

    worst = res.sort_values("Scaling_Score_0to1", ascending=False).head(5)
    quick = res.sort_values("Priority_Score", ascending=False).head(5)

    st.markdown("**Top 5 Scaling Risks (steepest slopes):**")
    st.dataframe(worst[["Process_Name","Process_Area","Current_State","Scaling_Score_0to1","R2","N_Obs"]])

    st.markdown("**Top 5 Quick Wins (impact × readiness ÷ constraint):**")
    st.dataframe(quick[[
        "Process_Name","Process_Area","Current_State","Automation_Readiness",
        "Regulatory_Constraint","Scaling_Score_0to1","Priority_Score",
        "Target_State","Target_Intervention"
    ]])

    # Area summary
    area_agg = res.groupby("Process_Area")["Scaling_Score_0to1"].mean().sort_values(ascending=False)
    st.markdown("**Avg Scaling Score by Area (higher = worse):**")
    fig_bar, ax_bar = plt.subplots()
    ax_bar.bar(area_agg.index, area_agg.values)
    ax_bar.set_ylabel("Average Scaling Score (0–1)")
    ax_bar.set_title("Area Contribution to Scaling Problem")
    st.pyplot(fig_bar)

    # Executive summary
    st.subheader("Auto-Generated Summary")
    n_proc = len(res)
    top_proc = worst.iloc[0]["Process_Name"] if len(worst) else "N/A"
    top_area = area_agg.index[0] if len(area_agg) else "N/A"
    st.markdown(dedent(f"""
    - Analyzed **{n_proc}** processes for metric **{metric}**.
    - Worst scaling area: **{top_area}**.
    - Highest-risk process: **{top_proc}**.
    - Focus next: **Top 5 Quick Wins** above.
    """))
else:
    st.info("Provide a valid repo dataset (sidebar) or switch to 'Upload file'.")

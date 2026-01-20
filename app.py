import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -------------------- Page Setup --------------------
st.set_page_config(page_title="UIDAI Dashboard", layout="wide")
st.title("UIDAI Aadhaar Enrolment & Updates Analysis Dashboard")

# -------------------- Load Data --------------------
@st.cache_data
def load_data():
    df = pd.read_csv("merged_df.csv")
    return df

df = load_data()

# -------------------- Feature Engineering --------------------
df["demo_total"] = df["demo_age_5_17"] + df["demo_age_17_"]
df["bio_total"]  = df["bio_age_5_17"] + df["bio_age_17_"]
df["enro_total"] = df["enro_age_0_5"] + df["enro_age_5_17"] + df["enro_age_18_greater"]

df["updates_total"] = df["demo_total"] + df["bio_total"]
df["total_activity"] = df["updates_total"] + df["enro_total"]

# Stress Index
df["stress_index"] = (2 * df["bio_total"]) + (1 * df["demo_total"]) + (0.5 * df["enro_total"])

# Bio Share
df["bio_share"] = df["bio_total"] / (df["bio_total"] + df["demo_total"] + 1)

# -------------------- Sidebar Filters --------------------
st.sidebar.header("Filters")

state_list = sorted(df["state"].dropna().unique())
selected_state = st.sidebar.selectbox("State", ["All"] + state_list)

df_filtered = df.copy()
if selected_state != "All":
    df_filtered = df_filtered[df_filtered["state"] == selected_state]

district_list = sorted(df_filtered["district"].dropna().unique())
selected_district = st.sidebar.selectbox("District", ["All"] + district_list)

if selected_district != "All":
    df_filtered = df_filtered[df_filtered["district"] == selected_district]

month_list = sorted(df_filtered["month"].dropna().unique())
selected_month = st.sidebar.selectbox("Month", ["All"] + month_list)

if selected_month != "All":
    df_filtered = df_filtered[df_filtered["month"] == selected_month]

weekday_list = sorted(df_filtered["weekday"].dropna().unique())
selected_weekday = st.sidebar.selectbox("Weekday", ["All"] + weekday_list)

if selected_weekday != "All":
    df_filtered = df_filtered[df_filtered["weekday"] == selected_weekday]

st.sidebar.write("Filtered Rows:", df_filtered.shape[0])

# Download filtered data
st.sidebar.download_button(
    "Download Filtered CSV",
    df_filtered.to_csv(index=False).encode("utf-8"),
    "uidai_filtered.csv",
    "text/csv"
)

# -------------------- Tabs --------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Dashboard",
    "Stress Index",
    "Mature Regions",
    "Anomalies",
    "80/20 Concentration",
    "Bio Dominance",
    "Report Summary"
])

# =========================================================
# TAB 1: DASHBOARD
# =========================================================
with tab1:
    st.subheader("Key Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Enrolments", int(df_filtered["enro_total"].sum()))
    c2.metric("Demographic Updates", int(df_filtered["demo_total"].sum()))
    c3.metric("Biometric Updates", int(df_filtered["bio_total"].sum()))
    c4.metric("Total Activity", int(df_filtered["total_activity"].sum()))

    st.markdown("---")

    colA, colB = st.columns(2)

    with colA:
        st.subheader("Monthly Trend (Demo vs Bio vs Enrol)")
        monthly = df_filtered.groupby("month")[["demo_total","bio_total","enro_total"]].sum()

        fig, ax = plt.subplots(figsize=(7,4))
        monthly.plot(ax=ax, marker="o")
        ax.set_xlabel("Month")
        ax.set_ylabel("Count")
        ax.grid(True)
        st.pyplot(fig)

    with colB:
        st.subheader("Weekday Distribution (Total Activity)")
        weekday_sum = df_filtered.groupby("weekday")["total_activity"].sum().sort_values(ascending=False)

        fig2, ax2 = plt.subplots(figsize=(7,4))
        weekday_sum.plot(kind="bar", ax=ax2)
        ax2.set_xlabel("Weekday")
        ax2.set_ylabel("Total Activity")
        ax2.grid(axis="y")
        st.pyplot(fig2)

    st.markdown("---")

    st.subheader("Top 10 Districts by Total Activity")
    top_dist = df_filtered.groupby(["state","district"])["total_activity"].sum().sort_values(ascending=False).head(10)

    fig3, ax3 = plt.subplots(figsize=(12,4))
    top_dist.plot(kind="bar", ax=ax3)
    ax3.set_xlabel("State, District")
    ax3.set_ylabel("Total Activity")
    ax3.grid(axis="y")
    plt.xticks(rotation=45)
    st.pyplot(fig3)

    st.markdown("---")
    st.subheader("Download Top Tables")

    colx, coly = st.columns(2)

    with colx:
        top_pincodes = df_filtered.groupby("pincode")["total_activity"].sum().sort_values(ascending=False).head(20).reset_index()
        st.download_button(
            "Download Top 20 Pincodes (CSV)",
            top_pincodes.to_csv(index=False).encode("utf-8"),
            "top20_pincodes.csv",
            "text/csv"
        )
        st.dataframe(top_pincodes)

    with coly:
        top_districts_table = df_filtered.groupby(["state","district"])["total_activity"].sum().sort_values(ascending=False).head(20).reset_index()
        st.download_button(
            "Download Top 20 Districts (CSV)",
            top_districts_table.to_csv(index=False).encode("utf-8"),
            "top20_districts.csv",
            "text/csv"
        )
        st.dataframe(top_districts_table)

# =========================================================
# TAB 2: STRESS INDEX
# =========================================================
with tab2:
    st.subheader("Top 15 Districts by Stress Index")
    stress_table = df_filtered.groupby(["state","district"])["stress_index"].sum().sort_values(ascending=False).head(15)
    st.dataframe(stress_table)

    fig4, ax4 = plt.subplots(figsize=(12,4))
    stress_table.plot(kind="bar", ax=ax4)
    ax4.set_title("Stress Index (Weighted Workload)")
    ax4.set_xlabel("State, District")
    ax4.set_ylabel("Stress Index")
    ax4.grid(axis="y")
    plt.xticks(rotation=45)
    st.pyplot(fig4)

    stress_export = stress_table.reset_index()
    st.download_button(
        "Download Stress Index Top 15 (CSV)",
        stress_export.to_csv(index=False).encode("utf-8"),
        "stress_index_top15.csv",
        "text/csv"
    )

# =========================================================
# TAB 3: MATURE REGIONS
# =========================================================
with tab3:
    st.subheader("States with High Updates but Low Enrolments")

    state_summary = df_filtered.groupby("state")[["updates_total","enro_total"]].sum()
    state_summary["update_enrol_ratio"] = state_summary["updates_total"] / (state_summary["enro_total"] + 1)

    # safer threshold (or remove filter completely)
    state_summary = state_summary[state_summary["updates_total"] > 1000]

    if state_summary.empty:
        st.warning("No states found for the selected filters. Try removing some filters.")
    else:
        state_summary = state_summary.sort_values("update_enrol_ratio", ascending=False)

        st.dataframe(state_summary.head(15))

        fig5, ax5 = plt.subplots(figsize=(10,4))
        state_summary.head(10)["update_enrol_ratio"].plot(kind="bar", ax=ax5)
        ax5.set_title("Update/Enrolment Ratio (Top States)")
        ax5.set_xlabel("State")
        ax5.set_ylabel("Ratio")
        ax5.grid(axis="y")
        plt.xticks(rotation=45)
        st.pyplot(fig5)


# =========================================================
# TAB 4: ANOMALIES
# =========================================================
with tab4:
    st.subheader("Anomaly Detection using Z-Score (Total Activity Spikes)")

    mean_val = df_filtered["total_activity"].mean()
    std_val = df_filtered["total_activity"].std()

    temp_df = df_filtered.copy()
    temp_df["zscore"] = (temp_df["total_activity"] - mean_val) / (std_val + 1e-9)

    anomalies = temp_df[temp_df["zscore"] > 5].sort_values("zscore", ascending=False)

    st.write("Anomalies found:", anomalies.shape[0])

    anom_table = anomalies[["state","district","pincode","month","day","total_activity","zscore"]].head(30)
    st.dataframe(anom_table)

    anom_export = anomalies[["state","district","pincode","month","day","total_activity","zscore"]].head(100)
    st.download_button(
        "Download Top 100 Anomalies (CSV)",
        anom_export.to_csv(index=False).encode("utf-8"),
        "anomalies_top100.csv",
        "text/csv"
    )

# =========================================================
# TAB 5: 80/20 Concentration
# =========================================================
with tab5:
    st.subheader("80/20 Pincode Concentration Analysis")

    pincode_sum = df_filtered.groupby("pincode")["total_activity"].sum().sort_values(ascending=False)
    cum_share = pincode_sum.cumsum() / (pincode_sum.sum() + 1e-9)

    pincodes_for_80 = (cum_share <= 0.80).sum()

    st.write("Total Pincodes:", pincode_sum.shape[0])
    st.write("Pincodes needed for 80% activity:", pincodes_for_80)

    fig6, ax6 = plt.subplots(figsize=(10,4))
    ax6.plot(cum_share.values)
    ax6.set_title("Cumulative Share of Activity by Pincode")
    ax6.set_xlabel("Pincodes (sorted high → low)")
    ax6.set_ylabel("Cumulative Share")
    ax6.grid(True)
    st.pyplot(fig6)

# =========================================================
# TAB 6: Bio Dominance
# =========================================================
with tab6:
    st.subheader("Biometric Dominance Regions (Bio Share Analysis)")

    bio_district = df_filtered.groupby(["state","district"])[["bio_total","demo_total"]].sum()
    bio_district["bio_share"] = bio_district["bio_total"] / (bio_district["bio_total"] + bio_district["demo_total"] + 1)
    bio_district = bio_district.sort_values("bio_share", ascending=False)

    st.write("Top 20 districts with highest biometric dominance:")
    st.dataframe(bio_district.head(20))

    top20_bio = bio_district.head(20)["bio_share"]

    fig7, ax7 = plt.subplots(figsize=(12,4))
    top20_bio.plot(kind="bar", ax=ax7)
    ax7.set_title("Top 20 Districts by Biometric Share")
    ax7.set_xlabel("State, District")
    ax7.set_ylabel("Bio Share")
    ax7.grid(axis="y")
    plt.xticks(rotation=45)
    st.pyplot(fig7)

    bio_export = bio_district.head(50).reset_index()
    st.download_button(
        "Download Top 50 Bio Dominance Districts (CSV)",
        bio_export.to_csv(index=False).encode("utf-8"),
        "bio_dominance_top50.csv",
        "text/csv"
    )

# =========================================================
# TAB 7: REPORT SUMMARY GENERATOR
# =========================================================
with tab7:
    st.subheader("Auto Report Summary (Based on Current Filters)")

    total_enro = int(df_filtered["enro_total"].sum())
    total_demo = int(df_filtered["demo_total"].sum())
    total_bio  = int(df_filtered["bio_total"].sum())
    total_act  = int(df_filtered["total_activity"].sum())

    # Best month
    month_best = df_filtered.groupby("month")["total_activity"].sum().sort_values(ascending=False)
    best_month = month_best.index[0] if len(month_best) > 0 else "N/A"

    # Best weekday
    weekday_best = df_filtered.groupby("weekday")["total_activity"].sum().sort_values(ascending=False)
    best_day = weekday_best.index[0] if len(weekday_best) > 0 else "N/A"

    # Top district
    top_dist = df_filtered.groupby(["state","district"])["total_activity"].sum().sort_values(ascending=False)
    top_dist_name = top_dist.index[0] if len(top_dist) > 0 else ("N/A","N/A")
    top_dist_val = int(top_dist.iloc[0]) if len(top_dist) > 0 else 0

    # Updates ratio
    update_ratio = (total_demo + total_bio) / (total_act + 1)

    report = f"""
**Summary Report**
- Total Enrolments: **{total_enro}**
- Total Demographic Updates: **{total_demo}**
- Total Biometric Updates: **{total_bio}**
- Total Activity: **{total_act}**

**Peak Patterns**
- Highest Activity Month: **{best_month}**
- Highest Activity Weekday: **{best_day}**
- Top District: **{top_dist_name[1]} ({top_dist_name[0]})** with **{top_dist_val}** activity

**Service Nature**
- Updates Share (Demo+Bio / Total): **{round(update_ratio*100, 2)}%**
"""

    st.markdown(report)

    st.download_button(
        "Download Summary Report (TXT)",
        report.encode("utf-8"),
        "uidai_summary_report.txt",
        "text/plain"
    )

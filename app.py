import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np

# ── Page Config ─────────────────────────────────────────
st.set_page_config(
    page_title="AgroShe Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #f4f8f0; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a5c2a 0%, #2d8a47 60%, #4aab63 100%);
    }
    [data-testid="stSidebar"] * { color: white !important; }
    [data-testid="stSidebar"] .stSelectbox > div > div,
    [data-testid="stSidebar"] .stMultiSelect > div > div {
        background: rgba(255,255,255,0.15) !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.4) !important;
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] .stRadio > div {
        background: rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 8px;
    }
    [data-testid="stSidebar"] label { color: white !important; font-weight: 600 !important; }

    h1 { color: #1a5c2a !important; font-weight: 800 !important; }
    h2, h3 { color: #2d8a47 !important; font-weight: 700 !important; }

    .kpi-card {
        border-radius: 16px;
        padding: 20px 16px;
        text-align: center;
        box-shadow: 0 6px 20px rgba(0,0,0,0.12);
        margin-bottom: 10px;
        transition: transform 0.2s, box-shadow 0.2s;
        cursor: pointer;
    }
    .kpi-card:hover { transform: translateY(-4px); box-shadow: 0 12px 28px rgba(0,0,0,0.18); }
    .kpi-icon  { font-size: 2.2rem; margin-bottom: 4px; }
    .kpi-value { font-size: 2rem; font-weight: 800; margin: 4px 0; }
    .kpi-label { font-size: 0.82rem; font-weight: 600; opacity: 0.9; }
    .kpi-sub   { font-size: 0.75rem; opacity: 0.75; margin-top: 4px; }

    .kpi-green { background: linear-gradient(135deg, #1a5c2a, #4aab63); color: white; }
    .kpi-earth { background: linear-gradient(135deg, #7a5c1e, #c49a3c); color: white; }
    .kpi-red   { background: linear-gradient(135deg, #922b21, #e74c3c); color: white; }
    .kpi-blue  { background: linear-gradient(135deg, #1a6b8a, #2980b9); color: white; }

    .alert-green  { background:#d4edda; border-left:5px solid #28a745; padding:14px 18px; border-radius:10px; color:#155724; margin:10px 0; }
    .alert-yellow { background:#fff3cd; border-left:5px solid #ffc107; padding:14px 18px; border-radius:10px; color:#856404; margin:10px 0; }
    .alert-red    { background:#f8d7da; border-left:5px solid #dc3545; padding:14px 18px; border-radius:10px; color:#721c24; margin:10px 0; }

    .data-table { margin-top: 8px; border-radius: 10px; overflow: hidden; }

    hr { border: 1px solid #c8e6c9; margin: 1.5rem 0; }

    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #1a5c2a; border-radius: 10px; }
    ::-webkit-scrollbar-thumb { background: #a8d5b5; border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: #ffffff; }

    .stDownloadButton > button {
        background: linear-gradient(135deg, #2d8a47, #4aab63) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }

    .footer {
        text-align: center;
        padding: 18px;
        color: #2d4a1e;
        font-size: 15px;
        font-weight: 700;
        border-top: 2px solid #c8e6c9;
        margin-top: 30px;
        background: #e8f5e9;
        border-radius: 12px;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Data ────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("agrosphere_data.csv")
    df["Undernourishment_Pct"] = df["Undernourishment_Pct"].replace("<2.5", 2.5)
    df["Undernourishment_Pct"] = pd.to_numeric(df["Undernourishment_Pct"], errors="coerce")
    df.dropna(inplace=True)
    return df

df = load_data()
all_years    = sorted(df["Year"].unique())
all_countries = sorted(df["Country Name"].unique())

# ── Train AI Model ───────────────────────────────────────
@st.cache_resource
def train_model(df):
    X = df[["Women_Agriculture_Pct"]]
    y = df["Undernourishment_Pct"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    m = LinearRegression()
    m.fit(X_train, y_train)
    score = r2_score(y_test, m.predict(X_test))
    return m, score

model, model_score = train_model(df)

# ── Helper: format table ─────────────────────────────────
def show_table(data, caption="📋 Data Table"):
    st.caption(caption)
    st.dataframe(data, use_container_width=True, hide_index=True)

# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌱 AgroShe")
    st.markdown("*AI · Gender · Food Security*")
    st.markdown("---")
    st.markdown("### 📄 Navigation")
    page = st.radio("Go to", [
        "🏠 Overview & Insights",
        "🌍 Global Food & Gender Atlas",
        "📈 Progress Over Time",
        "⚖️ Head-to-Head Country Battle",
        "🤖 AI Food Risk Predictor",
        "📥 Download & Export"
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### ⚙️ Global Filters")

    year_options = ["🌐 All Years"] + [str(y) for y in all_years]
    selected_year_raw = st.selectbox("📅 Year", year_options, index=len(year_options)-1)
    selected_year = None if selected_year_raw == "🌐 All Years" else int(selected_year_raw)

    country_options = ["🌐 All Countries"] + all_countries
    selected_country_raw = st.multiselect("🌍 Countries", country_options, default=["🌐 All Countries"])
    if "🌐 All Countries" in selected_country_raw or not selected_country_raw:
        selected_countries = all_countries
        country_filter_label = "All Countries"
    else:
        selected_countries = selected_country_raw
        country_filter_label = ", ".join(selected_countries)

    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:11px; opacity:0.85; line-height:1.8;'>
    📅 Year: <b>{selected_year_raw}</b><br>
    🌍 Filter: <b>{country_filter_label[:30]}{'...' if len(country_filter_label)>30 else ''}</b><br><br>
    📊 Sources: World Bank & FAOSTAT
    </div>
    """, unsafe_allow_html=True)

# ── Filter Helper ────────────────────────────────────────
def filter_df(by_year=True, by_country=True):
    d = df.copy()
    if by_year and selected_year:
        d = d[d["Year"] == selected_year]
    if by_country:
        d = d[d["Country Name"].isin(selected_countries)]
    return d

# ════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW & INSIGHTS
# ════════════════════════════════════════════════════════
if page == "🏠 Overview & Insights":
    st.title("🌱 AgroShe - Overview & Insights")
    st.markdown("*How does women's participation in agriculture shape global food security*")
    st.markdown("---")

    d = filter_df()
    d_latest = d.sort_values("Year").drop_duplicates("Country Name", keep="last")

    avg_women  = d["Women_Agriculture_Pct"].mean()
    avg_hunger = d["Undernourishment_Pct"].mean()
    num_countries = d["Country Name"].nunique()
    high_risk  = d[d["Undernourishment_Pct"] > 15]["Country Name"].nunique()
    num_years  = d["Year"].nunique()

    # KPI Cards
    st.markdown("### 📊 Key Indicators")
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, cls, icon, val, label, sub in [
        (c1, "kpi-green", "🌍", f"{num_countries}", "Countries",            f"{num_years} year(s) of data"),
        (c2, "kpi-earth", "👩‍🌾", f"{avg_women:.1f}%", "Women in Agriculture","Avg across selection"),
        (c3, "kpi-red",   "🍽️", f"{avg_hunger:.1f}%", "Undernourishment",    "Avg hunger level"),
        (c4, "kpi-blue",  "🚨", f"{high_risk}",        "High-Risk Countries", "Hunger > 15%"),
        (c5, "kpi-green", "🤖", f"{model_score*100:.0f}%","AI Accuracy",       "R² prediction score"),
    ]:
        with col:
            st.markdown(f"""<div class='kpi-card {cls}'>
                <div class='kpi-icon'>{icon}</div>
                <div class='kpi-value'>{val}</div>
                <div class='kpi-label'>{label}</div>
                <div class='kpi-sub'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Scatter
    st.markdown("### 🔍 Core Insight: Does Gender Equality Reduce Hunger?")
    st.caption("Each bubble = one country. Bigger = higher hunger. Green = low risk, Red = high risk.")
    fig_sc = px.scatter(
        d, x="Women_Agriculture_Pct", y="Undernourishment_Pct",
        hover_name="Country Name",
        hover_data={"Women_Agriculture_Pct": ":.1f", "Undernourishment_Pct": ":.1f"},
        color="Undernourishment_Pct", size="Undernourishment_Pct",
        color_continuous_scale="RdYlGn_r",
        labels={"Women_Agriculture_Pct": "👩‍🌾 Women in Agriculture (%)",
                "Undernourishment_Pct":  "🍽️ Undernourishment (%)"},
        template="plotly_white"
    )
    fig_sc.update_layout(paper_bgcolor="#f4f8f0", plot_bgcolor="#f4f8f0",
                         font_color="#1a5c2a", height=420)
    st.plotly_chart(fig_sc, use_container_width=True)

    # Table under scatter
    tbl_sc = d_latest[["Country Name","Country Code","Women_Agriculture_Pct","Undernourishment_Pct"]].copy()
    tbl_sc.columns = ["Country","Code","Women in Agri (%)","Undernourishment (%)"]
    tbl_sc["Women in Agri (%)"]    = tbl_sc["Women in Agri (%)"].round(1)
    tbl_sc["Undernourishment (%)"] = tbl_sc["Undernourishment (%)"].round(1)
    show_table(tbl_sc.sort_values("Undernourishment (%)", ascending=False).reset_index(drop=True),
               "📋 Full Country Data - latest year per country")

    st.markdown("---")

    # Bar charts
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### 🏆 Top 10 - Women in Agriculture")
        t10 = d_latest.nlargest(10, "Women_Agriculture_Pct")
        fig_b1 = px.bar(t10, x="Women_Agriculture_Pct", y="Country Name",
                        orientation="h", color="Women_Agriculture_Pct",
                        color_continuous_scale="Greens", template="plotly_white",
                        labels={"Women_Agriculture_Pct": "Women (%)", "Country Name": ""},
                        text=t10["Women_Agriculture_Pct"].round(1).astype(str) + "%")
        fig_b1.update_traces(textposition="outside")
        fig_b1.update_layout(paper_bgcolor="#f4f8f0", plot_bgcolor="#f4f8f0",
                              font_color="#1a5c2a", showlegend=False, height=380)
        st.plotly_chart(fig_b1, use_container_width=True)
        tbl_b1 = t10[["Country Name","Women_Agriculture_Pct","Undernourishment_Pct"]].copy()
        tbl_b1.columns = ["Country","Women in Agri (%)","Hunger (%)"]
        tbl_b1["Women in Agri (%)"] = tbl_b1["Women in Agri (%)"].round(1)
        tbl_b1["Hunger (%)"]        = tbl_b1["Hunger (%)"].round(1)
        show_table(tbl_b1.reset_index(drop=True), "📋 Top 10 Women in Agriculture")

    with col_b:
        st.markdown("### 🚨 Top 10 - Highest Hunger")
        t10h = d_latest.nlargest(10, "Undernourishment_Pct")
        fig_b2 = px.bar(t10h, x="Undernourishment_Pct", y="Country Name",
                        orientation="h", color="Undernourishment_Pct",
                        color_continuous_scale="Reds", template="plotly_white",
                        labels={"Undernourishment_Pct": "Hunger (%)", "Country Name": ""},
                        text=t10h["Undernourishment_Pct"].round(1).astype(str) + "%")
        fig_b2.update_traces(textposition="outside")
        fig_b2.update_layout(paper_bgcolor="#f4f8f0", plot_bgcolor="#f4f8f0",
                              font_color="#1a5c2a", showlegend=False, height=380)
        st.plotly_chart(fig_b2, use_container_width=True)
        tbl_b2 = t10h[["Country Name","Undernourishment_Pct","Women_Agriculture_Pct"]].copy()
        tbl_b2.columns = ["Country","Hunger (%)","Women in Agri (%)"]
        tbl_b2["Hunger (%)"]        = tbl_b2["Hunger (%)"].round(1)
        tbl_b2["Women in Agri (%)"] = tbl_b2["Women in Agri (%)"].round(1)
        show_table(tbl_b2.reset_index(drop=True), "📋 Top 10 Highest Hunger Countries")

# ════════════════════════════════════════════════════════
# PAGE 2 — GLOBAL FOOD & GENDER ATLAS
# ════════════════════════════════════════════════════════
elif page == "🌍 Global Food & Gender Atlas":
    st.title("🌍 Global Food & Gender Atlas")
    st.markdown("*Interactive world maps - hover any country for full details*")
    st.markdown("---")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        map_year_opts = ["🌐 All (Average)"] + [str(y) for y in all_years]
        map_year_sel  = st.selectbox("📅 Map Year", map_year_opts,
                                      index=map_year_opts.index(str(selected_year)) if selected_year else 0)
    with col_f2:
        show_labels = st.toggle("Show Country Labels", value=False)

    if map_year_sel != "🌐 All (Average)":
        d_map = df[df["Year"] == int(map_year_sel)].copy()
    else:
        d_map = df.groupby(["Country Name","Country Code"], as_index=False).mean(numeric_only=True)
        d_map["Year"] = "Average"

    # Map 1 — Women in Agriculture
    st.markdown("### 👩‍🌾 Women's Share in Agriculture")
    st.caption("Darker green = more women working in agriculture. Hover any country to see values.")
    fig_m1 = px.choropleth(
        d_map, locations="Country Code",
        color="Women_Agriculture_Pct",
        hover_name="Country Name",
        custom_data=["Women_Agriculture_Pct","Undernourishment_Pct"],
        color_continuous_scale="Greens",
        labels={"Women_Agriculture_Pct": "Women in Agri (%)"},
        template="plotly_white"
    )
    fig_m1.update_traces(
        hovertemplate="<b>%{hovertext}</b><br>👩‍🌾 Women in Agriculture: <b>%{customdata[0]:.1f}%</b><br>🍽️ Undernourishment: <b>%{customdata[1]:.1f}%</b><extra></extra>"
    )
    if show_labels:
        fig_m1.add_scattergeo(
            locations=d_map["Country Code"], text=d_map["Country Name"],
            mode="text", textfont=dict(size=7, color="black")
        )
    fig_m1.update_layout(
        height=520, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="#f4f8f0",
        geo=dict(showframe=False, showcoastlines=True,
                 coastlinecolor="#2d8a47", landcolor="#e8f5e9",
                 oceancolor="#cce5ff", showocean=True,
                 lakecolor="#cce5ff", bgcolor="#f4f8f0")
    )
    st.plotly_chart(fig_m1, use_container_width=True)

    # Table under map 1
    tbl_m1 = d_map[["Country Name","Country Code","Women_Agriculture_Pct","Undernourishment_Pct"]].copy()
    tbl_m1.columns = ["Country","Code","Women in Agri (%)","Undernourishment (%)"]
    tbl_m1["Women in Agri (%)"]    = tbl_m1["Women in Agri (%)"].round(1)
    tbl_m1["Undernourishment (%)"] = tbl_m1["Undernourishment (%)"].round(1)
    show_table(tbl_m1.sort_values("Women in Agri (%)", ascending=False).reset_index(drop=True),
               "📋 Women in Agriculture — All Countries")

    st.markdown("---")

    # Map 2 - Undernourishment
    st.markdown("### 🍽️ Undernourishment Rate")
    st.caption("Darker red = higher hunger. Hover any country to see values.")
    fig_m2 = px.choropleth(
        d_map, locations="Country Code",
        color="Undernourishment_Pct",
        hover_name="Country Name",
        custom_data=["Undernourishment_Pct","Women_Agriculture_Pct"],
        color_continuous_scale="YlOrRd",
        labels={"Undernourishment_Pct": "Hunger (%)"},
        template="plotly_white"
    )
    fig_m2.update_traces(
        hovertemplate="<b>%{hovertext}</b><br>🍽️ Undernourishment: <b>%{customdata[0]:.1f}%</b><br>👩‍🌾 Women in Agriculture: <b>%{customdata[1]:.1f}%</b><extra></extra>"
    )
    if show_labels:
        fig_m2.add_scattergeo(
            locations=d_map["Country Code"], text=d_map["Country Name"],
            mode="text", textfont=dict(size=7, color="black")
        )
    fig_m2.update_layout(
        height=520, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="#f4f8f0",
        geo=dict(showframe=False, showcoastlines=True,
                 coastlinecolor="#c0392b", landcolor="#fff8f0",
                 oceancolor="#cce5ff", showocean=True,
                 lakecolor="#cce5ff", bgcolor="#f4f8f0")
    )
    st.plotly_chart(fig_m2, use_container_width=True)

    # Table under map 2
    tbl_m2 = d_map[["Country Name","Country Code","Undernourishment_Pct","Women_Agriculture_Pct"]].copy()
    tbl_m2.columns = ["Country","Code","Undernourishment (%)","Women in Agri (%)"]
    tbl_m2["Undernourishment (%)"] = tbl_m2["Undernourishment (%)"].round(1)
    tbl_m2["Women in Agri (%)"]    = tbl_m2["Women in Agri (%)"].round(1)
    show_table(tbl_m2.sort_values("Undernourishment (%)", ascending=False).reset_index(drop=True),
               "📋 Undernourishment - All Countries")

# ════════════════════════════════════════════════════════
# PAGE 3 — PROGRESS OVER TIME
# ════════════════════════════════════════════════════════
elif page == "📈 Progress Over Time":
    st.title("📈 Progress Over Time")
    st.markdown("*Track how countries are advancing - or falling behind - on gender and food security*")
    st.markdown("---")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        trend_countries = st.multiselect(
            "🌍 Select Countries to Track", all_countries,
            default=["Rwanda","Nigeria","India","Ethiopia","Bangladesh"]
        )
    with col_f2:
        trend_year_range = st.select_slider(
            "📅 Year Range", options=all_years,
            value=(min(all_years), max(all_years))
        )

    if not trend_countries:
        st.warning("⚠️ Please select at least one country.")
        st.stop()

    d_trend = df[
        (df["Country Name"].isin(trend_countries)) &
        (df["Year"] >= trend_year_range[0]) &
        (df["Year"] <= trend_year_range[1])
    ]

    # Chart 1
    st.markdown("### 👩‍🌾 Women in Agriculture - How Has It Changed?")
    fig_t1 = px.line(d_trend, x="Year", y="Women_Agriculture_Pct",
                     color="Country Name", markers=True,
                     color_discrete_sequence=px.colors.qualitative.Safe,
                     labels={"Women_Agriculture_Pct": "Women in Agri (%)", "Country Name": "Country"},
                     template="plotly_white")
    fig_t1.update_traces(line=dict(width=2.5))
    fig_t1.update_layout(paper_bgcolor="#f4f8f0", plot_bgcolor="#f4f8f0",
                          font_color="#1a5c2a", height=400,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig_t1, use_container_width=True)
    tbl_t1 = d_trend.pivot_table(index="Country Name", columns="Year",
                                  values="Women_Agriculture_Pct").round(1).reset_index()
    tbl_t1.columns.name = None
    show_table(tbl_t1, "📋 Women in Agriculture (%) by Country & Year")

    st.markdown("---")

    # Chart 2
    st.markdown("### 🍽️ Undernourishment - Are Countries Getting Better?")
    fig_t2 = px.line(d_trend, x="Year", y="Undernourishment_Pct",
                     color="Country Name", markers=True,
                     color_discrete_sequence=px.colors.qualitative.Safe,
                     labels={"Undernourishment_Pct": "Hunger (%)", "Country Name": "Country"},
                     template="plotly_white")
    fig_t2.update_traces(line=dict(width=2.5))
    fig_t2.update_layout(paper_bgcolor="#f4f8f0", plot_bgcolor="#f4f8f0",
                          font_color="#1a5c2a", height=400,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig_t2, use_container_width=True)
    tbl_t2 = d_trend.pivot_table(index="Country Name", columns="Year",
                                  values="Undernourishment_Pct").round(1).reset_index()
    tbl_t2.columns.name = None
    show_table(tbl_t2, "📋 Undernourishment (%) by Country & Year")

    st.markdown("---")

    # Combined view — Grouped Bar (friendly, replaces area chart)
    st.markdown("### 📊 Combined View - Women in Agriculture vs Hunger Side by Side")
    st.caption("Compare both indicators for each country at a glance.")
    d_trend_latest = d_trend.sort_values("Year").drop_duplicates("Country Name", keep="last")
    d_melt = d_trend_latest.melt(
        id_vars="Country Name",
        value_vars=["Women_Agriculture_Pct","Undernourishment_Pct"],
        var_name="Indicator", value_name="Value"
    )
    d_melt["Indicator"] = d_melt["Indicator"].map({
        "Women_Agriculture_Pct": "👩‍🌾 Women in Agri (%)",
        "Undernourishment_Pct":  "🍽️ Undernourishment (%)"
    })
    fig_grp = px.bar(
        d_melt, x="Country Name", y="Value", color="Indicator",
        barmode="group",
        color_discrete_map={
            "👩‍🌾 Women in Agri (%)": "#2d8a47",
            "🍽️ Undernourishment (%)": "#e74c3c"
        },
        text=d_melt["Value"].round(1).astype(str) + "%",
        labels={"Value": "(%)", "Country Name": ""},
        template="plotly_white"
    )
    fig_grp.update_traces(textposition="outside")
    fig_grp.update_layout(paper_bgcolor="#f4f8f0", plot_bgcolor="#f4f8f0",
                           font_color="#1a5c2a", height=420,
                           legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig_grp, use_container_width=True)
    tbl_comb = d_trend_latest[["Country Name","Women_Agriculture_Pct","Undernourishment_Pct"]].copy()
    tbl_comb.columns = ["Country","Women in Agri (%)","Undernourishment (%)"]
    tbl_comb["Women in Agri (%)"]    = tbl_comb["Women in Agri (%)"].round(1)
    tbl_comb["Undernourishment (%)"] = tbl_comb["Undernourishment (%)"].round(1)
    show_table(tbl_comb.reset_index(drop=True), "📋 Combined Indicators — Latest Year")

# ════════════════════════════════════════════════════════
# PAGE 4 — HEAD-TO-HEAD COUNTRY BATTLE
# ════════════════════════════════════════════════════════
elif page == "⚖️ Head-to-Head Country Battle":
    st.title("⚖️ Head-to-Head Country Battle")
    st.markdown("*Compare any two countries side by side across all indicators and years*")
    st.markdown("---")

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        country_a = st.selectbox("🟢 Country A", all_countries,
                                  index=all_countries.index("Rwanda") if "Rwanda" in all_countries else 0)
    with col_f2:
        country_b = st.selectbox("🔴 Country B", all_countries,
                                  index=all_countries.index("Nigeria") if "Nigeria" in all_countries else 1)
    with col_f3:
        compare_year_opts = ["🌐 All Years"] + [str(y) for y in all_years]
        compare_year = st.selectbox("📅 Year", compare_year_opts, index=len(compare_year_opts)-1)

    df_a = df[df["Country Name"] == country_a].sort_values("Year")
    df_b = df[df["Country Name"] == country_b].sort_values("Year")

    snap_a = df_a[df_a["Year"] == int(compare_year)] if compare_year != "🌐 All Years" else df_a.tail(1)
    snap_b = df_b[df_b["Year"] == int(compare_year)] if compare_year != "🌐 All Years" else df_b.tail(1)

    if not snap_a.empty and not snap_b.empty:
        wa = snap_a["Women_Agriculture_Pct"].values[0]
        wb = snap_b["Women_Agriculture_Pct"].values[0]
        ha = snap_a["Undernourishment_Pct"].values[0]
        hb = snap_b["Undernourishment_Pct"].values[0]

        st.markdown("### 📊 Snapshot Comparison")
        c1, c2, c3, c4 = st.columns(4)
        for col, cls, icon, val, lbl, sub in [
            (c1, "kpi-green", "👩‍🌾", f"{wa:.1f}%", f"🟢 {country_a}", "Women in Agriculture"),
            (c2, "kpi-earth", "👩‍🌾", f"{wb:.1f}%", f"🔴 {country_b}", "Women in Agriculture"),
            (c3, "kpi-green" if ha < hb else "kpi-red", "🍽️", f"{ha:.1f}%", f"🟢 {country_a}", "Undernourishment"),
            (c4, "kpi-green" if hb < ha else "kpi-red", "🍽️", f"{hb:.1f}%", f"🔴 {country_b}", "Undernourishment"),
        ]:
            with col:
                st.markdown(f"""<div class='kpi-card {cls}'>
                    <div class='kpi-icon'>{icon}</div><div class='kpi-value'>{val}</div>
                    <div class='kpi-label'>{lbl}</div><div class='kpi-sub'>{sub}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")
        winner_women  = country_a if wa > wb else country_b
        winner_hunger = country_a if ha < hb else country_b
        st.success(f"👩‍🌾 **{winner_women}** has more women in agriculture ({max(wa,wb):.1f}%)")
        st.success(f"🍽️ **{winner_hunger}** has better food security ({min(ha,hb):.1f}% hunger)")

    st.markdown("---")

    # Trend lines
    df_compare = pd.concat([df_a.assign(Country=country_a), df_b.assign(Country=country_b)])

    col_l, col_r = st.columns(2)
    with col_l:
        fig_c1 = px.line(df_compare, x="Year", y="Women_Agriculture_Pct",
                         color="Country", markers=True,
                         color_discrete_map={country_a:"#2d8a47", country_b:"#e74c3c"},
                         title="👩‍🌾 Women in Agriculture (%)", template="plotly_white",
                         labels={"Women_Agriculture_Pct":"Women (%)"})
        fig_c1.update_layout(paper_bgcolor="#f4f8f0", plot_bgcolor="#f4f8f0", font_color="#1a5c2a")
        st.plotly_chart(fig_c1, use_container_width=True)

    with col_r:
        fig_c2 = px.line(df_compare, x="Year", y="Undernourishment_Pct",
                         color="Country", markers=True,
                         color_discrete_map={country_a:"#2d8a47", country_b:"#e74c3c"},
                         title="🍽️ Undernourishment (%)", template="plotly_white",
                         labels={"Undernourishment_Pct":"Hunger (%)"})
        fig_c2.update_layout(paper_bgcolor="#f4f8f0", plot_bgcolor="#f4f8f0", font_color="#1a5c2a")
        st.plotly_chart(fig_c2, use_container_width=True)

    # Table under trends
    tbl_cmp = df_compare[["Country","Year","Women_Agriculture_Pct","Undernourishment_Pct"]].copy()
    tbl_cmp.columns = ["Country","Year","Women in Agri (%)","Undernourishment (%)"]
    tbl_cmp["Women in Agri (%)"]    = tbl_cmp["Women in Agri (%)"].round(1)
    tbl_cmp["Undernourishment (%)"] = tbl_cmp["Undernourishment (%)"].round(1)
    show_table(tbl_cmp.sort_values(["Country","Year"]).reset_index(drop=True),
               f"📋 {country_a} vs {country_b} - All Years Data")

    st.markdown("---")

    # ── Friendly Grouped Bar (replaces radar) ──
    st.markdown("### 📊 Side-by-Side Indicator Comparison")
    st.caption("A clear bar comparison of both countries across both key indicators.")
    bar_data = pd.DataFrame({
        "Indicator": ["👩‍🌾 Women in Agri (%)", "👩‍🌾 Women in Agri (%)",
                      "🍽️ Undernourishment (%)", "🍽️ Undernourishment (%)"],
        "Country":   [country_a, country_b, country_a, country_b],
        "Value":     [wa, wb, ha, hb]
    })
    fig_bar_cmp = px.bar(
        bar_data, x="Indicator", y="Value", color="Country",
        barmode="group",
        color_discrete_map={country_a: "#2d8a47", country_b: "#e74c3c"},
        text=bar_data["Value"].round(1).astype(str) + "%",
        template="plotly_white",
        labels={"Value": "Value (%)"}
    )
    fig_bar_cmp.update_traces(textposition="outside")
    fig_bar_cmp.update_layout(
        paper_bgcolor="#f4f8f0", plot_bgcolor="#f4f8f0",
        font_color="#1a5c2a", height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig_bar_cmp, use_container_width=True)

    tbl_bar = pd.DataFrame({
        "Country":             [country_a, country_b],
        "Women in Agri (%)":   [round(wa,1), round(wb,1)],
        "Undernourishment (%)": [round(ha,1), round(hb,1)],
    })
    show_table(tbl_bar, f"📋 {country_a} vs {country_b} - Indicator Summary")

# ════════════════════════════════════════════════════════
# PAGE 5 — AI FOOD RISK PREDICTOR
# ════════════════════════════════════════════════════════
elif page == "🤖 AI Food Risk Predictor":
    st.title("🤖 AI Food Risk Predictor")
    st.markdown("*Use AI to predict a country's food insecurity risk based on women's agricultural participation*")
    st.markdown("---")

    pred_mode = st.radio(
        "🎯 Prediction Mode",
        ["🎛️ Manual Slider", "🌍 Pick a Specific Country", "🌐 Compare All Countries"],
        horizontal=True
    )
    st.markdown("---")

    # ── Mode 1: Manual Slider ──
    if pred_mode == "🎛️ Manual Slider":
        col1, col2 = st.columns([1,1])
        with col1:
            user_input = st.slider("👩‍🌾 Women in Agriculture (%)", 0.0, 100.0, 30.0, 0.5)
            prediction = max(0, model.predict([[user_input]])[0])

            if prediction < 5:
                st.markdown(f"""<div class='alert-green'>
                    ✅ <strong>LOW RISK — {prediction:.1f}% Undernourishment</strong><br>
                    Women's participation strongly supports food security.<br>
                    💡 <em>Recommendation: Maintain policies that keep women active in agriculture.
                    Expand women's land ownership rights to sustain progress.</em>
                </div>""", unsafe_allow_html=True)
            elif prediction < 15:
                st.markdown(f"""<div class='alert-yellow'>
                    ⚠️ <strong>MODERATE RISK — {prediction:.1f}% Undernourishment</strong><br>
                    Food insecurity is present but manageable with targeted interventions.<br>
                    💡 <em>Recommendation: Invest in training programs for women farmers.
                    Improve access to agricultural credit and markets for women.</em>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class='alert-red'>
                    🚨 <strong>HIGH RISK — {prediction:.1f}% Undernourishment</strong><br>
                    Low women's participation is linked to severe hunger. Urgent action needed.<br>
                    💡 <em>Recommendation: Prioritize gender-inclusive agricultural policies.
                    FAO intervention programs and school feeding initiatives are critical.</em>
                </div>""", unsafe_allow_html=True)

            st.markdown(f"<br>🤖 **Model Accuracy: {model_score*100:.1f}%**", unsafe_allow_html=True)

        with col2:
            global_avg = df["Undernourishment_Pct"].mean()
            sdg_target = 2.5
            fig_bm = go.Figure()
            fig_bm.add_trace(go.Bar(
                x=["🤖 Your Prediction", "🌍 Global Average", "🎯 SDG2 Target"],
                y=[prediction, global_avg, sdg_target],
                marker_color=[
                    "#e74c3c" if prediction > 15 else "#f39c12" if prediction > 5 else "#2d8a47",
                    "#2980b9", "#27ae60"
                ],
                text=[f"{prediction:.1f}%", f"{global_avg:.1f}%", f"{sdg_target}%"],
                textposition="outside", width=0.4
            ))
            fig_bm.update_layout(
                title="📊 Prediction vs Benchmarks",
                paper_bgcolor="#f4f8f0", plot_bgcolor="#f4f8f0",
                font_color="#1a5c2a", height=380,
                yaxis=dict(title="Undernourishment (%)", range=[0, max(prediction, global_avg)+8]),
                showlegend=False
            )
            st.plotly_chart(fig_bm, use_container_width=True)

        show_table(pd.DataFrame({
            "Metric":      ["Your Prediction","Global Average","SDG2 Target"],
            "Undernourishment (%)": [round(prediction,1), round(global_avg,1), 2.5]
        }), "📋 Prediction vs Benchmarks")

    # ── Mode 2: Specific Country ──
    elif pred_mode == "🌍 Pick a Specific Country":
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            sel_country = st.selectbox("🌍 Choose Country", all_countries)
        with col_f2:
            yr_opts = ["Latest"] + [str(y) for y in all_years]
            sel_yr  = st.selectbox("📅 Year", yr_opts)

        d_c = df[df["Country Name"] == sel_country]
        d_c = d_c[d_c["Year"] == int(sel_yr)] if sel_yr != "Latest" else d_c.tail(1)

        if not d_c.empty:
            actual_women  = d_c["Women_Agriculture_Pct"].values[0]
            actual_hunger = d_c["Undernourishment_Pct"].values[0]
            predicted     = max(0, model.predict([[actual_women]])[0])

            c1, c2, c3 = st.columns(3)
            for col, cls, icon, val, lbl, sub in [
                (c1,"kpi-green","👩‍🌾",f"{actual_women:.1f}%","Women in Agriculture",sel_country),
                (c2,"kpi-red",  "🍽️", f"{actual_hunger:.1f}%","Actual Hunger","From FAOSTAT"),
                (c3,"kpi-blue", "🤖", f"{predicted:.1f}%",    "AI Predicted Hunger","Model estimate"),
            ]:
                with col:
                    st.markdown(f"""<div class='kpi-card {cls}'>
                        <div class='kpi-icon'>{icon}</div><div class='kpi-value'>{val}</div>
                        <div class='kpi-label'>{lbl}</div><div class='kpi-sub'>{sub}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("---")
            if predicted < 5:
                st.markdown(f"""<div class='alert-green'>
                    ✅ <strong>{sel_country} — LOW RISK</strong><br>
                    Women's participation at {actual_women:.1f}% supports strong food security.<br>
                    💡 <em>Recommendation: Scale up digital tools for women farmers to boost productivity further.</em>
                </div>""", unsafe_allow_html=True)
            elif predicted < 15:
                st.markdown(f"""<div class='alert-yellow'>
                    ⚠️ <strong>{sel_country} — MODERATE RISK</strong><br>
                    Room for improvement in both gender inclusion and food security.<br>
                    💡 <em>Recommendation: Strengthen women's cooperatives and access to seeds, tools, and markets.</em>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class='alert-red'>
                    🚨 <strong>{sel_country} — HIGH RISK</strong><br>
                    Low women's participation ({actual_women:.1f}%) is correlated with high hunger.<br>
                    💡 <em>Recommendation: Emergency programs — women's land rights, nutrition support, and climate-resilient farming training.</em>
                </div>""", unsafe_allow_html=True)

            show_table(pd.DataFrame({
                "Metric":  ["Women in Agriculture (%)","Actual Hunger (%)","AI Predicted Hunger (%)"],
                "Value":   [round(actual_women,1), round(actual_hunger,1), round(predicted,1)],
                "Country": [sel_country]*3
            }), f"📋 {sel_country} — Full Indicator Summary")

    # ── Mode 3: All Countries ──
    elif pred_mode == "🌐 Compare All Countries":
        st.markdown("### 🌐 AI Risk Assessment — All Countries")
        sel_yr_all = st.selectbox("📅 Select Year", [str(y) for y in all_years], index=len(all_years)-1)

        d_all = df[df["Year"] == int(sel_yr_all)].copy()
        d_all["AI_Predicted_Hunger"] = d_all["Women_Agriculture_Pct"].apply(
            lambda x: max(0, model.predict([[x]])[0])
        )
        d_all["Risk Level"] = d_all["AI_Predicted_Hunger"].apply(
            lambda x: "🟢 Low Risk" if x < 5 else ("🟡 Moderate Risk" if x < 15 else "🔴 High Risk")
        )

        risk_counts = d_all["Risk Level"].value_counts()
        c1, c2, c3 = st.columns(3)
        for col, lbl, cls, icon in [
            (c1,"🟢 Low Risk",   "kpi-green","✅"),
            (c2,"🟡 Moderate Risk","kpi-earth","⚠️"),
            (c3,"🔴 High Risk",  "kpi-red",  "🚨"),
        ]:
            with col:
                count = risk_counts.get(lbl, 0)
                st.markdown(f"""<div class='kpi-card {cls}'>
                    <div class='kpi-icon'>{icon}</div>
                    <div class='kpi-value'>{count}</div>
                    <div class='kpi-label'>{lbl.split(" ",1)[1]} Countries</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")
        fig_risk = px.bar(
            d_all.sort_values("AI_Predicted_Hunger", ascending=False).head(30),
            x="Country Name", y="AI_Predicted_Hunger",
            color="Risk Level",
            color_discrete_map={"🟢 Low Risk":"#2d8a47","🟡 Moderate Risk":"#f39c12","🔴 High Risk":"#e74c3c"},
            text=d_all.sort_values("AI_Predicted_Hunger",ascending=False).head(30)["AI_Predicted_Hunger"].round(1).astype(str)+"%",
            labels={"AI_Predicted_Hunger":"Predicted Hunger (%)","Country Name":""},
            title="Top 30 Countries by Predicted Food Risk",
            template="plotly_white"
        )
        fig_risk.update_traces(textposition="outside")
        fig_risk.update_layout(paper_bgcolor="#f4f8f0", plot_bgcolor="#f4f8f0",
                                font_color="#1a5c2a", height=480, xaxis_tickangle=-45)
        st.plotly_chart(fig_risk, use_container_width=True)

        tbl_all = d_all[["Country Name","Women_Agriculture_Pct","Undernourishment_Pct",
                          "AI_Predicted_Hunger","Risk Level"]].copy()
        tbl_all.columns = ["Country","Women in Agri (%)","Actual Hunger (%)","AI Predicted (%)","Risk Level"]
        tbl_all["Women in Agri (%)"]  = tbl_all["Women in Agri (%)"].round(1)
        tbl_all["Actual Hunger (%)"]  = tbl_all["Actual Hunger (%)"].round(1)
        tbl_all["AI Predicted (%)"]   = tbl_all["AI Predicted (%)"].round(1)
        show_table(tbl_all.sort_values("AI Predicted (%)", ascending=False).reset_index(drop=True),
                   "📋 Full Country Risk Table")

# ════════════════════════════════════════════════════════
# PAGE 6 — DOWNLOAD & EXPORT
# ════════════════════════════════════════════════════════
elif page == "📥 Download & Export":
    st.title("📥 Download & Export")
    st.markdown("*Download the data powering this dashboard*")
    st.markdown("---")

    d = filter_df()
    st.markdown("### 📋 Data Preview")
    st.dataframe(
        d.style.background_gradient(subset=["Women_Agriculture_Pct"], cmap="Greens")
               .background_gradient(subset=["Undernourishment_Pct"], cmap="Reds"),
        use_container_width=True, height=380
    )
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 📥 Full Dataset")
        st.download_button("⬇️ Download All Data",
                           df.to_csv(index=False).encode("utf-8"),
                           "agrosphere_full.csv","text/csv")
    with col2:
        st.markdown("#### 📥 Current Filter")
        st.download_button("⬇️ Download Filtered",
                           d.to_csv(index=False).encode("utf-8"),
                           "agrosphere_filtered.csv","text/csv")
    with col3:
        st.markdown("#### 📥 Statistics Summary")
        st.download_button("⬇️ Download Summary",
                           df.describe().round(2).to_csv().encode("utf-8"),
                           "agrosphere_summary.csv","text/csv")

    st.markdown("---")
    st.markdown("### 📊 Statistical Summary")
    st.dataframe(df.describe().round(2), use_container_width=True)

# ── Footer ───────────────────────────────────────────────
st.markdown("""
<div class='footer'>
    🌱 AgroShe - The Future We Aspire 🌱
</div>
""", unsafe_allow_html=True)
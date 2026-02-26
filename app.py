import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

import tempfile

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="National Intelligence Portal",
    layout="wide"
)

# ---------------- LOAD ALL DATASETS ----------------

@st.cache_data
def load_data():

    def clean_df(path):
        df = pd.read_csv(path)

        if "Description" in df.columns:
            df = df.drop(columns=["Description"])

        return df.set_index("Year")

    return {
        "GDP": clean_df("data/gdp_Haryana.csv"),
        "Income": clean_df("data/income_Haryana.csv"),
        "Literacy": clean_df("data/literacy_Haryana.csv"),
        "Employment": clean_df("data/employment_Haryana.csv")
    }

datasets = load_data()

# ---------------- SESSION CONTROL ----------------

if "page" not in st.session_state:
    st.session_state.page = "welcome"

# ---------------- WELCOME PAGE ----------------

if st.session_state.page == "welcome":

    st.image("images.jpg", use_container_width=True)

    st.title("National Socio Economic Intelligence Portal")

    st.markdown("""
    AI Based Governance Analytics Platform

    ✔ District Development Analysis  
    ✔ Multi Indicator Prediction  
    ✔ Policy Intelligence Reports  
    """)

    if st.button("Enter Dashboard 🚀"):
        st.session_state.page = "dashboard"
        st.rerun()

    st.stop()

# ---------------- DASHBOARD ----------------

st.title("📊 Socio Economic Intelligence Dashboard")

if st.button("Home"):
    st.session_state.page = "welcome"
    st.rerun()

# ---------------- USER SELECTION ----------------

# Require explicit user selection (no pre-selected values)
category_options = ["-- Select Indicator --"] + list(datasets.keys())
category = st.selectbox("Select Indicator", category_options, key="category")

if category == "-- Select Indicator --":
    st.info("Please select an Indicator to continue.")
    st.stop()

df = datasets[category]

district_options = ["-- Select District --"] + list(df.columns)
district = st.selectbox("Select District", district_options, key="district")

if district == "-- Select District --":
    st.info("Please select a District to continue.")
    st.stop()

# ---------------- TREND ANALYSIS ----------------

st.subheader(f"{district} - {category} Trend")

# interactive trend chart (Plotly) with hover markers and units
units_map = {
    "GDP": "₹",
    "Income": "₹",
    "Literacy": "%",
    "Employment": "%"
}

units = units_map.get(category, "units")

df_trend = df.reset_index()
df_trend = df_trend.rename(columns={df_trend.columns[0]: "Year"})

fig = px.line(
    df_trend,
    x="Year",
    y=district,
    markers=True,
    labels={district: category},
    title=f"{district} - {category} Trend"
)
fig.update_traces(hovertemplate=f"%{{y:.2f}} {units}", line=dict(color="royalblue"))
fig.update_layout(hovermode="x unified")

st.plotly_chart(fig, use_container_width=True)

# ---------------- RANKING ----------------

st.subheader("🏆 District Ranking (Latest Year)")

latest = df.iloc[-1].sort_values(ascending=False)

ranking_df = latest.reset_index()
ranking_df.columns = ["District", category]

st.dataframe(ranking_df, use_container_width=True)

# ---------------- GROWTH ANALYSIS ----------------

st.subheader("📈 Growth Analysis")

growth = ((df.iloc[-1] - df.iloc[0]) / df.iloc[0]) * 100
growth_df = growth.reset_index()
growth_df.columns = ["District", "Growth"]

# interactive bar chart with distinct colors per bar
fig2 = px.bar(
    growth_df,
    x="District",
    y="Growth",
    color="District",
    color_discrete_sequence=px.colors.qualitative.Vivid,
    labels={"Growth": "Growth (%)"}
)
fig2.update_traces(hovertemplate="%{y:.2f}%")
st.plotly_chart(fig2, use_container_width=True)

# ---------------- AI PREDICTION MODEL ----------------

st.subheader("🧠 AI Prediction Engine")

X = np.array(df.index).reshape(-1,1)
y = df[district].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor()
model.fit(X_train, y_train)

mae = mean_absolute_error(y_test, model.predict(X_test))
r2 = r2_score(y_test, model.predict(X_test))

st.write(f"Model MAE : {mae:.2f}")
st.write(f"Model R2 Score : {r2:.2f}")

# ---------------- FUTURE PREDICTION ----------------

future_year = st.number_input("Enter Future Year", value=2026)

# make the Predict button visually prominent (dark)
st.markdown(
    """
    <style>
    .dark-btn button {
        background-color:#0b3d91;
        color: white;
        border-radius:6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="dark-btn">', unsafe_allow_html=True)
if st.button("Predict Future Value"):

    future_scaled = scaler.transform([[future_year]])
    prediction = model.predict(future_scaled)[0]

    units = units_map.get(category, "units")

    st.success(
        f"Predicted {category} for {district} in {future_year} = {prediction:.2f} {units}"
    )

    policy_text = """
    Development projection suggests continuous government monitoring.

    Recommended Focus:
    - Infrastructure Development  
    - Employment Generation  
    - Education Quality Improvement  
    """

    st.subheader("📜 Policy Recommendation")
    st.write(policy_text)

    # ---------------- PDF REPORT ----------------

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")

    doc = SimpleDocTemplate(temp_file.name)
    styles = getSampleStyleSheet()

    elements = []

    elements.append(
        Paragraph("Socio Economic Intelligence Report", styles["Title"])
    )

    elements.append(Spacer(1, inch*0.3))

    elements.append(Paragraph(f"Indicator: {category}", styles["Normal"]))
    elements.append(Paragraph(f"District: {district}", styles["Normal"]))
    elements.append(
        Paragraph(
            f"Prediction {future_year}: {prediction:.2f} {units}",
            styles["Normal"]
        )
    )

    doc.build(elements)

    with open(temp_file.name, "rb") as f:
        st.download_button(
            "Download Report PDF",
            f,
            file_name="Intelligence_Report.pdf"
        )
st.markdown('</div>', unsafe_allow_html=True)
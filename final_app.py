import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import tempfile

st.set_page_config(page_title="Government Intelligence Portal", layout="wide")

# ---------------- PAGE CONTROL ----------------
if "page" not in st.session_state:
    st.session_state.page = "welcome"

# ---------------- WELCOME PAGE ----------------
if st.session_state.page == "welcome":

    st.image("images.jpg", use_container_width=True)

    st.markdown("""
    <h1 style='text-align:center;color:#0b3d91;'>
    National Socio-Economic Intelligence Portal
    </h1>
    """, unsafe_allow_html=True)

    st.markdown("""
    This AI-powered governance platform strengthens public policy and
    government decision-making through predictive analytics,
    district comparison, and strategic recommendations.

    ✔ AI-Based Forecasting  
    ✔ District Performance Analysis  
    ✔ Growth Comparison  
    ✔ Policy Recommendation Engine  
    """)

    if st.button("🚀 Enter Analytics Dashboard"):
        st.session_state.page = "dashboard"
        st.rerun()

    st.stop()

# ---------------- DASHBOARD ----------------
col1, col2 = st.columns([8,1])
with col2:
    if st.button("⬅ Home"):
        st.session_state.page = "welcome"
        st.rerun()

st.title("📊 District Development Analytics Dashboard")

st.markdown("---")

# ---------------- SAMPLE DATA ----------------
years = list(range(2015, 2024))

base_data = {
    "Ambala": [50, 55, 60, 65, 70, 72, 75, 80, 85],
    "Hisar": [40, 42, 45, 48, 50, 52, 55, 58, 60],
    "Rohtak": [45, 47, 49, 52, 54, 57, 60, 63, 65],
    "Gurgaon": [80, 85, 90, 100, 110, 120, 130, 145, 160],
    "Panipat": [35, 37, 40, 42, 45, 48, 50, 53, 55]
}

categories = {
    "GDP": pd.DataFrame(base_data, index=years),
    "Income": pd.DataFrame({k: np.array(v)*2 for k,v in base_data.items()}, index=years),
    "Literacy": pd.DataFrame({k: np.array(v)/2 for k,v in base_data.items()}, index=years),
    "Employment": pd.DataFrame({k: np.array(v)*1.5 for k,v in base_data.items()}, index=years),
}

# ---------------- USER SELECTION ----------------
st.subheader("Select Analysis Parameters")

category = st.selectbox("Choose Category", ["-- Select Category --"] + list(categories.keys()))

if category == "-- Select Category --":
    st.warning("Please select a category.")
    st.stop()

df = categories[category]

district = st.selectbox("Choose District", ["-- Select District --"] + list(df.columns))

if district == "-- Select District --":
    st.warning("Please select a district.")
    st.stop()

# ---------------- TREND ----------------
st.subheader(f"{district} - {category} Trend Analysis")

fig, ax = plt.subplots(figsize=(8,4))
ax.plot(df.index, df[district], marker="o")
ax.set_xlabel("Year")
ax.set_ylabel(category)
st.pyplot(fig)

# ---------------- RANKING ----------------
st.subheader("🏆 District Ranking (Latest Year)")

latest = df.iloc[-1].sort_values(ascending=False)
ranking_df = latest.reset_index()
ranking_df.columns = ["District", category]
st.dataframe(ranking_df, use_container_width=True)

# ---------------- COMPARISON ----------------
st.subheader("📊 Latest Year Comparison")

fig2, ax2 = plt.subplots()
latest.plot(kind="bar", ax=ax2)
plt.xticks(rotation=45)
st.pyplot(fig2)

# ---------------- GROWTH ----------------
st.subheader("📈 Overall Growth (2015-2023)")

growth = ((df.iloc[-1] - df.iloc[0]) / df.iloc[0]) * 100
fig3, ax3 = plt.subplots()
growth.plot(kind="bar", ax=ax3)
plt.xticks(rotation=45)
st.pyplot(fig3)

# ---------------- AI MODEL ----------------
st.subheader("🧠 AI Prediction Model")

X = np.array(df.index).reshape(-1,1)
y = df[district].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor()
model.fit(X_train, y_train)

mae = mean_absolute_error(y_test, model.predict(X_test))
r2 = r2_score(y_test, model.predict(X_test))

st.write(f"Model MAE: {mae:.2f}")
st.write(f"Model R² Score: {r2:.2f}")
# ---------------- FUTURE PREDICTION ----------------
future_year = st.number_input("Enter Future Year", value=2025)

if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

if st.button("Predict Future Value"):
    future_scaled = scaler.transform([[future_year]])
    prediction = model.predict(future_scaled)[0]

    st.session_state.prediction = prediction
    st.session_state.prediction_done = True

if st.session_state.prediction_done:

    prediction = st.session_state.prediction

    st.success(f"Predicted {category} for {district} in {future_year}: {prediction:.2f}")

    policy_text = f"""
    The projected value for {district} indicates its expected development trajectory.
    Sustained growth suggests continuation of current governance strategies.

    If stagnation or decline is observed, targeted fiscal intervention,
    infrastructure expansion, employment generation programs,
    and sector-specific reforms should be implemented.

    Limitations:
    This model is trend-based and does not account for economic shocks,
    demographic shifts, or policy changes.
    """

    st.subheader("📜 Policy Recommendation")
    st.write(policy_text)

    # -------- Generate PDF Automatically --------
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(temp_file.name)
    elements = []

    styles = getSampleStyleSheet()
    elements.append(Paragraph("District Development AI Report", styles["Title"]))
    elements.append(Spacer(1, 0.5 * inch))
    elements.append(Paragraph(f"Category: {category}", styles["Normal"]))
    elements.append(Paragraph(f"District: {district}", styles["Normal"]))
    elements.append(Paragraph(f"Predicted Value ({future_year}): {prediction:.2f}", styles["Normal"]))
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph(policy_text, styles["Normal"]))

    doc.build(elements)

    with open(temp_file.name, "rb") as f:
        st.download_button(
            label="📄 Download Full Report as PDF",
            data=f,
            file_name="AI_Development_Report.pdf",
            mime="application/pdf"
        )


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os


st.set_page_config(
    page_title="🏛️ AI Socio-Economic Policy System",
    page_icon="📊",
    layout="wide"
)


INDICATORS = {
    "GDP": {
        "file": "data/gdp_Haryana.csv",
        "target": " Yamuna Nagar",
        "unit": "₹ Crores",
        "policy": {
            "low": "Industrial Expansion & Employment Boost",
            "mid": "Infrastructure & Digital Growth",
            "high": "Innovation & Sustainable Development"
        }
    },
    "Income": {
        "file": "data/income_Haryana.csv",
        "target": " Yamuna Nagar",
        "unit": "₹ Per Capita",
        "policy": {
            "low": "Skill Development & Wage Support",
            "mid": "SME & Startup Incentives",
            "high": "Tax Optimization & Investment Growth"
        }
    },
    "Literacy": {
        "file": "data/literacy_Haryana.csv",
        "target": " Yamuna Nagar",
        "unit": "%",
        "policy": {
            "low": "School Expansion & Scholarship Programs",
            "mid": "Digital Learning Initiatives",
            "high": "Higher Education & Research Funding"
        }
    },
    "Employment": {
        "file": "data/employment_Haryana.csv",
        "target": " Yamuna Nagar",
        "unit": "%",
        "policy": {
            "low": "Job Creation Schemes",
            "mid": "Skill-Based Training Programs",
            "high": "Industry-Academia Collaboration"
        }
    }
}


st.sidebar.header("⚙️ Government Control Panel")

selected_indicator = st.sidebar.selectbox(
    "Select Socio-Economic Indicator",
    list(INDICATORS.keys())
)

indicator_config = INDICATORS[selected_indicator]
DATA_FILE = indicator_config["file"]
TARGET = indicator_config["target"]

MODEL_FILE = f"model_{selected_indicator.lower()}.pkl"


@st.cache_data
def load_data(file):
    return pd.read_csv(file)

df = load_data(DATA_FILE)
df_clean = df.copy()


FEATURES = df.columns[2:].tolist()


encoder = LabelEncoder()

if "Year" in df_clean.columns:
    df_clean["Year"] = encoder.fit_transform(df_clean["Year"])

if "Description" in df_clean.columns:
    df_clean["Description"] = encoder.fit_transform(df_clean["Description"])

df_clean.fillna(df_clean.mean(numeric_only=True), inplace=True)


def train_model(df, features, target):
    X = df[features]
    y = df[target]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_scaled, y)

    joblib.dump({
        "model": model,
        "features": features,
        "scaler": scaler
    }, MODEL_FILE)

    return model, scaler

if os.path.exists(MODEL_FILE):
    bundle = joblib.load(MODEL_FILE)
    model = bundle["model"]
    FEATURES = bundle["features"]
    scaler = bundle["scaler"]
else:
    st.info("🤖 Training AI model for this indicator...")
    model, scaler = train_model(df_clean, FEATURES, TARGET)
    st.success("✅ Model trained and saved!")


st.sidebar.header("📍 District Selection")
districts = df.columns[2:]
selected_district = st.sidebar.selectbox(
    "Choose District",
    districts
)


col1, col2 = st.columns(2)


with col1:
    st.subheader(f"📋 {selected_indicator} History — {selected_district}")
    st.dataframe(
        df[["Year", selected_district]].tail(10),
        use_container_width=True
    )



latest_values = df.iloc[-1][2:].astype(float)
district_value = float(df[selected_district].iloc[-1])

p33 = np.percentile(latest_values, 33)
p66 = np.percentile(latest_values, 66)

if district_value <= p33:
    stage = "🟥 Low"
    color = "red"
elif district_value <= p66:
    stage = "🟨 Medium"
    color = "orange"
else:
    stage = "🟩 High"
    color = "green"

with col2:
    st.subheader("🏗️ Development Status")
    st.markdown(
        f"""
        <h2 style='color:{color};'>{stage}</h2>
        <p><b>Latest Value:</b> {district_value:.2f} {indicator_config['unit']}</p>
        """,
        unsafe_allow_html=True
    )


st.subheader("📈 Trend Over Years")

fig, ax = plt.subplots()
ax.plot(df["Year"], df[selected_district], marker="o")
ax.set_xlabel("Year")
ax.set_ylabel(f"{selected_indicator} Value")
ax.set_title(f"{selected_indicator} Trend — {selected_district}")
plt.xticks(rotation=45)
st.pyplot(fig)


st.subheader("📊 District Comparison (Latest Year)")

latest_data = df.iloc[-1][2:].astype(float)

fig2, ax2 = plt.subplots(figsize=(10, 5))
latest_data.sort_values().plot(kind="bar", ax=ax2)
ax2.set_ylabel(f"{selected_indicator} Value")
ax2.set_title(f"Latest {selected_indicator} Across Districts")
st.pyplot(fig2)


st.subheader("🔮 AI Future Prediction")

prediction = None

if st.button("Predict Future Value"):
    X_latest = df_clean[FEATURES].iloc[-1].values.reshape(1, -1)
    X_latest_scaled = scaler.transform(X_latest)

    prediction = model.predict(X_latest_scaled)[0]

    st.success(
        f"📈 Predicted Future {selected_indicator} for {selected_district}: "
        f"{prediction:.2f} {indicator_config['unit']}"
    )


st.subheader("🧠 AI Policy Insights")

growth = df[selected_district].pct_change().mean() * 100
policy = indicator_config["policy"]

recommendation = (
    policy["low"] if stage == "🟥 Low"
    else policy["mid"] if stage == "🟨 Medium"
    else policy["high"]
)

insight = f"""
### 🏛️ Policy Intelligence Report — {selected_indicator}

**District:** {selected_district}  
**Current Stage:** {stage.replace("🟥","").replace("🟨","").replace("🟩","")}  
**Average Growth Rate:** {growth:.2f}%  
**Future Trend:** {"Positive Growth Expected" if prediction and prediction > district_value else "Stable / Needs Strategic Focus"}

---

### 📌 AI Policy Recommendation
> **{recommendation}**

### 🧠 Strategic Government Actions
- Identify underperforming districts using AI ranking
- Allocate funds using predictive risk modeling
- Launch district-specific policy interventions
- Monitor policy success using next-year AI forecast
"""

st.markdown(insight)


st.markdown("---")
st.caption("🚀 AI-Powered Socio-Economic Policy System | Machine Learning | Streamlit | Random Forest")

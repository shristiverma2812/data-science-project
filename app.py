import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Load dataset
advertising = pd.read_csv("Advertising.csv")

# Load trained model
with open("advertising_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Advertising Sales Prediction Dashboard")

# Sidebar theme selector with slider style switch
st.sidebar.header("Theme Settings")
mode = st.sidebar.toggle("🌞 Light / 🌙 Dark Mode", value=True)  # True = Light, False = Dark

# Background + text color + emoji
if mode:  # Light Mode
    background = "#FDFDFD"
    text_color = "#000000"
    emoji = "🌞"
else:  # Dark Mode
    background = "#121212"
    text_color = "#FFFFFF"
    emoji = "🌙"

# Inject CSS with animation + conditional text color
st.markdown(
    f"""
    <style>
    .stApp {{
        background: {background};
        color: {text_color};
        transition: background 1s ease, color 1s ease;
    }}
    /* Apply text color to all key containers */
    .stMetric, .stDataFrame, .stTable, .stMarkdown, .stPlotlyChart, .stPyplot, .stTabs [role="tab"] {{
        color: {text_color} !important;
        transition: color 0.8s ease;
    }}
    /* Metric specific text */
    div[data-testid="stMetricValue"] {{
        color: {text_color} !important;
        transition: color 0.8s ease;
    }}
    div[data-testid="stMetricLabel"] {{
        color: {text_color} !important;
        transition: color 0.8s ease;
    }}
    /* Box styling */
    .stMetric, .stDataFrame, .stTable, .stMarkdown, .stPlotlyChart, .stPyplot {{
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px;
        padding: 10px;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.6s ease;
    }}
    .stMetric:hover, .stDataFrame:hover, .stTable:hover, .stMarkdown:hover, 
    .stPlotlyChart:hover, .stPyplot:hover {{
        box-shadow: 0 0 25px rgba(255, 255, 255, 0.6);
        transform: scale(1.02);
    }}
    h1, h2, h3, h4 {{
        background: linear-gradient(90deg, #ff6a00, #ee0979);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        transition: all 1s ease;
    }}
    div[data-testid="stEmpty"] {{
        display: none;
    }}
    /* Emoji rotation animation */
    .emoji {{
        font-size: 40px;
        display: inline-block;
        transition: transform 0.8s ease, opacity 0.8s ease;
    }}
    .emoji.rotate {{
        transform: rotateY(180deg);
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Show emoji with rotation animation
st.markdown(f"<div class='emoji rotate'>{emoji}</div>", unsafe_allow_html=True)

# Sidebar Inputs with icons
st.sidebar.header("Enter Advertising Budget")
tv = st.sidebar.number_input("📺 TV Budget ($)", 0.0, 500.0, 100.0)
radio = st.sidebar.number_input("📻 Radio Budget ($)", 0.0, 50.0, 20.0)
newspaper = st.sidebar.number_input("📰 Newspaper Budget ($)", 0.0, 100.0, 30.0)

input_df = pd.DataFrame({
    'TV': [tv],
    'Radio': [radio],
    'Newspaper': [newspaper]
})

if st.sidebar.button("Predict Sales"):
    prediction = model.predict(input_df)[0]
    st.success(f"📊 Predicted Sales: {prediction:.2f} units")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Coefficients", 
    "📂 Dataset Info", 
    "📉 Correlation Heatmap",
    "📈 Scatter Plots",
    "⚙️ Model Performance"
])

with tab1:
    st.header("Regression Coefficients")
    coef_df = pd.DataFrame({
        'Feature': ['TV', 'Radio', 'Newspaper'],
        'Coefficient': model.coef_
    })

    def color_coeff(val):
        color = 'green' if val > 0 else 'red' if val < 0 else 'blue'
        return f'color: {color}; font-weight: bold;'

    st.write(coef_df.style.applymap(color_coeff, subset=['Coefficient']))

    intercept_value = model.intercept_
    if intercept_value > 0:
        intercept_color = "#28a745"
    elif intercept_value < 0:
        intercept_color = "#dc3545"
    else:
        intercept_color = "#007bff"

    st.markdown(
        f"<h4 style='color:{intercept_color}; transition: color 1s ease;'>Intercept: {intercept_value:.2f}</h4>",
        unsafe_allow_html=True
    )

with tab2:
    st.header("Dataset Overview")
    st.write(advertising.head(10))
    st.write(f"Dataset Shape: {advertising.shape}")

with tab3:
    st.header("Correlation Heatmap")
    corr = advertising.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

with tab4:
    st.header("Scatter Plots")
    feature = st.selectbox("Choose feature", ['TV','Radio','Newspaper'])
    fig, ax = plt.subplots()
    sns.scatterplot(x=advertising[feature], y=advertising['Sales'], ax=ax)
    st.pyplot(fig)

with tab5:
    st.header("Model Performance")
    X = advertising[['TV','Radio','Newspaper']]
    y = advertising['Sales']
    y_pred = model.predict(X)
    score = r2_score(y, y_pred)

    st.metric("R² Score", f"{score:.3f}")

    residuals = y - y_pred
    fig, ax = plt.subplots()
    sns.histplot(residuals, kde=True, ax=ax, color="cyan")
    ax.set_facecolor("#111111" if not mode else "#FFFFFF")
    ax.set_title("Residuals Distribution", color="white" if not mode else "black")
    ax.set_xlabel("Error", color="white" if not mode else "black")
    ax.set_ylabel("Frequency", color="white" if not mode else "black")
    st.pyplot(fig)
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# 1. Konfigurasi Halaman Dashboard (Tema Dark Mode)
st.set_page_config(
    page_title="Executive BI Dashboard & Predictive Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS untuk gaya Dark Mode yang modern
st.markdown(
    """
    <style>
    .main { background-color: #0f172a; }
    div.stButton > button { width: 100%; border-radius: 8px; }
    .kpi-card {
        background-color: #1e293b;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #334155;
        text-align: center;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# 2. Fungsi Load Data (Menggunakan latihan1.csv / Data Dummy)
@st.cache_data
def load_data():
    csv_path = "latihan1.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        np.random.seed(42)
        n = 100
        genders = np.random.choice(["F", "M"], size=n, p=[0.55, 0.45])
        ages = np.random.randint(18, 25, size=n)
        study_hours = np.round(np.random.uniform(1.0, 10.0, size=n), 1)
        sleep_hours = np.round(np.random.uniform(4.0, 9.0, size=n), 1)
        attendance = np.round(np.random.uniform(60.0, 100.0, size=n), 1)
        monthly_spending = np.round(np.random.uniform(500, 3500, size=n), 0)
        screen_hours = np.round(np.random.uniform(2.0, 12.0, size=n), 1)
        satisfaction = np.random.randint(1, 6, size=n)

        noise = np.random.normal(0, 3.5, size=n)
        exam_score = (
            25.0
            + (4.2 * study_hours)
            + (0.35 * attendance)
            + (1.1 * sleep_hours)
            - (0.8 * screen_hours)
            + noise
        )
        exam_score = np.round(np.clip(exam_score, 0, 100), 1)

        df = pd.DataFrame({
            "Student_ID": [f"STD-{1001 + i}" for i in range(n)],
            "Gender": genders,
            "Age": ages,
            "Study_Hours": study_hours,
            "Sleep_Hours": sleep_hours,
            "Attendance": attendance,
            "Exam_Score": exam_score,
            "Monthly_Spending": monthly_spending,
            "Screen_Hours": screen_hours,
            "Satisfaction": satisfaction,
        })
        df.to_csv(csv_path, index=False)
    return df


df_raw = load_data()

# 3. Sidebar (Filter Gender)
st.sidebar.title("🔍 Filter Dashboard")
gender_filter = st.sidebar.selectbox(
    "Pilih Gender:", ["Semua", "Female (F)", "Male (M)"]
)

if gender_filter == "Female (F)":
    df = df_raw[df_raw["Gender"] == "F"].copy()
elif gender_filter == "Male (M)":
    df = df_raw[df_raw["Gender"] == "M"].copy()
else:
    df = df_raw.copy()

# Header Dashboard
st.title("📊 Executive BI Dashboard & Predictive Analytics")
st.caption(
    "Sistem Pengambilan Keputusan Evaluasi Performa Akademik & Gaya Hidup"
    " Siswa"
)

# 4. Ringkasan Eksekutif (Banner)
top_corr_val = df[["Exam_Score", "Study_Hours"]].corr().iloc[0, 1]
at_risk_count = (df["Exam_Score"] < 60).sum()

st.info(
    f"💡 **Ringkasan Eksekutif:** Berdasarkan data **{len(df)} siswa**, faktor"
    " yang paling berpengaruh positif terhadap **Exam Score** adalah"
    f" **Study_Hours** (korelasi: +{top_corr_val:.2f}). Terdapat **{at_risk_count}"
    " siswa** yang butuh intervensi khusus (skor < 60)."
)

# 5. KPI Cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("Rata-rata Exam Score", f"{df['Exam_Score'].mean():.1f}")
col2.metric("Rata-rata Jam Belajar", f"{df['Study_Hours'].mean():.1f} hrs")
col3.metric("Rata-rata Kehadiran", f"{df['Attendance'].mean():.1f}%")
col4.metric(
    "Siswa Risk (< 60)",
    f"{at_risk_count} Siswa",
    delta=f"{(at_risk_count/len(df))*100:.1f}%",
    delta_color="inverse",
)

st.markdown("---")

# 6. Modul 1: What-If Simulator & Radar Chart Profile
mod1_col1, mod1_col2 = st.columns([1, 1])

with mod1_col1:
    st.subheader("🎛️ What-If Simulator Prediksi Exam Score")

    sim_study = st.slider("Jam Belajar (Study Hours):", 1.0, 10.0, 5.0, 0.1)
    sim_att = st.slider("Kehadiran (Attendance %):", 60, 100, 80, 1)
    sim_sleep = st.slider("Jam Tidur (Sleep Hours):", 4.0, 9.0, 7.0, 0.1)
    sim_screen = st.slider("Screen Time (Screen Hours):", 2.0, 12.0, 5.0, 0.1)

    # Kalkulasi OLS
    pred_score = (
        25.0
        + (4.20 * sim_study)
        + (0.35 * sim_att)
        + (1.10 * sim_sleep)
        - (0.80 * sim_screen)
    )
    pred_score = min(100, max(0, pred_score))

    grade = (
        "A (Sangat Memuaskan)"
        if pred_score >= 85
        else (
            "B (Baik)"
            if pred_score >= 75
            else (
                "C (Cukup)"
                if pred_score >= 65
                else "D/F (Butuh Perbaikan)"
            )
        )
    )

    st.markdown(
        f"""
    <div class="kpi-card">
        <h3 style="color: #94a3b8; margin:0;">ESTIMASI EXAM SCORE</h3>
        <h1 style="color: #34d399; font-size: 3rem; margin:0;">{pred_score:.1f}</h1>
        <p style="color: #fbbf24; margin:0;"><b>Grade: {grade}</b></p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    with st.expander("💻 Behind the Scenes (Sistem Persamaan OLS)"):
        st.code(
            "Exam_Score = 25.0 + (4.20 * Study) + (0.35 * Attendance) + (1.10 *"
            " Sleep) - (0.80 * Screen)"
        )
        st.write(
            f"Perhitungan: 25.0 + ({4.20*sim_study:.2f}) +"
            f" ({0.35*sim_att:.2f}) + ({1.10*sim_sleep:.2f}) -"
            f" ({0.80*sim_screen:.2f}) = **{pred_score:.2f}**"
        )

with mod1_col2:
    st.subheader("🕸️ Profil Multi-Aksis (Female vs Male)")
    radar_cols = [
        "Study_Hours",
        "Sleep_Hours",
        "Attendance",
        "Screen_Hours",
        "Satisfaction",
    ]
    f_means = df_raw[df_raw["Gender"] == "F"][radar_cols].mean().tolist()
    m_means = df_raw[df_raw["Gender"] == "M"][radar_cols].mean().tolist()

    fig_radar = go.Figure()
    fig_radar.add_trace(
        go.Scatterpolar(
            r=f_means,
            theta=radar_cols,
            fill="toself",
            name="Female (F)",
            line_color="#38bdf8",
        )
    )
    fig_radar.add_trace(
        go.Scatterpolar(
            r=m_means,
            theta=radar_cols,
            fill="toself",
            name="Male (M)",
            line_color="#818cf8",
        )
    )
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, showticklabels=False)),
        template="plotly_dark",
        height=380,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

st.markdown("---")

# 7. Modul 2: Ranking Korelasi & Scatter Chart
mod2_col1, mod2_col2 = st.columns(2)

numeric_cols = [
    "Age",
    "Study_Hours",
    "Sleep_Hours",
    "Attendance",
    "Exam_Score",
    "Monthly_Spending",
    "Screen_Hours",
    "Satisfaction",
]

with mod2_col1:
    st.subheader("📊 Ranking Pengaruh Faktor terhadap Target")
    target_var = st.selectbox("Pilih Target Variable (Y):", ["Exam_Score", "Satisfaction"])
    
    corr_series = df[numeric_cols].corr()[target_var].drop(target_var).sort_values()
    
    fig_bar = px.bar(
        x=corr_series.values,
        y=corr_series.index,
        orientation='h',
        color=corr_series.values,
        color_continuous_scale=['#f87171', '#34d399'],
        labels={'x': 'Nilai Korelasi', 'y': 'Faktor (X)'}
    )
    fig_bar.update_layout(template="plotly_dark", height=320, coloraxis_showscale=False)
    st.plotly_chart(fig_bar, use_container_width=True)

with mod2_col2:
    st.subheader("📈 Scatter Detail: Target vs Faktor")
    factor_var = st.selectbox("Pilih Faktor (X):", [c for c in numeric_cols if c != target_var])
    
    fig_scatter = px.scatter(
        df,
        x=factor_var,
        y=target_var,
        color="Gender",
        trendline="ols",
        template="plotly_dark",
        color_discrete_map={'F': '#38bdf8', 'M': '#818cf8'}
    )
    fig_scatter.update_layout(height=320)
    st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

# 8. Modul 3: Matriks Korelasi & Data Table Interaktif
st.subheader("🔥 Matriks Korelasi Lengkap")
corr_matrix = df[numeric_cols].corr().round(2)
fig_heatmap = px.imshow(
    corr_matrix,
    text_auto=True,
    color_continuous_scale='RdBu_r',
    template="plotly_dark",
    aspect="auto"
)
fig_heatmap.update_layout(height=400)
st.plotly_chart(fig_heatmap, use_container_width=True)

st.subheader("📑 Data Student Records")
st.dataframe(df, use_container_width=True)
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Config Halaman
st.set_page_config(
    page_title="Executive BI Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load Data
@st.cache_data
def load_data():
    csv_path = 'latihan1.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            'Student_ID': [f'STD-{1001 + i}' for i in range(n)],
            'Gender': np.random.choice(['F', 'M'], size=n),
            'Age': np.random.randint(18, 25, size=n),
            'Study_Hours': np.round(np.random.uniform(1.0, 10.0, size=n), 1),
            'Sleep_Hours': np.round(np.random.uniform(4.0, 9.0, size=n), 1),
            'Attendance': np.round(np.random.uniform(60.0, 100.0, size=n), 1),
            'Exam_Score': np.round(np.random.uniform(50.0, 100.0, size=n), 1),
            'Monthly_Spending': np.round(np.random.uniform(500, 3500, size=n), 0),
            'Screen_Hours': np.round(np.random.uniform(2.0, 12.0, size=n), 1),
            'Satisfaction': np.random.randint(1, 6, size=n)
        })
    return df

df = load_data()

# Layout Dashboard
st.title("📊 Executive BI Dashboard & Predictive Analytics")

# KPI Cards
col1, col2, col3 = st.columns(3)
col1.metric("Rata-rata Nilai Ujian", f"{df['Exam_Score'].mean():.1f}")
col2.metric("Rata-rata Jam Belajar", f"{df['Study_Hours'].mean():.1f} Jam")
col3.metric("Rata-rata Kehadiran", f"{df['Attendance'].mean():.1f}%")

st.markdown("---")

# Visualisasi
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Hubungan Jam Belajar vs Nilai Ujian")
    fig_scatter = px.scatter(df, x="Study_Hours", y="Exam_Score", color="Gender", template="plotly_dark")
    st.plotly_chart(fig_scatter, use_container_width=True)

with col_right:
    st.subheader("Distribusi Nilai Ujian")
    fig_hist = px.histogram(df, x="Exam_Score", nbins=15, template="plotly_dark")
    st.plotly_chart(fig_hist, use_container_width=True)

st.subheader("📑 Data Student Records")
st.dataframe(df, use_container_width=True)
import os
import webbrowser
from threading import Timer
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np

app = Flask(__name__)

DATA_FILE = 'latihan1.csv'

def load_data():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
    else:
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            'Student_ID': range(1, n + 1),
            'Gender': np.random.choice(['F', 'M'], size=n),
            'Age': np.random.randint(18, 23, size=n),
            'Study_Hours': np.random.randint(4, 20, size=n),
            'Sleep_Hours': np.round(np.random.uniform(5.0, 9.0, size=n), 1),
            'Attendance': np.random.randint(65, 100, size=n),
            'Exam_Score': np.random.randint(45, 98, size=n),
            'Monthly_Spending': np.random.randint(500, 2000, size=n),
            'Screen_Hours': np.round(np.random.uniform(2.0, 10.0, size=n), 1),
            'Satisfaction': np.random.randint(3, 10, size=n)
        })
    return df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data', methods=['GET'])
def get_data():
    df = load_data()
    
    gender_filter = request.args.get('gender', 'Semua')
    if gender_filter in ['F', 'M']:
        df_filtered = df[df['Gender'] == gender_filter].copy()
    else:
        df_filtered = df.copy()

    numeric_cols = ['Age', 'Study_Hours', 'Sleep_Hours', 'Attendance', 
                    'Exam_Score', 'Monthly_Spending', 'Screen_Hours', 'Satisfaction']
    
    stats = {}
    for col in numeric_cols:
        stats[col] = {
            'mean': round(float(df_filtered[col].mean()), 2),
            'median': round(float(df_filtered[col].median()), 2),
            'min': round(float(df_filtered[col].min()), 2),
            'max': round(float(df_filtered[col].max()), 2),
            'std': round(float(df_filtered[col].std()), 2)
        }

    corr_df = df_filtered[numeric_cols].corr().round(3)
    corr_matrix = corr_df.to_dict()

    feature_cols = ['Study_Hours', 'Attendance', 'Sleep_Hours', 'Screen_Hours']
    X = df_filtered[feature_cols].values
    y = df_filtered['Exam_Score'].values
    
    X_design = np.hstack([np.ones((X.shape[0], 1)), X])
    
    try:
        weights_array = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ y
        weights = {
            'intercept': round(float(weights_array[0]), 4),
            'Study_Hours': round(float(weights_array[1]), 4),
            'Attendance': round(float(weights_array[2]), 4),
            'Sleep_Hours': round(float(weights_array[3]), 4),
            'Screen_Hours': round(float(weights_array[4]), 4)
        }
    except Exception:
        weights = {
            'intercept': 12.8094,
            'Study_Hours': 0.5686,
            'Attendance': 0.5837,
            'Sleep_Hours': 0.1760,
            'Screen_Hours': -2.9483
        }

    radar_cols = ['Study_Hours', 'Sleep_Hours', 'Attendance', 'Screen_Hours', 'Satisfaction']
    radar_data = {
        'F': df[df['Gender'] == 'F'][radar_cols].mean().round(2).to_dict(),
        'M': df[df['Gender'] == 'M'][radar_cols].mean().round(2).to_dict()
    }

    at_risk_count = int((df_filtered['Exam_Score'] < 65).sum())
    total_students = len(df_filtered)

    return jsonify({
        'raw_data': df_filtered.to_dict(orient='records'),
        'stats': stats,
        'corr_matrix': corr_matrix,
        'numeric_cols': numeric_cols,
        'weights': weights,
        'radar_data': radar_data,
        'at_risk_count': at_risk_count,
        'total_students': total_students
    })

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

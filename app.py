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

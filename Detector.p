# Dynamic Energy Usage Pattern Detector in Google Colab

# ============================
# STEP 1: INSTALL REQUIRED PACKAGES
# ============================
!pip install pandas numpy matplotlib seaborn plotly scikit-learn gradio twilio -q
!pip install -U kaleido -q

# ============================
# STEP 2: IMPORT LIBRARIES
# ============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import gradio as gr
from twilio.rest import Client

# ============================
# STEP 3: UPLOAD AND LOAD DATA
# ============================
from google.colab import files
uploaded = files.upload()

# Assuming the file uploaded is 'energy_data_sample.csv'
df = pd.read_csv("energy_data_sample.csv", parse_dates=["datetime"])
df.set_index("datetime", inplace=True)

# ============================
# STEP 4: PREPROCESSING
# ============================
# Keep only the global active power column
energy = df[["Global_active_power"]].copy()
energy = energy.resample('H').mean().fillna(method='ffill')

# Normalize the data
scaler = StandardScaler()
energy_scaled = scaler.fit_transform(energy)

# ============================
# STEP 5: ANOMALY DETECTION USING ISOLATION FOREST
# ============================
model = IsolationForest(contamination=0.01, random_state=42)
energy['anomaly'] = model.fit_predict(energy_scaled)
energy['anomaly'] = energy['anomaly'].map({1: 0, -1: 1})

# ============================
# STEP 6: VISUALIZATION
# ============================
fig = px.line(energy, y='Global_active_power', title='Hourly Energy Usage')
fig.add_scatter(x=energy[energy['anomaly']==1].index,
                y=energy[energy['anomaly']==1]['Global_active_power'],
                mode='markers', name='Anomaly')
fig.show()

# ============================
# STEP 7: ALERT SYSTEM (TWILIO)
# ============================
def send_alert(message):
    account_sid = 'your_account_sid_here'
    auth_token = 'your_auth_token_here'
    client = Client(account_sid, auth_token)
    client.messages.create(
        body=message,
        from_='+1234567890',  # Twilio number
        to='+1987654321'     # Your verified number
    )

# Example trigger
if energy['anomaly'].iloc[-1] == 1:
    send_alert("⚠️ High energy usage anomaly detected!")

# ============================
# STEP 8: GRADIO DASHBOARD
# ============================
def detect_anomaly(hours):
    future_time = energy.tail(hours)
    trace = go.Scatter(x=future_time.index, y=future_time['Global_active_power'], mode='lines', name='Usage')
    fig = go.Figure(data=[trace])
    fig.update_layout(title=f"Last {hours} Hours of Energy Usage")
    return fig

demo = gr.Interface(
    fn=detect_anomaly,
    inputs=gr.Slider(1, 48, step=1, label="Hours to Visualize"),
    outputs=gr.Plot(label="Energy Usage Graph"),
    title="Energy Usage Pattern Dashboard"
)

demo.launch()

# ============================
# END OF SYSTEM
# ============================
# This notebook performs anomaly detection on smart meter data,
# provides visualization, Twilio alerting, and a user-friendly Gradio UI.

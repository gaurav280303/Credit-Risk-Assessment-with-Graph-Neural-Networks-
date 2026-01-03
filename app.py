import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import plotly.graph_objects as go
import plotly.express as px
import io

# -----------------------------------------------------------------------------
# 1. APP CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Financial Guard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp { background: radial-gradient(circle at 10% 20%, rgb(15, 32, 39) 0%, rgb(32, 58, 67) 90.2%); color: white; }
    .css-1r6slb0, .stDataFrame, .stPlotlyChart { background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 15px; }
    h1 { background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .stButton>button { background: linear-gradient(90deg, #00C9FF, #92FE9D); color: #0f2027; font-weight: bold; border: none; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. MODEL CLASSES (Must match training)
# -----------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim=95):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 2)
        )
    def forward(self, x): return self.network(x)

class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(10, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class RNNModel(nn.Module):
    def __init__(self, model_type='lstm'):
        super(RNNModel, self).__init__()
        self.rnn = nn.LSTM(10, 128, 2, batch_first=True, dropout=0.3) if model_type=='lstm' else nn.GRU(10, 128, 2, batch_first=True, dropout=0.3)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = F.relu(self.fc1(out[:, -1, :]))
        return self.fc2(out)

# -----------------------------------------------------------------------------
# 3. LOADING & SESSION STATE
# -----------------------------------------------------------------------------

# Initialize Session State to hold data across reruns
if 'input_df' not in st.session_state:
    st.session_state.input_df = None

@st.cache_resource
def load_all():
    assets = {}
    models = {}
    
    # Load Assets
    try:
        with open('feature_names.pkl', 'rb') as f: assets['features'] = pickle.load(f)
    except: assets['features'] = [f"Col_{i}" for i in range(95)]
    
    try: with open('scaler.pkl', 'rb') as f: assets['scaler'] = pickle.load(f)
    except: pass
    
    try: with open('ensemble_weights.pkl', 'rb') as f: assets['weights'] = pickle.load(f)
    except: assets['weights'] = {}

    # Load Models
    device = torch.device('cpu')
    try: models['lr'] = pickle.load(open('logistic_regression.pkl', 'rb'))
    except: pass
    try: models['rf'] = pickle.load(open('random_forest.pkl', 'rb'))
    except: pass
    try: models['gb'] = pickle.load(open('gradient_boosting.pkl', 'rb'))
    except: pass
    try: models['xgb'] = pickle.load(open('xgboost.pkl', 'rb'))
    except: pass
    
    try: 
        m = MLP(95); m.load_state_dict(torch.load('mlp_model.pth', map_location=device)); m.eval(); models['mlp'] = m
    except: pass
    
    try:
        m = CNN1D(); m.load_state_dict(torch.load('cnn_model.pth', map_location=device), strict=False); m.eval(); models['cnn'] = m
    except: pass
    
    try:
        m = RNNModel('lstm'); m.load_state_dict(torch.load('lstm_model.pth', map_location=device), strict=False); m.eval(); models['lstm'] = m
    except: pass
    
    return assets, models

assets, models = load_all()

# -----------------------------------------------------------------------------
# 4. SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("Financial Guard AI")
    st.info("The Industry Standard for Bankruptcy Prediction.")
    st.markdown("### 📥 Need Test Data?")
    
    # Create a template CSV for download
    if assets['features']:
        template_df = pd.DataFrame(columns=assets['features'])
        csv = template_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV Template",
            data=csv,
            file_name="financial_data_template.csv",
            mime="text/csv",
            help="Download this blank file, fill in your data, and upload it."
        )

# -----------------------------------------------------------------------------
# 5. MAIN INTERFACE
# -----------------------------------------------------------------------------
st.title("🛡️ Financial Guard AI")

# TABS
tab1, tab2 = st.tabs(["🎲 Quick Demo (No File Needed)", "📂 Upload File"])

with tab1:
    st.write("### Simulate a Company")
    col_d1, col_d2 = st.columns(2)
    
    # Button logic updates Session State directly
    if col_d1.button("🟢 Generate Healthy Company"):
        # Create random healthy data
        data = np.random.rand(1, 95) * 0.5 + 0.5 # Bias towards higher values (usually healthy)
        st.session_state.input_df = pd.DataFrame(data, columns=assets['features'])
        st.success("Generated Healthy Data! Scroll down to Analyze.")

    if col_d2.button("🔴 Generate Distressed Company"):
        # Create random distressed data
        data = np.random.rand(1, 95) * 0.4 # Bias towards lower values (usually risky)
        st.session_state.input_df = pd.DataFrame(data, columns=assets['features'])
        st.warning("Generated Distressed Data! Scroll down to Analyze.")

with tab2:
    st.write("### Analyze Real Data")
    uploaded_file = st.file_uploader("Upload CSV with 95 Financial Features", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        # Validate columns
        if df.shape[1] < 95:
            st.error(f"Error: File has {df.shape[1]} columns, but model requires 95.")
        else:
            if 'Bankrupt?' in df.columns: df = df.drop('Bankrupt?', axis=1)
            st.session_state.input_df = df
            st.success("File Uploaded Successfully.")

# -----------------------------------------------------------------------------
# 6. ANALYSIS ENGINE
# -----------------------------------------------------------------------------
if st.session_state.input_df is not None:
    st.markdown("---")
    st.subheader("📊 Analysis Control Panel")
    
    with st.expander("View Input Data"):
        st.dataframe(st.session_state.input_df)

    if st.button("🚀 Run Risk Assessment Protocol", type="primary"):
        with st.spinner("Running Hybrid Ensemble Inference..."):
            
            # 1. Preprocess
            input_row = st.session_state.input_df.iloc[[0]] # Take first row
            
            if 'scaler' in assets:
                try: X_scaled = assets['scaler'].transform(input_row).astype(np.float32)
                except: X_scaled = input_row.values.astype(np.float32)
            else:
                X_scaled = input_row.values.astype(np.float32)
                
            # Prepare Tensors
            X_tensor = torch.tensor(X_scaled)
            
            # Reshape for DL (Need 90 features for 9x10 grid)
            X_90 = X_scaled[:, :90]
            X_rnn = torch.tensor(X_90.reshape(1, 9, 10)).float()
            X_cnn = X_rnn.transpose(1, 2) # [1, 10, 9]

            # 2. Predict
            probs = {}
            
            # Sklearn
            if 'xgb' in models: probs['XGBoost'] = models['xgb'].predict_proba(X_scaled)[:, 1][0]
            if 'rf' in models: probs['Random Forest'] = models['rf'].predict_proba(X_scaled)[:, 1][0]
            
            # Deep Learning
            with torch.no_grad():
                if 'mlp' in models: probs['Deep MLP'] = F.softmax(models['mlp'](X_tensor), dim=1)[:, 1].item()
                if 'cnn' in models: probs['1D-CNN'] = F.softmax(models['cnn'](X_cnn), dim=1)[:, 1].item()
                if 'lstm' in models: probs['LSTM'] = F.softmax(models['lstm'](X_rnn), dim=1)[:, 1].item()

            # 3. Weighted Average
            final_score = np.mean(list(probs.values())) # Simple average if weights miss
            
            # 4. Display
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                # Gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number", value = final_score * 100,
                    title = {'text': "Insolvency Risk"},
                    gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': "red" if final_score > 0.5 else "green"}}
                ))
                fig.update_layout(height=300, margin=dict(t=50,b=20,l=20,r=20), paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            
            with col_res2:
                st.markdown("### Risk Verdict")
                if final_score > 0.5:
                    st.error(f"HIGH RISK ({final_score*100:.1f}%) detected.")
                    st.write("The ensemble models have detected significant financial distress signals.")
                else:
                    st.success(f"STABLE ({final_score*100:.1f}%) detected.")
                    st.write("The company appears financially healthy based on provided metrics.")
                
                # Breakdown
                st.write("#### Model Consensus")
                st.bar_chart(pd.Series(probs))

else:
    st.info("waiting for data input...")
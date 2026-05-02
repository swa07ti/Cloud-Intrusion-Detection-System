import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from utils import preprocess_data


st.set_page_config(page_title="Cloud IDS Dashboard", page_icon="🛡️", layout="wide")


st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .main .block-container {
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .normal-text {
        color: #2e7d32;
        font-weight: bold;
    }
    .attack-text {
        color: #c62828;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        with open("rf_model.pkl", "rb") as f:
            rf_model = pickle.load(f)
        with open("svm_model.pkl", "rb") as f:
            svm_model = pickle.load(f)
        with open("preprocessing.pkl", "rb") as f:
            prep = pickle.load(f)
        with open("metrics.pkl", "rb") as f:
            metrics = pickle.load(f)
        return rf_model, svm_model, prep, metrics
    except Exception as e:
        return None, None, None, None

rf_model, svm_model, prep, training_metrics = load_models()

st.title("🛡️ Cloud Intrusion Detection System")
st.markdown("A professional security dashboard leveraging machine learning to detect and classify network intrusions.")

if rf_model is None:
    st.error("⚠️ Models not found. Please run `python train_model.py` first to generate the required model artifacts.")
    st.stop()


st.sidebar.header("Control Panel")
st.sidebar.markdown("Upload your network traffic log (CSV) to scan for intrusions.")

uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=['csv'])
selected_model = st.sidebar.selectbox("Select Model", ["Random Forest", "SVM", "Compare Both"])

st.sidebar.markdown("---")
st.sidebar.subheader("Training Metrics Overview")
if training_metrics:
    acc_rf = training_metrics['Random Forest']['accuracy']
    acc_svm = training_metrics['SVM']['accuracy']
    st.sidebar.write(f"**RF Accuracy:** {acc_rf:.2%}")
    st.sidebar.write(f"**SVM Accuracy:** {acc_svm:.2%}")


if uploaded_file is not None:
    with st.spinner('Processing data...'):
        df = pd.read_csv(uploaded_file)
        
       
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        try:
            X_scaled, y_encoded = preprocess_data(df, is_training=False, prep_artifacts=prep)
            
            
            models_to_run = []
            if selected_model == "Random Forest" or selected_model == "Compare Both":
                models_to_run.append(("Random Forest", rf_model))
            if selected_model == "SVM" or selected_model == "Compare Both":
                models_to_run.append(("SVM", svm_model))
                
            y_encoder = prep['y_encoder']
            
            st.markdown("---")
            st.header("Results Analysis")
            
            for name, model in models_to_run:
                st.subheader(f"Model: {name}")
                
                preds = model.predict(X_scaled)
                pred_labels = y_encoder.inverse_transform(preds)
                
                
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_scaled)
                    confidences = np.max(probs, axis=1)
                else:
                    confidences = np.ones(len(preds))
                
              
                col1, col2, col3 = st.columns(3)
                
                total_samples = len(pred_labels)
                num_attacks = sum([1 for p in pred_labels if str(p).lower() not in ['normal', '0']])
                num_normal = total_samples - num_attacks
                
                col1.metric("Total Samples", total_samples)
                col2.metric("Normal Traffic", num_normal)
                col3.metric("Detected Intrusions", num_attacks, delta_color="inverse")
                
                if total_samples == 1:
                    status = "Attack" if num_attacks > 0 else "Normal"
                    color = "#c62828" if status == "Attack" else "#2e7d32"
                    st.markdown(f"### Prediction: <span style='color:{color}'>{status}</span> ({pred_labels[0]})", unsafe_allow_html=True)
                    st.write(f"**Confidence:** {confidences[0]:.2%}")
                else:
                 
                    fig = px.pie(values=[num_normal, num_attacks], names=['Normal', 'Attack'], 
                                 title='Traffic Distribution', color_discrete_sequence=['#2e7d32', '#c62828'])
                    st.plotly_chart(fig, use_container_width=True)
                
                
                if y_encoded is not None:
                    st.markdown(f"**Evaluation Metrics against uploaded labels ({name}):**")
                    acc = accuracy_score(y_encoded, preds)
                    prec = precision_score(y_encoded, preds, average='weighted', zero_division=0)
                    rec = recall_score(y_encoded, preds, average='weighted', zero_division=0)
                    f1_val = f1_score(y_encoded, preds, average='weighted', zero_division=0)
                    
                    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                    m_col1.metric("Accuracy", f"{acc:.2%}")
                    m_col2.metric("Precision", f"{prec:.2%}")
                    m_col3.metric("Recall", f"{rec:.2%}")
                    m_col4.metric("F1-Score", f"{f1_val:.2%}")
                    
                 
                    cm = confusion_matrix(y_encoded, preds)
                    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                                     labels=dict(x="Predicted", y="Actual", color="Count"),
                                     x=y_encoder.classes_, y=y_encoder.classes_,
                                     title=f"Confusion Matrix - {name}")
                    st.plotly_chart(fig_cm, use_container_width=True)
                    
            st.markdown("---")
            st.header("Model Insights")
            col_insight1, col_insight2 = st.columns(2)
            
            with col_insight1:
                st.subheader("Accuracy Comparison")
                if training_metrics:
                    acc_df = pd.DataFrame({
                        'Model': ['Random Forest', 'SVM'],
                        'Accuracy': [training_metrics['Random Forest']['accuracy'], training_metrics['SVM']['accuracy']]
                    })
                    fig_acc = px.bar(acc_df, x='Model', y='Accuracy', color='Model', range_y=[0, 1], title="Training Accuracy")
                    st.plotly_chart(fig_acc, use_container_width=True)
            
            with col_insight2:
                if 'Random Forest' in training_metrics and selected_model in ["Random Forest", "Compare Both"]:
                    st.subheader("Feature Importance (Random Forest)")
                    importances = training_metrics['Random Forest']['feature_importances']
                    features = prep['expected_features']
                    
                    if len(features) > 10:
                     
                        feat_df = pd.DataFrame({'Feature': features, 'Importance': importances})
                        feat_df = feat_df.sort_values(by='Importance', ascending=False).head(10)
                    else:
                        feat_df = pd.DataFrame({'Feature': features, 'Importance': importances})
                        feat_df = feat_df.sort_values(by='Importance', ascending=False)
                        
                    fig_feat = px.bar(feat_df, x='Importance', y='Feature', orientation='h', title="Top Features")
                    fig_feat.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_feat, use_container_width=True)
                
        except Exception as e:
            import traceback
            st.error(f"Error processing the data: {str(e)}")
            st.code(traceback.format_exc())
else:
    
    st.info("👈 Please upload a CSV file from the sidebar to begin analysis.")
    
    st.markdown("""
    ### Dashboard Capabilities
    - **Dual Model Inference**: Compare Random Forest and SVM predictions side-by-side.
    - **Live Metrics**: Visualize real-time predictions, accuracy comparisons, and confusion matrices.
    - **Smart Preprocessing**: Automatically handles missing values and encoding out-of-the-box.
    - **Feature Insights**: View top network indicators that contribute to intrusion detection.
    """)
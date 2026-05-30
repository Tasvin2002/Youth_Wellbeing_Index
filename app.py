import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# ----------------------------------------------------------
#  Load artifacts
# ----------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model    = tf.keras.models.load_model('model/ann_optimised_best.keras')
    scaler   = joblib.load('model/scaler.pkl')
    le       = joblib.load('model/label_encoders.pkl')
    features = joblib.load('model/feature_columns.pkl')
    metrics  = joblib.load('model/metrics.pkl')
    metadata = joblib.load('model/metadata.pkl')
    baseline = joblib.load('model/linear_regression_model.pkl')
    return model, scaler, le, features, metrics, metadata, baseline

model, scaler, label_encoders, feature_columns, metrics, metadata, baseline = load_artifacts()

# ----------------------------------------------------------
#  Page config
# ----------------------------------------------------------
st.set_page_config(page_title='Youth Wellbeing Predictor', layout='wide')
st.title('Youth Wellbeing Index Predictor')
st.markdown('Predict a student\'s wellbeing index based on digital behaviour and mental health indicators.')

# ----------------------------------------------------------
#  Sidebar — user inputs
# ----------------------------------------------------------
st.sidebar.header('Student Inputs')

age                     = st.sidebar.slider('Age',                         10, 25, 17)
gender                  = st.sidebar.selectbox('Gender',                   label_encoders['gender'].classes_.tolist())
social_media_hours      = st.sidebar.slider('Social Media Hours/Day',      0.0, 12.0, 3.0, 0.1)
sleep_hours             = st.sidebar.slider('Sleep Hours/Day',             3.0, 12.0, 7.0, 0.1)
brain_rot_index         = st.sidebar.slider('Brain Rot Index',             0.0, 10.0, 5.0, 0.1)
late_night_usage        = st.sidebar.selectbox('Late Night Usage',         label_encoders['late_night_usage'].classes_.tolist())
digital_addiction_score = st.sidebar.slider('Digital Addiction Score',     0.0, 10.0, 5.0, 0.1)
average_session_length  = st.sidebar.slider('Avg Session Length (mins)',   1.0, 180.0, 30.0, 1.0)
sessions_per_day        = st.sidebar.slider('Sessions Per Day',            1, 20, 5)
stress_level            = st.sidebar.slider('Stress Level',                0.0, 10.0, 5.0, 0.1)
anxiety_score           = st.sidebar.slider('Anxiety Score',               0.0, 10.0, 5.0, 0.1)
depression_score        = st.sidebar.slider('Depression Score',            0.0, 10.0, 5.0, 0.1)

# ----------------------------------------------------------
#  Encode categoricals the same way training did
# ----------------------------------------------------------
gender_encoded      = label_encoders['gender'].transform([gender])[0]
late_night_encoded  = label_encoders['late_night_usage'].transform([late_night_usage])[0]

# ----------------------------------------------------------
#  Assemble input in exact feature order
# ----------------------------------------------------------
input_dict = {
    'age':                            age,
    'gender':                         gender_encoded,
    'social_media_hours':             social_media_hours,
    'sleep_hours':                    sleep_hours,
    'brain_rot_index':                brain_rot_index,
    'late_night_usage':               late_night_encoded,
    'digital_addiction_score':        digital_addiction_score,
    'average_session_length_minutes': average_session_length,
    'sessions_per_day':               sessions_per_day,
    'stress_level':                   stress_level,
    'anxiety_score':                  anxiety_score,
    'depression_score':               depression_score,
}

input_array  = np.array([[input_dict[f] for f in feature_columns]])
input_scaled = scaler.transform(input_array)

# ----------------------------------------------------------
#  Predictions from both models
# ----------------------------------------------------------
pred_ann = model.predict(input_scaled, verbose=0).flatten()[0]
pred_lr  = baseline.predict(input_scaled)[0]

# ----------------------------------------------------------
#  Display predictions
# ----------------------------------------------------------
st.subheader('Prediction Results')
col1, col2 = st.columns(2)
col1.metric('Optimised ANN Prediction',   f'{pred_ann:.2f}')
col2.metric('Linear Regression Baseline', f'{pred_lr:.2f}')

# ----------------------------------------------------------
#  Model comparison metrics
# ----------------------------------------------------------
st.subheader('Model Performance (Test Set)')
col1, col2, col3 = st.columns(3)

for col, (label, key) in zip(
    [col1, col2, col3],
    [('Linear Regression', 'linear_regression'),
     ('Initial ANN',       'initial_ann'),
     ('Optimised ANN',     'optimised_ann')]
):
    m = metrics[key]
    col.markdown(f'**{label}**')
    col.write(f'R²  : {m["R2"]:.4f}')
    col.write(f'MAE : {m["MAE"]:.4f}')
    col.write(f'RMSE: {m["RMSE"]:.4f}')

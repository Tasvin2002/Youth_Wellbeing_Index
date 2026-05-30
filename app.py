import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

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

st.set_page_config(page_title='Youth Wellbeing Index', layout='centered', page_icon='🌱')

st.markdown("""
<style>
    /* Hide Streamlit default chrome */
    #MainMenu, footer, header {visibility: hidden;}

    /* Page background */
    .stApp { background: #f0f4ff; }

    /* Survey card */
    .survey-card {
        background: white;
        border-radius: 20px;
        padding: 2.5rem 2rem;
        box-shadow: 0 4px 24px rgba(83,74,183,0.08);
        margin-bottom: 1.5rem;
    }

    /* Section heading */
    .section-label {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #7F77DD;
        margin-bottom: 0.25rem;
    }

    /* Big heading */
    h1.hero {
        font-size: 2rem;
        font-weight: 800;
        color: #26215C;
        margin: 0 0 0.5rem;
        line-height: 1.2;
    }

    /* Step pill */
    .step-pill {
        display: inline-block;
        background: #EEEDFE;
        color: #534AB7;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 4px 14px;
        margin-bottom: 1rem;
    }

    /* Question text */
    .question-text {
        font-size: 1rem;
        font-weight: 600;
        color: #26215C;
        margin-bottom: 0.2rem;
    }

    /* Slider value badge */
    .val-badge {
        display: inline-block;
        background: #EEEDFE;
        color: #534AB7;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 700;
        padding: 2px 10px;
    }

    /* Slider styling */
    .stSlider > div > div > div > div {
        background: #7F77DD !important;
    }
    .stSlider > div > div > div {
        background: #EEEDFE !important;
    }

    /* Radio buttons */
    .stRadio > label { font-weight: 600; color: #26215C; }
    .stRadio > div > label {
        background: #f5f5ff;
        border-radius: 12px;
        padding: 0.5rem 1.2rem !important;
        margin-right: 8px;
        border: 2px solid transparent;
        transition: all 0.15s;
    }
    .stRadio > div > label:has(input:checked) {
        background: #EEEDFE;
        border-color: #7F77DD;
        color: #534AB7;
    }

    /* Result card */
    .result-card {
        background: linear-gradient(135deg, #534AB7 0%, #7F77DD 100%);
        border-radius: 24px;
        padding: 2.5rem 2rem;
        color: white;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .result-score {
        font-size: 4rem;
        font-weight: 900;
        line-height: 1;
        margin: 0.5rem 0;
    }
    .result-label {
        font-size: 1.1rem;
        font-weight: 600;
        opacity: 0.85;
    }

    /* Category pill on result */
    .cat-pill {
        display: inline-block;
        border-radius: 999px;
        font-size: 0.9rem;
        font-weight: 700;
        padding: 6px 20px;
        margin-top: 1rem;
    }

    /* Insight card */
    .insight-card {
        background: white;
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 0.75rem;
        border-left: 4px solid #7F77DD;
    }

    /* Big CTA button */
    .stButton > button {
        background: #534AB7 !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        padding: 0.75rem 2.5rem !important;
        width: 100% !important;
        transition: background 0.2s !important;
        margin-top: 1rem;
    }
    .stButton > button:hover {
        background: #3C3489 !important;
    }

    /* Progress bar */
    .progress-track {
        background: #EEEDFE;
        border-radius: 99px;
        height: 6px;
        width: 100%;
        margin-bottom: 2rem;
    }
    .progress-fill {
        background: #7F77DD;
        border-radius: 99px;
        height: 6px;
    }

    div[data-testid="stVerticalBlock"] { gap: 0.5rem; }
    .element-container { margin-bottom: 0.1rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────
if 'step' not in st.session_state:
    st.session_state.step = 0   # 0 = welcome, 1-3 = survey, 4 = results

step = st.session_state.step

# ── Helpers ────────────────────────────────────────────────
def next_step(): st.session_state.step += 1
def restart():   st.session_state.step = 0

def category(score):
    if score >= 70:  return ("Thriving 🌟", "#1D9E75", "#E1F5EE")
    if score >= 50:  return ("Doing okay 🙂", "#185FA5", "#E6F1FB")
    if score >= 35:  return ("Needs attention ⚠️", "#BA7517", "#FAEEDA")
    return ("At risk 🔴", "#A32D2D", "#FCEBEB")

def insight(age, social_h, sleep_h, stress, anxiety, depression, addiction, late_night):
    tips = []
    if sleep_h < 6:
        tips.append(("Sleep", "You're getting less than 6 hours. Aim for 8–9 hours for better mental health."))
    if social_h > 5:
        tips.append(("Screen time", "Over 5 hours of social media daily is linked to higher anxiety. Try a daily limit."))
    if stress > 7:
        tips.append(("Stress", "Your stress level is high. Regular exercise or mindfulness can help bring it down."))
    if late_night == "Yes":
        tips.append(("Late-night usage", "Using devices late at night disrupts sleep quality and mood the next day."))
    if anxiety > 7:
        tips.append(("Anxiety", "Your anxiety score is elevated. Talking to someone you trust can make a difference."))
    if not tips:
        tips.append(("Keep it up", "Your digital habits look healthy. Keep maintaining those boundaries!"))
    return tips

# ════════════════════════════════════════════════════════════
# STEP 0 — Welcome
# ════════════════════════════════════════════════════════════
if step == 0:
    st.markdown("""
    <div class="survey-card" style="text-align:center; padding: 3rem 2rem;">
        <div style="font-size:3rem; margin-bottom:1rem;">🌱</div>
        <div class="section-label">Youth Wellbeing Index</div>
        <h1 class="hero" style="text-align:center;">How are you <em>really</em> doing?</h1>
        <p style="color:#5F5E5A; font-size:1rem; margin: 1rem 0 0;">
            Answer 12 quick questions about your digital habits and mental health.<br>
            We'll predict your wellbeing score and give personalised insights.
        </p>
        <p style="color:#B4B2A9; font-size:0.8rem; margin-top:1.5rem;">Takes about 2 minutes · No data is stored</p>
    </div>
    """, unsafe_allow_html=True)
    st.button("Start survey →", on_click=next_step)

# ════════════════════════════════════════════════════════════
# STEP 1 — About you
# ════════════════════════════════════════════════════════════
elif step == 1:
    st.markdown("""
    <div class="progress-track"><div class="progress-fill" style="width:33%"></div></div>
    <div class="step-pill">Step 1 of 3</div>
    <h2 style="color:#26215C; font-weight:800; margin-bottom:1.5rem;">About you</h2>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<p class="question-text">How old are you?</p>', unsafe_allow_html=True)
        age = st.slider("Age", 10, 25, 17, label_visibility="collapsed")

        st.markdown('<p class="question-text" style="margin-top:1rem;">What is your gender?</p>', unsafe_allow_html=True)
        gender = st.radio("Gender", label_encoders['gender'].classes_.tolist(),
                          horizontal=True, label_visibility="collapsed")

        st.markdown('<p class="question-text" style="margin-top:1rem;">How many hours do you sleep per night?</p>', unsafe_allow_html=True)
        sleep_hours = st.slider("Sleep hours", 3.0, 12.0, 7.0, 0.5, label_visibility="collapsed")

    st.session_state['age'] = age
    st.session_state['gender'] = gender
    st.session_state['sleep_hours'] = sleep_hours
    st.button("Next →", on_click=next_step)

# ════════════════════════════════════════════════════════════
# STEP 2 — Digital habits
# ════════════════════════════════════════════════════════════
elif step == 2:
    st.markdown("""
    <div class="progress-track"><div class="progress-fill" style="width:66%"></div></div>
    <div class="step-pill">Step 2 of 3</div>
    <h2 style="color:#26215C; font-weight:800; margin-bottom:1.5rem;">Your digital habits</h2>
    """, unsafe_allow_html=True)

    st.markdown('<p class="question-text">Social media hours per day?</p>', unsafe_allow_html=True)
    social_media_hours = st.slider("Social media", 0.0, 12.0, 3.0, 0.5, label_visibility="collapsed")

    st.markdown('<p class="question-text" style="margin-top:1rem;">Do you use your phone late at night?</p>', unsafe_allow_html=True)
    late_night_usage = st.radio("Late night", label_encoders['late_night_usage'].classes_.tolist(),
                                horizontal=True, label_visibility="collapsed")

    st.markdown('<p class="question-text" style="margin-top:1rem;">Digital addiction score (0 = not at all, 10 = very addicted)</p>', unsafe_allow_html=True)
    digital_addiction_score = st.slider("Addiction", 0.0, 10.0, 5.0, 0.5, label_visibility="collapsed")

    st.markdown('<p class="question-text" style="margin-top:1rem;">Average session length (minutes)?</p>', unsafe_allow_html=True)
    average_session_length = st.slider("Session length", 1.0, 180.0, 30.0, 5.0, label_visibility="collapsed")

    st.markdown('<p class="question-text" style="margin-top:1rem;">How many sessions per day?</p>', unsafe_allow_html=True)
    sessions_per_day = st.slider("Sessions", 1, 20, 5, label_visibility="collapsed")

    st.markdown('<p class="question-text" style="margin-top:1rem;">Brain rot index (0 = fresh, 10 = very rotted)</p>', unsafe_allow_html=True)
    brain_rot_index = st.slider("Brain rot", 0.0, 10.0, 5.0, 0.5, label_visibility="collapsed")

    st.session_state.update({
        'social_media_hours': social_media_hours,
        'late_night_usage': late_night_usage,
        'digital_addiction_score': digital_addiction_score,
        'average_session_length': average_session_length,
        'sessions_per_day': sessions_per_day,
        'brain_rot_index': brain_rot_index,
    })
    st.button("Next →", on_click=next_step)

# ════════════════════════════════════════════════════════════
# STEP 3 — Mental health
# ════════════════════════════════════════════════════════════
elif step == 3:
    st.markdown("""
    <div class="progress-track"><div class="progress-fill" style="width:99%"></div></div>
    <div class="step-pill">Step 3 of 3</div>
    <h2 style="color:#26215C; font-weight:800; margin-bottom:0.5rem;">Mental health check-in</h2>
    <p style="color:#888780; font-size:0.9rem; margin-bottom:1.5rem;">Be honest — this is just for you.</p>
    """, unsafe_allow_html=True)

    st.markdown('<p class="question-text">Stress level (0 = calm, 10 = very stressed)</p>', unsafe_allow_html=True)
    stress_level = st.slider("Stress", 0.0, 10.0, 5.0, 0.5, label_visibility="collapsed")

    st.markdown('<p class="question-text" style="margin-top:1rem;">Anxiety score (0 = none, 10 = severe)</p>', unsafe_allow_html=True)
    anxiety_score = st.slider("Anxiety", 0.0, 10.0, 5.0, 0.5, label_visibility="collapsed")

    st.markdown('<p class="question-text" style="margin-top:1rem;">Depression score (0 = none, 10 = severe)</p>', unsafe_allow_html=True)
    depression_score = st.slider("Depression", 0.0, 10.0, 5.0, 0.5, label_visibility="collapsed")

    st.session_state.update({
        'stress_level': stress_level,
        'anxiety_score': anxiety_score,
        'depression_score': depression_score,
    })
    st.button("See my results →", on_click=next_step)

# ════════════════════════════════════════════════════════════
# STEP 4 — Results
# ════════════════════════════════════════════════════════════
elif step == 4:
    s = st.session_state

    gender_encoded     = label_encoders['gender'].transform([s['gender']])[0]
    late_night_encoded = label_encoders['late_night_usage'].transform([s['late_night_usage']])[0]

    input_dict = {
        'age':                            s['age'],
        'gender':                         gender_encoded,
        'social_media_hours':             s['social_media_hours'],
        'sleep_hours':                    s['sleep_hours'],
        'brain_rot_index':                s['brain_rot_index'],
        'late_night_usage':               late_night_encoded,
        'digital_addiction_score':        s['digital_addiction_score'],
        'average_session_length_minutes': s['average_session_length'],
        'sessions_per_day':               s['sessions_per_day'],
        'stress_level':                   s['stress_level'],
        'anxiety_score':                  s['anxiety_score'],
        'depression_score':               s['depression_score'],
    }

    input_array  = np.array([[input_dict[f] for f in feature_columns]])
    input_scaled = scaler.transform(input_array)
    pred_ann     = float(model.predict(input_scaled, verbose=0).flatten()[0])
    score        = round(pred_ann, 1)

    cat_label, cat_color, cat_bg = category(score)

    st.markdown(f"""
    <div class="result-card">
        <div style="font-size:0.85rem; font-weight:600; opacity:0.7; text-transform:uppercase; letter-spacing:0.1em;">Your wellbeing score</div>
        <div class="result-score">{score}</div>
        <div class="result-label">out of 100</div>
        <div class="cat-pill" style="background:{cat_bg}; color:{cat_color};">{cat_label}</div>
    </div>
    """, unsafe_allow_html=True)

    # Score bar visual
    pct = min(max(score, 0), 100)
    bar_color = cat_color
    st.markdown(f"""
    <div style="background:white; border-radius:16px; padding:1.25rem 1.5rem; margin-bottom:1rem;">
        <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:#888780; margin-bottom:6px;">
            <span>0 — At risk</span><span>100 — Thriving</span>
        </div>
        <div style="background:#F1EFE8; border-radius:99px; height:14px;">
            <div style="background:{bar_color}; width:{pct}%; border-radius:99px; height:14px; transition:width 0.6s;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Personalised insights
    st.markdown("<h3 style='color:#26215C; font-weight:800; margin: 1.5rem 0 0.75rem;'>Personalised insights</h3>", unsafe_allow_html=True)
    tips = insight(s['age'], s['social_media_hours'], s['sleep_hours'],
                   s['stress_level'], s['anxiety_score'], s['depression_score'],
                   s['digital_addiction_score'], s['late_night_usage'])
    for title, body in tips:
        st.markdown(f"""
        <div class="insight-card">
            <div style="font-weight:700; color:#26215C; margin-bottom:4px;">{title}</div>
            <div style="color:#5F5E5A; font-size:0.9rem;">{body}</div>
        </div>
        """, unsafe_allow_html=True)

    # Model performance — hidden in expander
    with st.expander("Model performance details", expanded=False):
        m_lr  = metrics['linear_regression']
        m_ann = metrics['optimised_ann']
        c1, c2 = st.columns(2)
        c1.markdown("**Linear Regression**")
        c1.write(f"R²: {m_lr['R2']:.4f}  |  MAE: {m_lr['MAE']:.4f}  |  RMSE: {m_lr['RMSE']:.4f}")
        c2.markdown("**Optimised ANN**")
        c2.write(f"R²: {m_ann['R2']:.4f}  |  MAE: {m_ann['MAE']:.4f}  |  RMSE: {m_ann['RMSE']:.4f}")

    st.button("↩ Take the survey again", on_click=restart)

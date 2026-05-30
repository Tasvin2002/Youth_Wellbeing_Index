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
    #MainMenu, footer, header {visibility: hidden;}

    .stApp {
        background: #eef0fb;
    }

    .block-container {
        max-width: 720px !important;
        padding: 2rem 2rem 4rem !important;
        margin: 0 auto;
    }

    /* Progress bar */
    .prog-wrap {
        background: #d6d3f5;
        border-radius: 99px;
        height: 8px;
        width: 100%;
        margin-bottom: 1.5rem;
    }
    .prog-fill {
        background: #534AB7;
        border-radius: 99px;
        height: 8px;
    }

    /* Step pill */
    .step-pill {
        display: inline-block;
        background: #534AB7;
        color: white;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 5px 16px;
        margin-bottom: 0.75rem;
        letter-spacing: 0.05em;
    }

    /* Page heading */
    .page-heading {
        font-size: 1.4rem;
        font-weight: 800;
        color: #26215C;
        margin-bottom: 0.25rem;
        line-height: 1.3;
    }
    .page-subheading {
        font-size: 0.9rem;
        color: #6B6B9A;
        margin-bottom: 1.75rem;
    }

    /* Question card */
    .q-card {
        background: white;
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        border: 0.5px solid #dddaf5;
    }
    .q-label {
        font-size: 0.95rem;
        font-weight: 700;
        color: #26215C;
        margin-bottom: 0.5rem;
    }
    .q-hint {
        font-size: 0.78rem;
        color: #9896C8;
        margin-top: -0.25rem;
        margin-bottom: 0.5rem;
    }

    /* Override Streamlit slider thumb and track to purple */
    [data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
        background: #534AB7 !important;
        border-color: #534AB7 !important;
    }
    [data-testid="stSlider"] [data-baseweb="slider"] div[class*="sliderTrackFilled"] {
        background: #534AB7 !important;
    }
    [data-testid="stSlider"] div[data-testid="stTickBarMin"],
    [data-testid="stSlider"] div[data-testid="stTickBarMax"] {
        color: #9896C8 !important;
        font-size: 0.75rem !important;
    }

    /* Radio button override */
    [data-testid="stRadio"] > label {
        font-size: 0.85rem;
        font-weight: 700;
        color: #26215C;
    }
    [data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {
        font-size: 0.9rem;
    }

    /* Next / Submit button */
    .stButton > button {
        background: #534AB7 !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        padding: 0.65rem 0 !important;
        width: 100% !important;
        margin-top: 0.5rem;
        letter-spacing: 0.02em;
    }
    .stButton > button:hover {
        background: #3C3489 !important;
    }

    /* Welcome hero */
    .hero-card {
        background: linear-gradient(135deg, #534AB7 0%, #7F77DD 100%);
        border-radius: 24px;
        padding: 3rem 2.5rem;
        text-align: center;
        color: white;
        margin-bottom: 1.5rem;
    }
    .hero-emoji { font-size: 3rem; margin-bottom: 0.75rem; }
    .hero-title {
        font-size: 1.8rem;
        font-weight: 900;
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }
    .hero-sub {
        font-size: 0.95rem;
        opacity: 0.8;
        line-height: 1.6;
    }
    .hero-note {
        font-size: 0.78rem;
        opacity: 0.55;
        margin-top: 1.25rem;
    }

    /* Result card */
    .result-hero {
        background: linear-gradient(135deg, #534AB7 0%, #7F77DD 100%);
        border-radius: 24px;
        padding: 2.5rem 2rem;
        text-align: center;
        color: white;
        margin-bottom: 1rem;
    }
    .result-score-label {
        font-size: 0.8rem;
        font-weight: 700;
        opacity: 0.7;
        text-transform: uppercase;
        letter-spacing: 0.12em;
    }
    .result-score {
        font-size: 5rem;
        font-weight: 900;
        line-height: 1;
        margin: 0.4rem 0;
    }
    .result-outof {
        font-size: 0.95rem;
        opacity: 0.65;
    }

    /* Score bar */
    .score-bar-wrap {
        background: white;
        border-radius: 16px;
        padding: 1.1rem 1.5rem;
        margin-bottom: 1rem;
        border: 0.5px solid #dddaf5;
    }
    .score-bar-labels {
        display: flex;
        justify-content: space-between;
        font-size: 0.75rem;
        color: #9896C8;
        margin-bottom: 6px;
    }
    .score-bar-track {
        background: #eef0fb;
        border-radius: 99px;
        height: 12px;
    }

    /* Insight card */
    .insight-card {
        background: white;
        border-radius: 14px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        border-left: 4px solid #534AB7;
        border-top: 0.5px solid #dddaf5;
        border-right: 0.5px solid #dddaf5;
        border-bottom: 0.5px solid #dddaf5;
    }
    .insight-title {
        font-size: 0.9rem;
        font-weight: 700;
        color: #26215C;
        margin-bottom: 4px;
    }
    .insight-body {
        font-size: 0.85rem;
        color: #6B6B9A;
        line-height: 1.5;
    }

    div[data-testid="stVerticalBlock"] { gap: 0rem; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────
if 'step' not in st.session_state:
    st.session_state.step = 0

def next_step(): st.session_state.step += 1
def restart():   st.session_state.step = 0

# ── Helpers ────────────────────────────────────────────────
def category(score):
    if score >= 70:  return ("Thriving 🌟",       "#0F6E56", "#E1F5EE")
    if score >= 50:  return ("Doing okay 🙂",      "#185FA5", "#E6F1FB")
    if score >= 35:  return ("Needs attention ⚠️", "#BA7517", "#FAEEDA")
    return                  ("At risk 💙",          "#A32D2D", "#FCEBEB")

def get_insights(s):
    tips = []
    if s['sleep_hours'] < 6:
        tips.append(("Sleep more",        "You're getting under 6 hours. Teens need 8–9 hours for focus and mood."))
    if s['social_media_hours'] > 5:
        tips.append(("Cut screen time",   "5+ hours of social media daily is linked to higher anxiety. Try setting a daily limit."))
    if s['stress_level'] > 7:
        tips.append(("Manage stress",     "Your stress is high. Even 10 minutes of walking or breathing exercises helps."))
    if s['late_night_usage'] == 'Yes':
        tips.append(("Phone before bed",  "Late-night device use disrupts sleep quality and your mood the next day."))
    if s['anxiety_score'] > 7:
        tips.append(("Talk to someone",   "Your anxiety score is elevated. Reaching out to a friend or counsellor can really help."))
    if s['depression_score'] > 7:
        tips.append(("You're not alone",  "Your depression score is high. Please consider speaking to someone you trust or a professional."))
    if s['digital_addiction_score'] > 7:
        tips.append(("Digital detox",     "Your addiction score is high. Try one hour without your phone each evening."))
    if not tips:
        tips.append(("Keep it up! 🎉",    "Your habits look healthy overall. Keep maintaining those boundaries and routines."))
    return tips

step = st.session_state.step

# ════════════════════════════════════════════════════════════
# STEP 0 — Welcome
# ════════════════════════════════════════════════════════════
if step == 0:
    st.markdown("""
    <div class="hero-card">
        <div class="hero-emoji">🌱</div>
        <div class="hero-title">How are you really doing?</div>
        <div class="hero-sub">
            Answer 12 quick questions about your digital habits<br>and mental health. We'll predict your wellbeing score<br>and give you personalised insights.
        </div>
        <div class="hero-note">Takes about 2 minutes &nbsp;·&nbsp; No data is stored</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; margin-bottom:1.5rem;">
        <div class="q-card" style="text-align:center; padding:1rem;">
            <div style="font-size:1.5rem;">📱</div>
            <div style="font-size:0.8rem; font-weight:700; color:#26215C; margin-top:6px;">Digital habits</div>
            <div style="font-size:0.75rem; color:#9896C8;">Screen time & usage patterns</div>
        </div>
        <div class="q-card" style="text-align:center; padding:1rem;">
            <div style="font-size:1.5rem;">🧠</div>
            <div style="font-size:0.8rem; font-weight:700; color:#26215C; margin-top:6px;">Mental health</div>
            <div style="font-size:0.75rem; color:#9896C8;">Stress, anxiety & mood</div>
        </div>
        <div class="q-card" style="text-align:center; padding:1rem;">
            <div style="font-size:1.5rem;">✨</div>
            <div style="font-size:0.8rem; font-weight:700; color:#26215C; margin-top:6px;">Your score</div>
            <div style="font-size:0.75rem; color:#9896C8;">Personalised insights</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.button("Start survey →", on_click=next_step)

# ════════════════════════════════════════════════════════════
# STEP 1 — About you
# ════════════════════════════════════════════════════════════
elif step == 1:
    st.markdown('<div class="prog-wrap"><div class="prog-fill" style="width:33%"></div></div>', unsafe_allow_html=True)
    st.markdown('<span class="step-pill">Step 1 of 3</span>', unsafe_allow_html=True)
    st.markdown('<div class="page-heading">About you</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subheading">Let\'s start with some basics about you.</div>', unsafe_allow_html=True)

    st.markdown('<div class="q-card"><div class="q-label">How old are you?</div><div class="q-hint">Move the slider to your age</div>', unsafe_allow_html=True)
    age = st.slider("Age", 10, 25, 17, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="q-card"><div class="q-label">What is your gender?</div>', unsafe_allow_html=True)
    gender = st.radio("Gender", label_encoders['gender'].classes_.tolist(), horizontal=True, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="q-card"><div class="q-label">How many hours do you sleep per night?</div><div class="q-hint">Recommended: 8–9 hours for teenagers</div>', unsafe_allow_html=True)
    sleep_hours = st.slider("Sleep", 3.0, 12.0, 7.0, 0.5, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    st.session_state['age'] = age
    st.session_state['gender'] = gender
    st.session_state['sleep_hours'] = sleep_hours
    st.button("Next →", on_click=next_step)

# ════════════════════════════════════════════════════════════
# STEP 2 — Digital habits
# ════════════════════════════════════════════════════════════
elif step == 2:
    st.markdown('<div class="prog-wrap"><div class="prog-fill" style="width:66%"></div></div>', unsafe_allow_html=True)
    st.markdown('<span class="step-pill">Step 2 of 3</span>', unsafe_allow_html=True)
    st.markdown('<div class="page-heading">Your digital habits</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subheading">Be honest — there are no wrong answers here.</div>', unsafe_allow_html=True)

    st.markdown('<div class="q-card"><div class="q-label">How many hours do you spend on social media per day?</div><div class="q-hint">Include TikTok, Instagram, YouTube, etc.</div>', unsafe_allow_html=True)
    social_media_hours = st.slider("Social media", 0.0, 12.0, 3.0, 0.5, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="q-card"><div class="q-label">Do you use your phone late at night (after 11pm)?</div>', unsafe_allow_html=True)
    late_night_usage = st.radio("Late night", label_encoders['late_night_usage'].classes_.tolist(), horizontal=True, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="q-card"><div class="q-label">How addicted do you feel to your devices?</div><div class="q-hint">0 = not at all &nbsp;·&nbsp; 10 = can\'t put it down</div>', unsafe_allow_html=True)
    digital_addiction_score = st.slider("Addiction", 0.0, 10.0, 5.0, 0.5, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="q-card"><div class="q-label">How long is your average session? (minutes)</div><div class="q-hint">One continuous period of device use</div>', unsafe_allow_html=True)
    average_session_length = st.slider("Session length", 1.0, 180.0, 30.0, 5.0, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="q-card"><div class="q-label">How many separate sessions do you have per day?</div>', unsafe_allow_html=True)
    sessions_per_day = st.slider("Sessions", 1, 20, 5, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="q-card"><div class="q-label">How would you rate your "brain rot" level?</div><div class="q-hint">0 = sharp & focused &nbsp;·&nbsp; 10 = can\'t focus on anything</div>', unsafe_allow_html=True)
    brain_rot_index = st.slider("Brain rot", 0.0, 10.0, 5.0, 0.5, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    st.session_state.update({
        'social_media_hours':     social_media_hours,
        'late_night_usage':       late_night_usage,
        'digital_addiction_score': digital_addiction_score,
        'average_session_length': average_session_length,
        'sessions_per_day':       sessions_per_day,
        'brain_rot_index':        brain_rot_index,
    })
    st.button("Next →", on_click=next_step)

# ════════════════════════════════════════════════════════════
# STEP 3 — Mental health
# ════════════════════════════════════════════════════════════
elif step == 3:
    st.markdown('<div class="prog-wrap"><div class="prog-fill" style="width:99%"></div></div>', unsafe_allow_html=True)
    st.markdown('<span class="step-pill">Step 3 of 3</span>', unsafe_allow_html=True)
    st.markdown('<div class="page-heading">Mental health check-in</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subheading">This is just for you. Answer as honestly as you can.</div>', unsafe_allow_html=True)

    st.markdown('<div class="q-card"><div class="q-label">How stressed have you been feeling lately?</div><div class="q-hint">0 = completely calm &nbsp;·&nbsp; 10 = extremely stressed</div>', unsafe_allow_html=True)
    stress_level = st.slider("Stress", 0.0, 10.0, 5.0, 0.5, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="q-card"><div class="q-label">How much anxiety have you been experiencing?</div><div class="q-hint">0 = none at all &nbsp;·&nbsp; 10 = severe, affecting daily life</div>', unsafe_allow_html=True)
    anxiety_score = st.slider("Anxiety", 0.0, 10.0, 5.0, 0.5, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="q-card"><div class="q-label">How are you feeling emotionally overall?</div><div class="q-hint">0 = feeling great &nbsp;·&nbsp; 10 = feeling very low</div>', unsafe_allow_html=True)
    depression_score = st.slider("Depression", 0.0, 10.0, 5.0, 0.5, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    st.session_state.update({
        'stress_level':    stress_level,
        'anxiety_score':   anxiety_score,
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
    score        = max(0.0, min(100.0, score))

    cat_label, cat_color, cat_bg = category(score)

    st.markdown(f"""
    <div class="result-hero">
        <div class="result-score-label">Your wellbeing score</div>
        <div class="result-score">{score}</div>
        <div class="result-outof">out of 100</div>
        <div style="display:inline-block; background:rgba(255,255,255,0.2);
                    border-radius:999px; padding:6px 20px; margin-top:12px;
                    font-size:1rem; font-weight:700;">{cat_label}</div>
    </div>
    """, unsafe_allow_html=True)

    pct = score
    st.markdown(f"""
    <div class="score-bar-wrap">
        <div class="score-bar-labels">
            <span>0 — At risk</span>
            <span>50 — Okay</span>
            <span>100 — Thriving</span>
        </div>
        <div class="score-bar-track">
            <div style="background:{cat_color}; width:{pct}%;
                        border-radius:99px; height:12px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:1rem; font-weight:800; color:#26215C; margin: 1.25rem 0 0.75rem;">Personalised insights</div>', unsafe_allow_html=True)

    tips = get_insights(s)
    for title, body in tips:
        st.markdown(f"""
        <div class="insight-card">
            <div class="insight-title">{title}</div>
            <div class="insight-body">{body}</div>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("Model performance details"):
        m_lr  = metrics['linear_regression']
        m_ann = metrics['optimised_ann']
        c1, c2 = st.columns(2)
        c1.markdown("**Linear Regression**")
        c1.write(f"R²: {m_lr['R2']:.4f}  |  MAE: {m_lr['MAE']:.4f}  |  RMSE: {m_lr['RMSE']:.4f}")
        c2.markdown("**Optimised ANN**")
        c2.write(f"R²: {m_ann['R2']:.4f}  |  MAE: {m_ann['MAE']:.4f}  |  RMSE: {m_ann['RMSE']:.4f}")

    st.button("↩ Take the survey again", on_click=restart)

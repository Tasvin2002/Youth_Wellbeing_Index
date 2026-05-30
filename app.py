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
        background: #EEEDFE;
    }

    .block-container {
        max-width: 680px !important;
        padding: 2.5rem 2rem 4rem !important;
        margin: 0 auto;
    }

    /* Force ALL slider thumbs and filled track to purple — override Streamlit's red */
    [data-testid="stSlider"] * { accent-color: #534AB7 !important; }
    [data-testid="stSlider"] [role="slider"] {
        background-color: #534AB7 !important;
        border-color: #534AB7 !important;
        outline-color: #534AB7 !important;
    }
    [data-testid="stSlider"] > div > div > div > div {
        background: #534AB7 !important;
    }
    .stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] {
        color: #534AB7 !important;
    }

    /* Radio — purple accent */
    [data-testid="stRadio"] * { accent-color: #534AB7 !important; }

    /* Section divider */
    .section-divider {
        border: none;
        border-top: 1.5px solid #D6D3F5;
        margin: 1.5rem 0;
    }

    /* Step pill */
    .step-pill {
        display: inline-block;
        background: #534AB7;
        color: white;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 800;
        padding: 7px 18px;
        margin-bottom: 0.6rem;
    }

    /* Progress bar */
    .prog-wrap {
        background: #C9C5F0;
        border-radius: 99px;
        height: 7px;
        margin-bottom: 1.25rem;
    }
    .prog-fill {
        background: #534AB7;
        border-radius: 99px;
        height: 7px;
    }

    /* Page title */
    .page-title {
        font-size: 1.5rem;
        font-weight: 800;
        color: #26215C;
        margin-bottom: 0.2rem;
    }
    .page-sub {
        font-size: 0.88rem;
        color: #7F77DD;
        margin-bottom: 1.5rem;
    }

    /* Question label above each input */
    .q-label {
        font-size: 0.95rem;
        font-weight: 700;
        color: #26215C;
        margin-top: 1.1rem;
        margin-bottom: 0rem;
    }
    .q-hint {
        font-size: 0.65rem !important;;
        color: #9896C8;
        margin-bottom: 0.1rem;
        margin-top: 0;
    }

    /* Streamlit default label — hide it since we use our own */
    [data-testid="stSlider"] label,
    [data-testid="stRadio"] > label {
        display: none !important;
    }

    /* Next button */
    .stButton > button {
        background: #534AB7 !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        padding: 0.7rem 0 !important;
        width: 100% !important;
        margin-top: 1.5rem;
    }
    .stButton > button:hover {
        background: #3C3489 !important;
    }
    .stButton > button:active {
        background: #26215C !important;
    }

    /* Welcome hero */
    .hero-wrap {
        background: linear-gradient(145deg, #534AB7, #7F77DD);
        border-radius: 24px;
        padding: 3rem 2.5rem;
        text-align: center;
        color: white;
        margin-bottom: 1.5rem;
    }
    .hero-emoji { font-size: 2.8rem; margin-bottom: 0.6rem; }
    .hero-title { font-size: 1.9rem; font-weight: 900; margin-bottom: 0.6rem; line-height: 1.2; }
    .hero-body  { font-size: 0.95rem; opacity: 0.85; line-height: 1.65; }
    .hero-note  { font-size: 0.75rem; opacity: 0.5; margin-top: 1.2rem; }

    /* Welcome 3-feature strip */
    .feature-strip {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 10px;
        margin-bottom: 1.5rem;
    }
    .feature-box {
        background: white;
        border-radius: 14px;
        padding: 1rem 0.75rem;
        text-align: center;
        border: 0.5px solid #D6D3F5;
    }
    .feature-icon { font-size: 1.4rem; margin-bottom: 6px; }
    .feature-name { font-size: 0.82rem; font-weight: 700; color: #26215C; }
    .feature-desc { font-size: 0.73rem; color: #9896C8; margin-top: 2px; }

    /* Result hero */
    .result-hero {
        background: linear-gradient(145deg, #534AB7, #7F77DD);
        border-radius: 24px;
        padding: 2.5rem 2rem;
        text-align: center;
        color: white;
        margin-bottom: 1rem;
    }
    .result-eyebrow {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        opacity: 0.65;
        margin-bottom: 4px;
    }
    .result-number {
        font-size: 5.5rem;
        font-weight: 900;
        line-height: 1;
        margin-bottom: 2px;
    }
    .result-outof {
        font-size: 0.85rem;
        opacity: 0.6;
        margin-bottom: 1rem;
    }
    .result-badge {
        display: inline-block;
        background: rgba(255,255,255,0.18);
        border-radius: 999px;
        padding: 7px 22px;
        font-size: 1rem;
        font-weight: 700;
    }

    /* Score bar */
    .bar-card {
        background: white;
        border-radius: 16px;
        padding: 1.1rem 1.5rem 1.2rem;
        margin-bottom: 0.9rem;
        border: 0.5px solid #D6D3F5;
    }
    .bar-labels {
        display: flex;
        justify-content: space-between;
        font-size: 0.73rem;
        color: #9896C8;
        margin-bottom: 7px;
    }
    .bar-track {
        background: #EEEDFE;
        border-radius: 99px;
        height: 13px;
    }

    /* Insight cards */
    .insight {
        background: white;
        border-radius: 14px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        border: 0.5px solid #D6D3F5;
        border-left: 4px solid #534AB7;
    }
    .insight-title {
        font-size: 0.9rem;
        font-weight: 700;
        color: #26215C;
        margin-bottom: 3px;
    }
    .insight-body {
        font-size: 0.84rem;
        color: #6B6B9A;
        line-height: 1.55;
    }

    /* Section heading on results */
    .results-section-head {
        font-size: 1rem;
        font-weight: 800;
        color: #26215C;
        margin: 1.4rem 0 0.6rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────
if 'step' not in st.session_state:
    st.session_state.step = 0

def next_step(): st.session_state.step += 1
def restart():   st.session_state.step = 0

def category(score):
    if score >= 70: return ("Thriving 🌟",        "#0F6E56", "#1D9E75")
    if score >= 50: return ("Doing okay 🙂",       "#185FA5", "#378ADD")
    if score >= 35: return ("Needs attention ⚠️",  "#BA7517", "#EF9F27")
    return                 ("At risk — seek help 💙","#A32D2D", "#E24B4A")

def get_insights(s):
    tips = []
    if s['sleep_hours'] < 6:
        tips.append(("Get more sleep",
                     f"You're sleeping {s['sleep_hours']} hours. Teenagers need 8–9 hours for focus, mood, and health."))
    if s['social_media_hours'] > 5:
        tips.append(("Reduce screen time",
                     f"You spend {s['social_media_hours']} hours on social media daily. Over 5 hours is linked to higher anxiety and lower wellbeing."))
    if s['stress_level'] > 7:
        tips.append(("Manage your stress",
                     "Your stress is high. Try 10 minutes of walking, deep breathing, or journalling each day."))
    if s['late_night_usage'] == 'Yes' or s['late_night_usage'] == 'Always' or s['late_night_usage'] == 'Often':
        tips.append(("Put the phone down at night",
                     "Late-night device use disrupts sleep quality and increases anxiety the next day. Try stopping 1 hour before bed."))
    if s['anxiety_score'] > 7:
        tips.append(("Address your anxiety",
                     "Your anxiety score is elevated. Talking to a trusted friend, teacher, or counsellor can really help."))
    if s['depression_score'] > 7:
        tips.append(("You are not alone",
                     "You seem to be feeling quite low. Please consider reaching out to someone you trust or a mental health professional."))
    if s['digital_addiction_score'] > 7:
        tips.append(("Try a digital detox",
                     "Your device addiction score is high. Start small — try one hour without your phone each evening."))
    if not tips:
        tips.append(("You're doing great! 🎉",
                     "Your habits and mental health indicators all look healthy. Keep maintaining your routines and boundaries."))
    return tips

step = st.session_state.step

# ════════════════════════════════════════════════════════════
# STEP 0 — Welcome
# ════════════════════════════════════════════════════════════
if step == 0:
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-emoji">🌱</div>
        <div class="hero-title">How are you really doing?</div>
        <div class="hero-body">
            Answer 12 quick questions about your digital habits and mental health.<br>
            We'll calculate your personal wellbeing score and give you tailored insights.
        </div>
        <div class="hero-note">Takes about 2 minutes &nbsp;·&nbsp; No data is stored &nbsp;·&nbsp; Powered by AI</div>
    </div>
    <div class="feature-strip">
        <div class="feature-box">
            <div class="feature-icon">📱</div>
            <div class="feature-name">Digital habits</div>
            <div class="feature-desc">Screen time & usage patterns</div>
        </div>
        <div class="feature-box">
            <div class="feature-icon">🧠</div>
            <div class="feature-name">Mental health</div>
            <div class="feature-desc">Stress, anxiety & mood</div>
        </div>
        <div class="feature-box">
            <div class="feature-icon">✨</div>
            <div class="feature-name">Your score</div>
            <div class="feature-desc">Personalised insights</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.button("Start survey →", on_click=next_step)

# ════════════════════════════════════════════════════════════
# STEP 1 — About you
# ════════════════════════════════════════════════════════════
elif step == 1:
    st.markdown('<div class="prog-wrap"><div class="prog-fill" style="width:33%"></div></div>', unsafe_allow_html=True)
    st.markdown('<span class="step-pill">Step 1 of 3 — About you</span>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Tell us about yourself</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Just a few basics to personalise your results.</div>', unsafe_allow_html=True)

    st.markdown('<p class="q-label">How old are you?</p>', unsafe_allow_html=True)
    age = st.slider("Age", 10, 25, 17, label_visibility="collapsed")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<p class="q-label">What is your gender?</p>', unsafe_allow_html=True)
    gender = st.radio("Gender", label_encoders['gender'].classes_.tolist(),
                      horizontal=True, label_visibility="collapsed")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<p class="q-label">How many hours do you sleep per night?</p>', unsafe_allow_html=True)
    st.markdown('<p class="q-hint">Recommended for teenagers: 8–9 hours</p>', unsafe_allow_html=True)
    sleep_hours = st.slider("Sleep", 3.0, 12.0, 7.0, 0.5, label_visibility="collapsed")

    st.session_state['age'] = age
    st.session_state['gender'] = gender
    st.session_state['sleep_hours'] = sleep_hours
    st.button("Next →", on_click=next_step)

# ════════════════════════════════════════════════════════════
# STEP 2 — Digital habits
# ════════════════════════════════════════════════════════════
elif step == 2:
    st.markdown('<div class="prog-wrap"><div class="prog-fill" style="width:66%"></div></div>', unsafe_allow_html=True)
    st.markdown('<span class="step-pill">Step 2 of 3 — Digital habits</span>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Your digital habits</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Be honest — there are no wrong answers here.</div>', unsafe_allow_html=True)

    st.markdown('<p class="q-label">How many hours do you spend on social media per day?</p>', unsafe_allow_html=True)
    st.markdown('<p class="q-hint">Include TikTok, Instagram, YouTube, X, etc.</p>', unsafe_allow_html=True)
    social_media_hours = st.slider("Social media", 0.0, 12.0, 3.0, 0.5, label_visibility="collapsed")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<p class="q-label">How often do you use your phone late at night (after 11pm)?</p>', unsafe_allow_html=True)
    late_night_usage = st.radio("Late night", label_encoders['late_night_usage'].classes_.tolist(),
                                horizontal=True, label_visibility="collapsed")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<p class="q-label">How addicted do you feel to your devices?</p>', unsafe_allow_html=True)
    st.markdown('<p class="q-hint">0 = not at all &nbsp;&nbsp; 10 = cannot put it down</p>', unsafe_allow_html=True)
    digital_addiction_score = st.slider("Addiction", 0.0, 10.0, 5.0, 0.5, label_visibility="collapsed")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<p class="q-label">How long is your average screen session?</p>', unsafe_allow_html=True)
    st.markdown('<p class="q-hint">One continuous period of device use, in minutes</p>', unsafe_allow_html=True)
    average_session_length = st.slider("Session length", 1.0, 180.0, 30.0, 5.0, label_visibility="collapsed")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<p class="q-label">How many separate device sessions do you have per day?</p>', unsafe_allow_html=True)
    sessions_per_day = st.slider("Sessions", 1, 20, 5, label_visibility="collapsed")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<p class="q-label">How is your ability to focus and concentrate lately?</p>', unsafe_allow_html=True)
    st.markdown('<p class="q-hint">0 = sharp and focused &nbsp;&nbsp; 10 = cannot focus on anything</p>', unsafe_allow_html=True)
    brain_rot_index = st.slider("Brain rot", 0.0, 10.0, 5.0, 0.5, label_visibility="collapsed")

    st.session_state.update({
        'social_media_hours':      social_media_hours,
        'late_night_usage':        late_night_usage,
        'digital_addiction_score': digital_addiction_score,
        'average_session_length':  average_session_length,
        'sessions_per_day':        sessions_per_day,
        'brain_rot_index':         brain_rot_index,
    })
    st.button("Next →", on_click=next_step)

# ════════════════════════════════════════════════════════════
# STEP 3 — Mental health
# ════════════════════════════════════════════════════════════
elif step == 3:
    st.markdown('<div class="prog-wrap"><div class="prog-fill" style="width:99%"></div></div>', unsafe_allow_html=True)
    st.markdown('<span class="step-pill">Step 3 of 3 — Mental health</span>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Mental health check-in</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">This is just for you. Answer as honestly as you can.</div>', unsafe_allow_html=True)

    st.markdown('<p class="q-label">How stressed have you been feeling lately?</p>', unsafe_allow_html=True)
    st.markdown('<p class="q-hint">0 = completely calm &nbsp;&nbsp; 10 = extremely stressed</p>', unsafe_allow_html=True)
    stress_level = st.slider("Stress", 0.0, 10.0, 5.0, 0.5, label_visibility="collapsed")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<p class="q-label">How much anxiety have you been experiencing?</p>', unsafe_allow_html=True)
    st.markdown('<p class="q-hint">0 = none at all &nbsp;&nbsp; 10 = severe, affecting daily life</p>', unsafe_allow_html=True)
    anxiety_score = st.slider("Anxiety", 0.0, 10.0, 5.0, 0.5, label_visibility="collapsed")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<p class="q-label">How are you feeling emotionally overall?</p>', unsafe_allow_html=True)
    st.markdown('<p class="q-hint">0 = feeling great &nbsp;&nbsp; 10 = feeling very low</p>', unsafe_allow_html=True)
    depression_score = st.slider("Depression", 0.0, 10.0, 5.0, 0.5, label_visibility="collapsed")

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
    score        = round(max(0.0, min(100.0, pred_ann)), 1)

    cat_label, cat_dark, cat_mid = category(score)

    st.markdown(f"""
    <div class="result-hero">
        <div class="result-eyebrow">Your wellbeing score</div>
        <div class="result-number">{score}</div>
        <div class="result-outof">out of 100</div>
        <div class="result-badge">{cat_label}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="bar-card">
        <div class="bar-labels">
            <span>0 — At risk</span><span>50 — Okay</span><span>100 — Thriving</span>
        </div>
        <div class="bar-track">
            <div style="background:{cat_mid}; width:{score}%; border-radius:99px; height:13px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="results-section-head">What this means for you</div>', unsafe_allow_html=True)

    for title, body in get_insights(s):
        st.markdown(f"""
        <div class="insight">
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

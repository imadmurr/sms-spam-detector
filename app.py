"""Streamlit demo: classify an SMS as spam or ham and show which regex fingerprints fired.

Run locally with:  streamlit run app.py
"""
import nltk
import streamlit as st

# NLTK data must exist before features.py is imported (it loads the stopword list on import).
for package in ["punkt_tab", "stopwords", "wordnet", "averaged_perceptron_tagger_eng"]:
    nltk.download(package, quiet=True)

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from features import FINGERPRINTS, PATTERNS, build_input  # noqa: E402

MODEL_PATH = "models/model3.joblib"

FINGERPRINT_LABELS = {
    "n_urls": "Link / URL",
    "n_phones": "Phone number",
    "n_shortcodes": "5-digit short code",
    "n_money": "Money amount",
    "n_caps": "CAPS word",
    "n_urgent": "Urgency word",
    "n_exclaim": "Exclamation mark",
}

# Hand-written examples (FR-16): two scam-style messages and two ordinary ones.
EXAMPLES = {
    "Parcel fee scam": "Your parcel is on hold. Pay the £1.99 redelivery fee within 24h: www.parcel-redeliver.com",
    "Prize scam": "URGENT! You have WON a £1000 cash prize. Call 09061701461 now to claim. Reply STOP to opt out",
    "Friend": "Hey, are we still on for dinner tonight? I'll be there around 8",
    "Family": "Mum says don't forget to pick up the bread on your way home",
}


@st.cache_resource
def load_model():
    """Load the trained Model 3 pipeline once per server, not on every click."""
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="SMS Spam Detector", page_icon="📱", layout="centered")
st.title("SMS Spam & Phishing Detector")
st.caption("TF-IDF words + regex fingerprints + POS features → Logistic Regression, "
           "trained on the UCI SMS Spam Collection.")

if "message" not in st.session_state:
    st.session_state.message = ""

st.write("Try an example:")
cols = st.columns(len(EXAMPLES))
for col, (name, text) in zip(cols, EXAMPLES.items()):
    if col.button(name, use_container_width=True):
        st.session_state.message = text

message = st.text_area("SMS message", key="message", height=120, placeholder="Paste or type an SMS…")

if st.button("Classify", type="primary", disabled=not message.strip()):
    model = load_model()
    p_spam = float(model.predict_proba(build_input(message))[0, 1])

    if p_spam >= 0.5:
        st.error(f"**Spam**: {p_spam:.0%} spam probability")
    else:
        st.success(f"**Ham** (not spam): {p_spam:.0%} spam probability")
    st.progress(p_spam)

    st.subheader("Fingerprints that fired")
    fired = [
        {"Fingerprint": FINGERPRINT_LABELS[name],
         "Count": len(matches),
         "Matched": ", ".join(m.strip() for m in matches)}
        for name in FINGERPRINTS
        if (matches := PATTERNS[name].findall(message))
    ]
    if fired:
        st.dataframe(pd.DataFrame(fired), hide_index=True, use_container_width=True)
    else:
        st.write("None: no links, phone numbers, short codes, money, CAPS, urgency words or exclamation marks.")
    st.caption(f"{len(fired)} of {len(FINGERPRINTS)} fingerprint types fired. "
               "The pure regex rule (Model 0) flags a message as spam when 3 or more fire.")

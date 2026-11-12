"""Streamlit demo: classify an SMS as spam or ham and explain why.

Run locally with:  streamlit run app.py
"""
import html

import nltk
import streamlit as st

# NLTK data must exist before features.py is imported (it loads the stopword list on import).
for package in ["punkt_tab", "stopwords", "wordnet", "averaged_perceptron_tagger_eng"]:
    nltk.download(package, quiet=True)

import joblib  # noqa: E402

from features import FINGERPRINTS, PATTERNS, build_input, clean_text, process_tokens  # noqa: E402

MODEL_PATH = "models/model3.joblib"
RULE_K = 3  # Model 0: spam if at least 3 distinct fingerprints fire (k tuned in the notebook)

# Fingerprint -> (label, highlight colour)
FINGERPRINT_STYLE = {
    "n_urls": ("Link", "#3B82F6"),
    "n_phones": ("Phone number", "#EF4444"),
    "n_shortcodes": ("Short code", "#F97316"),
    "n_money": ("Money", "#22C55E"),
    "n_caps": ("CAPS", "#A855F7"),
    "n_urgent": ("Urgency word", "#EAB308"),
    "n_exclaim": ("Exclamation", "#EC4899"),
}

EXAMPLES = {
    "Scams": {
        "📦 Parcel fee": "Royal Mail: your parcel is on hold due to an unpaid £1.99 shipping fee. Pay now at www.royalmail-redelivery.com to avoid return.",
        "🏦 Bank alert": "ALERT: Your Barclays account has been SUSPENDED due to unusual activity. Verify your details now at http://barclays-secure-login.co.uk",
        "🎁 Prize": "CONGRATULATIONS! You have been selected to receive a £500 Tesco voucher. Text CLAIM to 87121 now! T&Cs apply",
        "💸 Loan": "Need cash fast? Get up to £5000 today, no credit check! Call 08712345678 or reply YES. Txt STOP to opt out",
    },
    "Legit": {
        "🍝 Dinner": "Hey! Are we still on for dinner tonight? I'll be there around 8, save me a seat",
        "🎟️ Tickets": "Can you send me the £20 for the concert tickets when you get a chance? Thanks!",
        "🚗 Good news": "OMG I PASSED MY DRIVING TEST!!! Celebrating tonight, come over",
        "🔐 OTP code": "Your verification code is 482913. Do not share this code with anyone.",
    },
    "Tricky": {
        "📱 'Hi mum'": "Hi mum, I dropped my phone in water. This is my new number, can you message me on WhatsApp when you're free?",
        "💼 Job offer": "Hello, we saw your CV online. Remote job, earn 300 per day working from home. Interested?",
    },
}

CSS = """
<style>
.block-container {max-width: 1100px; padding-top: 2rem;}
.hero h1 {font-size: 2.4rem; margin-bottom: 0; background: linear-gradient(90deg, #7C5CFF, #22D3EE);
          -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
.hero p {color: #9AA0B4; margin-top: .25rem;}
.card {background: #1A1D29; border: 1px solid #2A2F42; border-radius: 14px; padding: 1.1rem 1.3rem; margin-bottom: 1rem;}
.verdict {font-size: 2rem; font-weight: 700;}
.muted {color: #9AA0B4; font-size: .85rem;}
.msg {line-height: 2; font-size: 1.05rem;}
.msg mark {color: #0E1117; border-radius: 6px; padding: 2px 5px; font-weight: 600;}
.chip {display: inline-block; border-radius: 999px; padding: 3px 10px; margin: 3px; font-size: .8rem;
       border: 1px solid #2A2F42; color: #6B7186;}
.chip.on {color: #0E1117; font-weight: 600; border-color: transparent;}
</style>
"""


@st.cache_resource
def load_model():
    """Load the trained Model 3 pipeline once per server, not on every rerun."""
    return joblib.load(MODEL_PATH)


def highlight(text: str) -> str:
    """Return the message as HTML with each fingerprint match marked in its colour.
    Text is HTML-escaped first, so user input can never inject markup."""
    spans = sorted((m.start(), m.end(), name) for name in FINGERPRINTS for m in PATTERNS[name].finditer(text))
    out, pos = [], 0
    for start, end, name in spans:
        if start < pos:  # overlaps an earlier match (e.g. "WON" is CAPS and urgency): keep the first
            continue
        colour = FINGERPRINT_STYLE[name][1]
        out.append(html.escape(text[pos:start]))
        out.append(f'<mark style="background:{colour}">{html.escape(text[start:end])}</mark>')
        pos = end
    out.append(html.escape(text[pos:]))
    return "".join(out)


OTHER_LABELS = {"n_chars": "Length", "n_words": "Word count", "digit_ratio": "Digit share",
                "pos_vb": "Verb share", "pos_cd": "Number share", "pos_nnp": "Proper-noun share"}


def readable(name: str, X) -> str:
    """'text__claim' -> '"claim"'; 'numeric__n_phones' -> 'Phone number = 0' (value in this message)."""
    kind, feature = name.split("__", 1)
    if kind == "text":
        return f'"{feature}"'
    label = FINGERPRINT_STYLE[feature][0] if feature in FINGERPRINT_STYLE else OTHER_LABELS[feature]
    value = X[feature].iloc[0]
    return f"{label} = {value:.2f}" if isinstance(value, float) else f"{label} = {value}"


def top_contributions(model, X, n=5):
    """Words/features that pushed this message toward spam (+) or ham (-): weight x feature value.
    Numeric features are scaled, so a count of 0 (below average) can push toward ham."""
    features, clf = model.named_steps["features"], model.named_steps["clf"]
    values = features.transform(X).toarray()[0] * clf.coef_[0]
    names = [readable(name, X) for name in features.get_feature_names_out()]
    ranked = sorted((v, name) for v, name in zip(values, names) if v != 0)
    return ranked[::-1][:n], ranked[:n]  # (toward spam, toward ham)


def set_message(text: str):
    st.session_state.message = text


# ---------------------------------------------------------------- page
st.set_page_config(page_title="SMS Spam Detector", page_icon="🛡️", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🛡️ SMS Spam & Phishing Detector</h1>'
            "<p>Regex fingerprints + NLP features + TF-IDF words → Logistic Regression · "
            "trained on 5,158 UCI SMS messages</p></div>", unsafe_allow_html=True)

# Test-set results from the notebook (section 12).
m1, m2, m3, m4 = st.columns(4)
m1.metric("Spam F1 (test)", "0.952")
m2.metric("Precision (test)", "98.3%")
m3.metric("Recall (test)", "92.2%")
m4.metric("Regex rules alone, F1", "0.848")

left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("Message")
    for group, examples in EXAMPLES.items():
        st.caption(group.upper())
        cols = st.columns(2)
        for i, (label, text) in enumerate(examples.items()):
            cols[i % 2].button(label, on_click=set_message, args=(text,), width="stretch")
    message = st.text_area("SMS message", key="message", height=140, label_visibility="collapsed",
                           placeholder="Paste or type an SMS, then press Ctrl/⌘ + Enter…")

with right:
    st.subheader("Analysis")
    if not message.strip():
        st.info("Pick an example or type a message to analyse it.")
    else:
        model = load_model()
        X = build_input(message)
        p_spam = float(model.predict_proba(X)[0, 1])
        fired = [name for name in FINGERPRINTS if PATTERNS[name].search(message)]

        is_spam = p_spam >= 0.5
        risk = "High" if p_spam >= 0.7 else "Medium" if p_spam >= 0.3 else "Low"
        colour = "#EF4444" if is_spam else "#22C55E"
        st.markdown(f'<div class="card"><div class="muted">ML model verdict · risk {risk}</div>'
                    f'<div class="verdict" style="color:{colour}">{"🚨 SPAM" if is_spam else "✅ HAM"} · {p_spam:.0%}</div>'
                    f'<div class="muted">probability that this message is spam</div></div>',
                    unsafe_allow_html=True)
        st.progress(p_spam)

        rule_spam = len(fired) >= RULE_K
        c1, c2 = st.columns(2)
        c1.metric("Regex rule (Model 0)", "Spam" if rule_spam else "Ham", f"{len(fired)}/7 fingerprints",
                  delta_color="off", delta_arrow="off")
        c2.metric("ML model (Model 3)", "Spam" if is_spam else "Ham", f"{p_spam:.0%} spam", delta_color="off",
                  delta_arrow="off")
        if rule_spam != is_spam:
            st.caption("The two disagree: the words carry signal the regex fingerprints alone miss.")

if message.strip():
    st.divider()
    st.subheader("Why")
    chips = "".join(
        f'<span class="chip on" style="background:{c}">{label}</span>' if name in fired
        else f'<span class="chip">{label}</span>'
        for name, (label, c) in FINGERPRINT_STYLE.items())
    st.markdown(f'<div class="card"><div class="muted">Regex fingerprints in the raw message</div>{chips}'
                f'<div class="msg">{highlight(message)}</div></div>', unsafe_allow_html=True)

    toward_spam, toward_ham = top_contributions(model, X)
    s, h = st.columns(2)
    s.markdown("**Pushing toward spam**")
    for value, name in toward_spam:
        if value > 0:
            s.progress(min(value / 5, 1.0), text=f"{name}  (+{value:.2f})")
    h.markdown("**Pushing toward ham**")
    for value, name in toward_ham:
        if value < 0:
            h.progress(min(-value / 5, 1.0), text=f"{name}  ({value:.2f})")

    with st.expander("How the message was processed"):
        cleaned = clean_text(message)
        for title, value in [("1. Raw: regex + POS features are computed here", message),
                             ("2. Cleaned: encoding, placeholders, slang, lowercase", cleaned),
                             ("3. Tokens: stopwords removed, lemmatized", " ".join(process_tokens(cleaned)))]:
            st.caption(title)
            st.code(value, language=None, wrap_lines=True)

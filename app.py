"""Streamlit demo: classify an SMS as spam or ham and explain why.

Run locally with:  streamlit run app.py
Styling lives in assets/style.css.
"""
import html
from pathlib import Path

import nltk
import streamlit as st

# NLTK data must exist before features.py is imported (it loads the stopword list on import).
for package in ["punkt_tab", "stopwords", "wordnet", "averaged_perceptron_tagger_eng"]:
    nltk.download(package, quiet=True)

import joblib  # noqa: E402

from features import FINGERPRINTS, PATTERNS, build_input, clean_text, process_tokens  # noqa: E402

MODEL_PATH = "models/model3.joblib"
CSS_PATH = Path(__file__).parent / "assets" / "style.css"
RULE_K = 3  # Model 0: spam if at least 3 distinct fingerprints fire (k tuned in the notebook)
SPAM, HAM = "#E11D48", "#059669"

# Fingerprint -> (label, highlight colour, dot colour)
FINGERPRINT_STYLE = {
    "n_urls": ("Link", "#DBEAFE", "#3B82F6"),
    "n_phones": ("Phone number", "#FFE4E6", "#F43F5E"),
    "n_shortcodes": ("Short code", "#FFEDD5", "#F97316"),
    "n_money": ("Money", "#D1FAE5", "#10B981"),
    "n_caps": ("CAPS", "#EDE9FE", "#8B5CF6"),
    "n_urgent": ("Urgency", "#FEF3C7", "#F59E0B"),
    "n_exclaim": ("Exclamation", "#FCE7F3", "#EC4899"),
}
OTHER_LABELS = {"n_chars": "Length", "n_words": "Word count", "digit_ratio": "Digit share",
                "pos_vb": "Verb share", "pos_cd": "Number share", "pos_nnp": "Proper-noun share"}

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
        out.append(html.escape(text[pos:start]))
        out.append(f'<mark style="background:{FINGERPRINT_STYLE[name][1]}">{html.escape(text[start:end])}</mark>')
        pos = end
    out.append(html.escape(text[pos:]))
    return "".join(out)


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
    return [r for r in ranked[::-1][:n] if r[0] > 0], [r for r in ranked[:n] if r[0] < 0]


def gauge(p: float, colour: str) -> str:
    """Circular SVG gauge showing the spam probability."""
    circumference = 2 * 3.14159 * 54
    return (f'<div class="gauge"><svg width="132" height="132" viewBox="0 0 132 132">'
            f'<circle cx="66" cy="66" r="54" fill="none" stroke="#EEF0F5" stroke-width="12"/>'
            f'<circle cx="66" cy="66" r="54" fill="none" stroke="{colour}" stroke-width="12" stroke-linecap="round" '
            f'stroke-dasharray="{p * circumference:.1f} {circumference:.1f}"/></svg>'
            f'<div class="num"><div><b>{p:.0%}</b><small>spam</small></div></div></div>')


def signal_rows(items, colour) -> str:
    """Rows of name + bar scaled to the contribution (full bar at 5)."""
    return "".join(
        f'<div class="sig"><span>{html.escape(name)}</span><div class="track">'
        f'<div style="width:{max(min(abs(v) / 5, 1), .04) * 100:.0f}%;background:{colour}"></div></div></div>'
        for v, name in items) or '<div class="sig"><span>—</span></div>'


def pick_example(group: str):
    """Pill callback: load and analyse the chosen example, clear the other groups' selection."""
    choice = st.session_state[f"pick_{group}"]
    if choice:
        st.session_state.message = st.session_state.analysed = EXAMPLES[group][choice]
        for other in EXAMPLES:
            if other != group:
                st.session_state[f"pick_{other}"] = None


def analyse():
    """Analyse button: analyse the typed text and clear any example selection."""
    st.session_state.analysed = st.session_state.message
    for group in EXAMPLES:
        st.session_state[f"pick_{group}"] = None


# ---------------------------------------------------------------- page
st.set_page_config(page_title="SMS Spam Detector", page_icon="🛡️", layout="wide")
st.markdown(f"<style>{CSS_PATH.read_text()}</style>", unsafe_allow_html=True)
st.session_state.setdefault("analysed", "")

# Test-set results from the notebook (section 12).
st.markdown(
    '<div class="topbar"><div class="brand"><div class="logo">🛡️</div><div>'
    '<div class="brand-name">SMS Spam Detector</div><div class="brand-tag">Spam &amp; phishing detection · NLP capstone</div>'
    '</div></div><div class="stats"><div class="stat">F1 <b>0.952</b></div><div class="stat">Precision <b>98.3%</b></div>'
    '<div class="stat">Recall <b>92.2%</b></div><div class="stat">5,158 training SMS</div></div></div>'
    '<div class="hero"><h1>Is this text message <span class="accent">a scam?</span></h1>'
    "<p>Regex fingerprints, NLP features and TF-IDF words feed a Logistic Regression model. "
    "Paste a message or try an example to see the verdict and exactly what drove it.</p></div>",
    unsafe_allow_html=True)

left, right = st.columns([1, 1], gap="large")

with left, st.container(border=True):
    st.markdown('<div class="card-title">Message</div>', unsafe_allow_html=True)
    st.text_area("Message", key="message", height=150, label_visibility="collapsed",
                 placeholder="Paste or type an SMS…")
    st.button("Analyse message", type="primary", on_click=analyse, width="stretch")
    st.markdown('<div class="card-title" style="margin-top:.8rem">Try an example</div>', unsafe_allow_html=True)
    for group, examples in EXAMPLES.items():
        st.pills(group, list(examples), key=f"pick_{group}", on_change=pick_example, args=(group,))

message = st.session_state.analysed.strip()

with right, st.container(border=True):
    st.markdown('<div class="card-title">Verdict</div>', unsafe_allow_html=True)
    if not message:
        st.markdown('<div class="empty"><div class="icon">💬</div><b>No message analysed yet</b>'
                    "Type a message and press Analyse, or pick an example.</div>", unsafe_allow_html=True)
    else:
        model = load_model()
        X = build_input(message)
        p_spam = float(model.predict_proba(X)[0, 1])
        fired = [name for name in FINGERPRINTS if PATTERNS[name].search(message)]
        is_spam, rule_spam = p_spam >= 0.5, len(fired) >= RULE_K
        colour = SPAM if is_spam else HAM
        risk = "High" if p_spam >= 0.7 else "Medium" if p_spam >= 0.3 else "Low"
        badge_bg = "#FFE4E6" if is_spam else "#D1FAE5"
        title = "Likely spam or phishing" if is_spam else "Looks legitimate"
        note = ('<div class="note">💡 The regex rule and the ML model disagree: '
                "the words catch what the patterns alone miss.</div>") if rule_spam != is_spam else ""

        st.markdown(
            f'<div class="result">{gauge(p_spam, colour)}<div>'
            f'<span class="badge" style="background:{badge_bg};color:{colour}">{"🚨 SPAM" if is_spam else "✅ NOT SPAM"}</span>'
            f'<div class="verdict-title">{title}</div><div class="verdict-sub">{risk} risk · Logistic Regression (Model 3)</div>'
            "</div></div>"
            f'<div class="compare"><div class="tile"><div class="t">Regex rule · Model 0</div>'
            f'<div class="v" style="color:{SPAM if rule_spam else HAM}">{"Spam" if rule_spam else "Not spam"}</div>'
            f'<div class="d">{len(fired)} of 7 fingerprints fired</div></div>'
            f'<div class="tile"><div class="t">ML model · Model 3</div>'
            f'<div class="v" style="color:{colour}">{"Spam" if is_spam else "Not spam"}</div>'
            f'<div class="d">{p_spam:.1%} spam probability</div></div></div>{note}',
            unsafe_allow_html=True)

if message:
    a, b = st.columns([1, 1], gap="large")
    with a, st.container(border=True):
        chips = "".join(
            f'<span class="chip on"><i style="background:{dot}"></i>{label}</span>' if name in fired
            else f'<span class="chip"><i></i>{label}</span>'
            for name, (label, _, dot) in FINGERPRINT_STYLE.items())
        st.markdown(f'<div class="card-title">Fingerprints in the message</div>'
                    f'<div class="bubble">{highlight(message)}</div><div class="chips">{chips}</div>',
                    unsafe_allow_html=True)

    with b, st.container(border=True):
        toward_spam, toward_ham = top_contributions(model, X)
        st.markdown(
            '<div class="card-title">What drove the decision</div><div class="signals">'
            f'<div><div class="sig-head" style="color:{SPAM}">▲ Toward spam</div>{signal_rows(toward_spam, SPAM)}</div>'
            f'<div><div class="sig-head" style="color:{HAM}">▼ Toward not spam</div>{signal_rows(toward_ham, HAM)}</div>'
            "</div>", unsafe_allow_html=True)

    with st.expander("How the message was processed (NLP pipeline)"):
        cleaned = clean_text(message)
        steps = [("Raw text", "Regex fingerprints and POS features are computed here", message),
                 ("Cleaned", "Encoding fixed, URLs/money/phones replaced, slang mapped, lowercased", cleaned),
                 ("Tokens", "Tokenized, stopwords removed, lemmatized → TF-IDF", " ".join(process_tokens(cleaned)))]
        st.markdown("".join(
            f'<div class="step"><div class="n">{i}</div><div style="min-width:0;flex:1"><div class="t">{t}</div>'
            f'<div class="d">{d}</div><code>{html.escape(v)}</code></div></div>'
            for i, (t, d, v) in enumerate(steps, 1)), unsafe_allow_html=True)

st.markdown('<div class="footer">Trained on the UCI SMS Spam Collection (Almeida &amp; Hidalgo, 2011) · '
            "English SMS only · classic NLP + scikit-learn</div>", unsafe_allow_html=True)

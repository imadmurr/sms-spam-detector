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

# Fingerprint -> (label, soft highlight colour)
FINGERPRINT_STYLE = {
    "n_urls": ("Link", "#DBEAFE"),
    "n_phones": ("Phone", "#FEE2E2"),
    "n_shortcodes": ("Short code", "#FFEDD5"),
    "n_money": ("Money", "#DCFCE7"),
    "n_caps": ("CAPS", "#EDE9FE"),
    "n_urgent": ("Urgency", "#FEF3C7"),
    "n_exclaim": ("!", "#FCE7F3"),
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

CSS = """
<style>
.block-container {max-width: 1180px; padding-top: 3.2rem; padding-bottom: 1rem;}
h1 {font-size: 1.7rem !important; font-weight: 700; letter-spacing: -0.02em; padding: 0 !important;}
.sub {color: #6B7280; font-size: .9rem; margin: -.4rem 0 1.2rem;}
.card {border: 1px solid #E5E7EB; border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: .8rem;}
.label {color: #6B7280; font-size: .75rem; text-transform: uppercase; letter-spacing: .06em; margin-bottom: .35rem;}
.verdict {font-size: 1.9rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1.2;}
.bar {height: 6px; background: #F3F4F6; border-radius: 99px; margin: .6rem 0 .7rem;}
.bar > div {height: 100%; border-radius: 99px;}
.models {font-size: .9rem; color: #374151;}
.models b {font-weight: 600;}
.msg {line-height: 1.9; font-size: .98rem; color: #111827;}
.msg mark {color: #111827; border-radius: 4px; padding: 1px 4px;}
.chip {display: inline-block; border-radius: 99px; padding: 1px 9px; margin: 0 4px 6px 0; font-size: .75rem;
       color: #9CA3AF; border: 1px solid #E5E7EB;}
.chip.on {color: #111827; border-color: transparent;}
.contrib {display: grid; grid-template-columns: 1fr 1fr; gap: 0 1.4rem;}
.row {display: grid; grid-template-columns: 1fr 70px; align-items: center; gap: 8px; font-size: .82rem;
      color: #374151; padding: 2px 0;}
.row span {overflow: hidden; text-overflow: ellipsis; white-space: nowrap;}
.mini {height: 4px; background: #F3F4F6; border-radius: 99px;}
.mini > div {height: 100%; border-radius: 99px;}
[data-testid="stButtonGroup"] > div {flex-wrap: wrap;}  /* example pills wrap on narrow screens */
@media (max-width: 640px) {.contrib {grid-template-columns: 1fr;}}
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
        out.append(html.escape(text[pos:start]))
        out.append(f'<mark style="background:{FINGERPRINT_STYLE[name][1]}">{html.escape(text[start:end])}</mark>')
        pos = end
    out.append(html.escape(text[pos:]))
    return "".join(out)


def readable(name: str, X) -> str:
    """'text__claim' -> '"claim"'; 'numeric__n_phones' -> 'Phone = 0' (value in this message)."""
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


def contrib_rows(items, colour) -> str:
    """Compact HTML rows: name + thin bar scaled to the contribution (full bar at 5)."""
    return "".join(
        f'<div class="row"><span>{html.escape(name)}</span>'
        f'<div class="mini"><div style="width:{min(abs(v) / 5, 1) * 100:.0f}%;background:{colour}"></div></div></div>'
        for v, name in items) or '<div class="row"><span>—</span></div>'


def pick_example(group: str):
    """Pill callback: load the chosen example and clear the other groups' selection."""
    choice = st.session_state[f"pick_{group}"]
    if choice:
        st.session_state.message = EXAMPLES[group][choice]
        for other in EXAMPLES:
            if other != group:
                st.session_state[f"pick_{other}"] = None


# ---------------------------------------------------------------- page
st.set_page_config(page_title="SMS Spam Detector", page_icon="🛡️", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
st.title("🛡️ SMS Spam & Phishing Detector")
# Test-set results from the notebook (section 12).
st.markdown('<div class="sub">Regex fingerprints + NLP features + TF-IDF → Logistic Regression · '
            "test F1 0.952 · precision 98.3% · regex rules alone F1 0.848</div>", unsafe_allow_html=True)

left, right = st.columns([5, 6], gap="large")

with left:
    for group, examples in EXAMPLES.items():
        st.pills(group, list(examples), key=f"pick_{group}", on_change=pick_example, args=(group,))
    message = st.text_area("Message", key="message", height=130,
                           placeholder="Paste or type an SMS, then press Ctrl/⌘ + Enter…")
    if message.strip():
        with st.expander("How the message was processed"):
            cleaned = clean_text(message)
            for title, value in [("Raw: regex + POS features", message),
                                 ("Cleaned: encoding, placeholders, slang, lowercase", cleaned),
                                 ("Tokens: stopwords removed, lemmatized", " ".join(process_tokens(cleaned)))]:
                st.caption(title)
                st.code(value, language=None, wrap_lines=True)

with right:
    if not message.strip():
        st.markdown('<div class="card"><div class="label">Result</div>'
                    '<div class="models">Pick an example or type a message.</div></div>', unsafe_allow_html=True)
    else:
        model = load_model()
        X = build_input(message)
        p_spam = float(model.predict_proba(X)[0, 1])
        fired = [name for name in FINGERPRINTS if PATTERNS[name].search(message)]
        is_spam, rule_spam = p_spam >= 0.5, len(fired) >= RULE_K
        colour = "#DC2626" if is_spam else "#16A34A"
        risk = "high" if p_spam >= 0.7 else "medium" if p_spam >= 0.3 else "low"
        note = (" · <i>they disagree: the words carry signal the fingerprints miss</i>"
                if is_spam != rule_spam else "")

        st.markdown(
            f'<div class="card"><div class="label">Result · {risk} risk</div>'
            f'<div class="verdict" style="color:{colour}">{"Spam" if is_spam else "Not spam"} · {p_spam:.0%}</div>'
            f'<div class="bar"><div style="width:{p_spam * 100:.0f}%;background:{colour}"></div></div>'
            f'<div class="models">Regex rule: <b>{"Spam" if rule_spam else "Ham"}</b> ({len(fired)}/7) · '
            f'ML model: <b>{"Spam" if is_spam else "Ham"}</b> ({p_spam:.0%}){note}</div></div>',
            unsafe_allow_html=True)

        chips = "".join(
            f'<span class="chip on" style="background:{c}">{label}</span>' if name in fired
            else f'<span class="chip">{label}</span>'
            for name, (label, c) in FINGERPRINT_STYLE.items())
        toward_spam, toward_ham = top_contributions(model, X)
        st.markdown(
            f'<div class="card"><div class="label">Why · fingerprints</div>{chips}'
            f'<div class="msg">{highlight(message)}</div></div>'
            f'<div class="card"><div class="contrib">'
            f'<div><div class="label">Toward spam</div>{contrib_rows(toward_spam, "#DC2626")}</div>'
            f'<div><div class="label">Toward ham</div>{contrib_rows(toward_ham, "#16A34A")}</div>'
            f"</div></div>",
            unsafe_allow_html=True)

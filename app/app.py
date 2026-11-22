import re
import time
import streamlit as st

from src.generation.generate import generate_activity

st.set_page_config(
    page_title="YouthActivity — Multi-Agent Spelgenerator",
    page_icon="🕵️",
    layout="wide",
)

# Visuele kleurconfiguratie per agent
AGENT_STYLES = {
    "rules": {
        "name": "Game Rules Agent",
        "color": "#1D4ED8",      # Blauw
        "bg": "#EFF6FF",
        "badge": "🟦 Game Rules Agent",
        "role": "Haalt uitsluitend groepsindeling, winconditie, spelstappen en echte materialen uit de bronnen ([Spel N]) — zonder personen/tikkers bij materialen of verzinsels.",
    },
    "story": {
        "name": "Storyteller Agent",
        "color": "#7E22CE",      # Paars
        "bg": "#FAF5FF",
        "badge": "🟪 Storyteller Agent",
        "role": "Schrijft een kort kaderverhaal aansluitend bij het spel en zet eigen creatieve aankleding apart onder 'Thema-aanpassing (zelf verzonnen)'.",
    },
    "safety": {
        "name": "Safety & Age Agent",
        "color": "#15803D",      # Groen
        "bg": "#F0FDF4",
        "badge": "🟩 Safety & Age Agent",
        "role": "Scheidt letterlijke bron-veiligheid ([Spel N]) van praktische leeftijd- en terreinadviezen (zonder [Spel N]) voor de gekozen doelgroep en locatie.",
    },
}

PRESETS = {
    "Spionnen op het grasplein (11–14j)": {
        "use_age": True,
        "age": (11, 14),
        "terrein_optie": "grasplein / open terrein",
        "k": 2,
        "strict_age": False,
        "verify": False,
        "brief": "actief estafettespel in teams met geheime opdrachten en codes kraken",
        "theme": "Spionnenmissie & Geheime Code",
    },
    "Bosavontuur voor jongsten (8–10j)": {
        "use_age": True,
        "age": (8, 10),
        "terrein_optie": "bos",
        "k": 2,
        "strict_age": True,
        "verify": False,
        "brief": "buiten rennen en opdrachten uitvoeren in groepjes",
        "theme": "Bosdetectives",
    },
    "Binnen / lokaal spel voor ouderen (14–16j)": {
        "use_age": True,
        "age": (14, 16),
        "terrein_optie": "binnen / lokaal",
        "k": 2,
        "strict_age": False,
        "verify": False,
        "brief": "samenwerkings- en tactisch spel met kleine teams en tijdsdruk",
        "theme": "Escape & Overleving",
    },
}


def _agent_card_header(style: dict) -> None:
    st.markdown(
        f"""
        <div style="
            border-left: 6px solid {style['color']};
            background-color: {style['bg']};
            padding: 10px 14px;
            border-radius: 6px;
            margin-bottom: 10px;
        ">
            <div style="font-weight: 700; color: {style['color']}; font-size: 1rem;">
                {style['badge']}
            </div>
            <div style="color: #1F2937; font-size: 0.88rem; margin-top: 2px;">
                {style['role']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _split_by_agent(output: str) -> dict[str, list[str]]:
    """Verdeel de gegenereerde tekst over de drie agents op basis van koppen."""
    buckets: dict[str, list[str]] = {"rules": [], "story": [], "safety": [], "meta": []}
    current = "rules"

    for line in output.splitlines():
        clean_heading = re.sub(r"[*_#]", "", line).strip().lower()
        if clean_heading.startswith("verhaal") or "thema-aanpassing" in clean_heading or "thema‑aanpassing" in clean_heading:
            current = "story"
        elif "veiligheidspunten" in clean_heading or "leeftijds- en terreininschatting" in clean_heading:
            current = "safety"
        elif clean_heading.startswith("bronnen") or clean_heading.startswith("controle"):
            current = "meta"
        elif clean_heading in {"titel", "groepsindeling en winconditie", "spelregels in stappen", "materialen", "niet in de bronnen"}:
            current = "rules"

        buckets[current].append(line)

    return buckets


# Initialiseer standaardwaarden (totaal andere settings dan voorheen)
if "initialized" not in st.session_state:
    preset = PRESETS["Spionnen op het grasplein (11–14j)"]
    st.session_state.initialized = True
    st.session_state.use_age = preset["use_age"]
    st.session_state.age = preset["age"]
    st.session_state.terrein_optie = preset["terrein_optie"]
    st.session_state.k = preset["k"]
    st.session_state.strict_age = preset["strict_age"]
    st.session_state.verify = preset["verify"]
    st.session_state.brief_input = preset["brief"]
    st.session_state.theme_input = preset["theme"]


def _apply_preset(name: str) -> None:
    p = PRESETS[name]
    st.session_state.use_age = p["use_age"]
    st.session_state.age = p["age"]
    st.session_state.terrein_optie = p["terrein_optie"]
    st.session_state.k = p["k"]
    st.session_state.strict_age = p["strict_age"]
    st.session_state.verify = p["verify"]
    st.session_state.brief_input = p["brief"]
    st.session_state.theme_input = p["theme"]


st.title("🕵️ YouthActivity — Multi-Agent Spelgenerator")
st.caption("RAG + gespecialiseerde agents voor jeugdbewegingsactiviteiten (Chiro-bronnen)")

# Overzichtsbalk van de 3 agents in kleur
st.markdown("### Hoe werken de agents samen?")
c1, c2, c3 = st.columns(3)
with c1:
    _agent_card_header(AGENT_STYLES["rules"])
with c2:
    _agent_card_header(AGENT_STYLES["story"])
with c3:
    _agent_card_header(AGENT_STYLES["safety"])

st.divider()

# Sneltest-presets bovenaan
st.markdown("**Snelle test-scenario's:**")
preset_cols = st.columns(len(PRESETS))
for idx, preset_name in enumerate(PRESETS):
    with preset_cols[idx]:
        if st.button(preset_name, use_container_width=True):
            _apply_preset(preset_name)
            st.rerun()

# Zijbalk met totaal andere standaardinstellingen
with st.sidebar:
    st.header("Filters & instellingen")

    use_age = st.checkbox("Leeftijdsfilter gebruiken", key="use_age")
    if use_age:
        min_age, max_age = st.slider(
            "Leeftijdscategorie (jaar)",
            min_value=6,
            max_value=18,
            key="age",
        )
    else:
        min_age, max_age = None, None

    terrein_optie = st.selectbox(
        "Terrein",
        options=["grasplein / open terrein", "bos", "binnen / lokaal", "alle terreinen"],
        key="terrein_optie",
    )
    terrein = None if terrein_optie == "alle terreinen" else terrein_optie.split(" / ")[0]

    k = st.slider("Aantal bronspellen (k)", min_value=1, max_value=4, key="k")
    strict_age = st.checkbox("Strikte leeftijdsgrens (strict_age)", key="strict_age")
    verify = st.checkbox("Bronverwijzingen nakijken (verify)", key="verify")

brief_input = st.text_area(
    "Activiteitenbrief / opdracht",
    key="brief_input",
    height=85,
)

theme_input = st.text_input(
    "Gewenst thema (optioneel)",
    key="theme_input",
    placeholder="Bijv. Spionnenmissie, Piraten, Ruimtereis...",
)

if st.button("Genereer activiteit", type="primary"):
    if not brief_input.strip():
        st.warning("Vul eerst een activiteitenbrief in.")
    else:
        full_brief = brief_input.strip()
        if theme_input.strip():
            full_brief = f"{full_brief} | gewenst thema: {theme_input.strip()}"
        if min_age is not None and max_age is not None:
            full_brief = f"{full_brief} | leeftijd: {min_age}-{max_age} jaar"
        if terrein:
            full_brief = f"{full_brief} | terrein: {terrein}"

        with st.spinner("1. GameRulesAgent → 2. StorytellerAgent & SafetyAgeAgent (parallel)..."):
            t0 = time.perf_counter()
            output = generate_activity(
                brief=full_brief,
                k=k,
                min_age=min_age,
                max_age=max_age,
                terrein=terrein,
                strict_age=strict_age,
                verify=verify,
            )
            elapsed = time.perf_counter() - t0

        st.success(f"Klaar in **{elapsed:.1f} seconden**.")

        sections = _split_by_agent(output)

        if sections["rules"]:
            _agent_card_header(AGENT_STYLES["rules"])
            st.markdown("\n".join(sections["rules"]).strip())
            st.markdown("---")

        if sections["story"]:
            _agent_card_header(AGENT_STYLES["story"])
            st.markdown("\n".join(sections["story"]).strip())
            st.markdown("---")

        if sections["safety"]:
            _agent_card_header(AGENT_STYLES["safety"])
            st.markdown("\n".join(sections["safety"]).strip())
            st.markdown("---")

        if sections["meta"]:
            st.markdown("\n".join(sections["meta"]).strip())

        with st.expander("Bekijk volledige ruwe Markdown-output"):
            st.code(output, language="markdown")

        st.download_button(
            label="Download als Markdown (.md)",
            data=output,
            file_name="youthactivity_spionnenmissie.md",
            mime="text/markdown",
        )
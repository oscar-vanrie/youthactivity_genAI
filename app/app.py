import time
import streamlit as st

from src.generation.generate import generate_activity

st.set_page_config(
    page_title="YouthActivity — Multi-Agent Spelgenerator",
    page_icon="🌲",
    layout="wide",
)

st.title("🌲 YouthActivity")
st.caption("RAG + Multi-Agent activiteitengenerator voor jeugdbewegingen (Chiro-bronnen)")

# Zijbalk met filters & instellingen (sluit aan op search_games() en generate_activity())
with st.sidebar:
    st.header("Filters & instellingen")

    use_age = st.checkbox("Leeftijdsfilter gebruiken", value=True)
    if use_age:
        min_age, max_age = st.slider(
            "Leeftijdscategorie (jaar)",
            min_value=6,
            max_value=18,
            value=(8, 10),
        )
    else:
        min_age, max_age = None, None

    terrein_optie = st.selectbox(
        "Terrein",
        options=["bos", "grasplein / open terrein", "binnen / lokaal", "alle terreinen"],
        index=0,
    )
    terrein = None if terrein_optie == "alle terreinen" else terrein_optie.split(" / ")[0]

    k = st.slider("Aantal bronspellen (k)", min_value=1, max_value=4, value=2)
    strict_age = st.checkbox("Strikte leeftijdsgrens (strict_age)", value=True)
    verify = st.checkbox("Bronverwijzingen nakijken (verify)", value=False)

# Hoofdinvoer: brief + optioneel thema uit het demo-scenario
default_brief = "buiten rennen en opdrachten uitvoeren in groepjes"
brief_input = st.text_area(
    "Activiteitenbrief / opdracht",
    value=default_brief,
    height=90,
    help="Beschrijf wat voor spel, sfeer of opdrachten je zoekt.",
)

theme_input = st.text_input(
    "Gewenst thema (optioneel)",
    value="",
    placeholder="Bijv. Space / Alien Invasion, bosdetectives, piraten...",
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

        with st.spinner("Agents halen bronspellen op en schrijven de activiteit..."):
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
        st.markdown(output)
        st.download_button(
            label="Download als Markdown (.md)",
            data=output,
            file_name="youthactivity_activiteit.md",
            mime="text/markdown",
        )
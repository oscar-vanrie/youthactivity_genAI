import os
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
    "Demo Proposal: Space & Alien Invasion (8–10j, bos)": {
        "use_age": True,
        "age": (8, 10),
        "terrein_optie": "bos",
        "k": 2,
        "strict_age": True,
        "verify": False,
        "brief": "2 uur durend bosspel in groepjes met smokkel- en opdrachtenmechaniek",
        "theme": "Space / Alien Invasion — Vind de verloren raketonderdelen",
    },
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
    """Toon een gekleurde agent-kop met badge, naam en rol."""
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
        clean_heading = re.sub(r"[\u2010-\u2015\u2212]", "-", clean_heading)
        if clean_heading.startswith("verhaal") or "thema-aanpassing" in clean_heading:
            current = "story"
        elif "veiligheidspunten" in clean_heading or "leeftijds- en terreininschatting" in clean_heading:
            current = "safety"
        elif clean_heading.startswith("bronnen") or clean_heading.startswith("controle"):
            current = "meta"
        elif clean_heading in {"titel", "groepsindeling en winconditie", "spelregels in stappen", "materialen", "niet in de bronnen"}:
            current = "rules"

        buckets[current].append(line)

    return buckets


# Registreer een Unicode TTF-font zodat speciale tekens (streepjes, trema's,
# enz.) correct renderen zonder encoding-crash. Core-fonts (Helvetica) ondersteunen
# deze tekens niet, dus zoeken we eerst een systeemfont op.
_FONT_CANDIDATES = [
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
    (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ),
    ("/Library/Fonts/Arial.ttf", "/Library/Fonts/Arial Bold.ttf"),
]


def _register_pdf_fonts(pdf) -> str:
    """Registreer een Unicode (regular + bold) fontpaar en geef de familienaam terug."""
    for regular, bold in _FONT_CANDIDATES:
        if os.path.exists(regular) and os.path.exists(bold):
            try:
                pdf.add_font("App", "", regular)
                pdf.add_font("App", "B", bold)
                return "App"
            except Exception:
                continue
    return "Helvetica"


def _pdf_write_line(pdf, family: str, text: str) -> None:
    """Schrijf een alinea met basisopmaak (bold-streepjes worden verwijderd)."""
    text = text.replace("**", "")
    pdf.multi_cell(0, 6, text, new_x="LMARGIN", new_y="NEXT")


def _markdown_to_pdf(output: str) -> bytes:
    """Zet de ruwe Markdown-output om naar een netjes geformatteerde PDF (bytes)."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    family = _register_pdf_fonts(pdf)

    # Titel uit de eerste # kop
    lines = output.splitlines()

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()

        if not stripped:
            pdf.ln(3)
            continue

        # Horizontale lijn / scheiding
        if re.fullmatch(r"-{3,}|\*{3,}|_{3,}", stripped):
            pdf.ln(1)
            pdf.set_draw_color(180, 180, 180)
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
            pdf.ln(2)
            continue

        # Koppen
        heading_match = re.match(r"^(#{1,6})\s+(.*)", stripped)
        if heading_match:
            level = len(heading_match.group(1))
            size = {1: 16, 2: 13, 3: 12}.get(level, 11)
            pdf.set_font(family, "B", size)
            pdf.set_text_color(31, 41, 55)
            pdf.multi_cell(0, size * 0.6, heading_match.group(2), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
            continue

        # Bulletpoints
        bullet_match = re.match(r"^[-*+]\s+(.*)", stripped)
        if bullet_match:
            pdf.set_font(family, "", 11)
            pdf.set_text_color(31, 41, 55)
            pdf.set_x(pdf.l_margin + 5)
            pdf.cell(5, 6, "•", new_x="RIGHT", new_y="TOP")
            pdf.multi_cell(
                0, 6, bullet_match.group(1).replace("**", ""), new_x="LMARGIN", new_y="NEXT"
            )
            continue

        # Gewone tekst (inclusief bronvermeldingen zoals [Spel N])
        pdf.set_font(family, "", 11)
        pdf.set_text_color(31, 41, 55)
        _pdf_write_line(pdf, family, line)

    # Encodeer naar bytes; vervang eventuele resterende niet-renderbare tekens zodat
    # de download nooit crasht op onverwachte tekens.
    return bytes(pdf.output())


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
    """Laad de instellingen van een preset in de sessiestatus."""
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

def _render_activity(output: str) -> None:
    """Render de gegenereerde activiteit (agent-kleuren, downloads) uit de ruwe output."""
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

    dl_md, dl_pdf = st.columns(2)
    with dl_md:
        st.download_button(
            label="Download als Markdown (.md)",
            data=output,
            file_name="youthactivity_spionnenmissie.md",
            mime="text/markdown",
        )
    with dl_pdf:
        try:
            pdf_bytes = _markdown_to_pdf(output)
            st.download_button(
                label="Download als PDF (.pdf)",
                data=pdf_bytes,
                file_name="youthactivity_spionnenmissie.pdf",
                mime="application/pdf",
            )
        except Exception as exc:  # nooit de app laten crashen door de PDF-export
            st.error(f"PDF-export is mislukt: {exc}")


def _to_png_bytes(data: bytes) -> bytes:
    """Zet eventuele JPEG/andere bytes om naar een geldige PNG (voor correcte download)."""
    try:
        from io import BytesIO

        from PIL import Image

        img = Image.open(BytesIO(data))
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return data


def _extract_title(output: str) -> str:
    """Haal een nette titel uit de output (eerste kop of **Titel**)."""
    match = re.search(r"(?m)^#+\s*(.+)$", output)
    if match:
        return match.group(1).strip()
    match = re.search(r"\*\*Titel\*\*\s*\n+\s*(.+)", output)
    if match:
        return match.group(1).strip()
    return ""


def _render_visual_section(output: str) -> None:
    """Seam-sectie voor een printbare thema-visual (badge / spelkaart)."""
    st.divider()
    st.markdown("### 🎨 Thema-visual (Badge / Spelkaart)")
    st.caption("Genereer optioneel een printbare thema-illustratie via een image-model.")

    title = _extract_title(output) or "Chiro spel"
    theme_value = st.session_state.get("theme_input", "").strip() or "jeugdbeweging"
    sections = _split_by_agent(output)
    story = "\n".join(sections["story"]).strip()[:800] or output[:800]

    if st.button("🎨 Genereer thema-visual", key="gen_visual"):
        st.session_state["visual_requested"] = True

    if st.session_state.get("visual_requested"):
        with st.spinner("Engelse image-prompt + afbeelding genereren..."):
            try:
                from src.generation.visuals import generate_activity_visual

                img_bytes, image_prompt = generate_activity_visual(
                    theme=theme_value,
                    title=title,
                    story=story,
                )
            except Exception as exc:  # de visual-sectie mag de app nooit breken
                img_bytes, image_prompt = None, f"(visual-module fout: {exc})"
        st.session_state["visual_bytes"] = img_bytes
        st.session_state["visual_prompt"] = image_prompt

    image_prompt = st.session_state.get("visual_prompt")
    img_bytes = st.session_state.get("visual_bytes")

    if image_prompt:
        with st.expander("Bekijk de gebruikte image-prompt"):
            st.code(image_prompt, language="text")

    if img_bytes:
        png_bytes = _to_png_bytes(img_bytes)
        st.image(png_bytes, caption=title, use_container_width=True)
        st.download_button(
            label="Download visual als PNG (.png)",
            data=png_bytes,
            file_name="youthactivity_visual.png",
            mime="image/png",
            key="dl_visual",
        )
    elif st.session_state.get("visual_requested"):
        st.info(
            "Geen afbeelding kunnen genereren (offline of geen image-API bereikbaar). "
            "De gebruikte prompt staat hierboven."
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

        # Bewaar de output zodat activiteit + visual ook na reruns (bv. visual-knop) zichtbaar blijven
        st.session_state["activity_output"] = output
        st.session_state["visual_bytes"] = None
        st.session_state["visual_prompt"] = None
        st.session_state["visual_requested"] = False

# Blijvende weergave van de laatst gegenereerde activiteit + bijhorende thema-visual
if st.session_state.get("activity_output"):
    _render_activity(st.session_state["activity_output"])
    _render_visual_section(st.session_state["activity_output"])
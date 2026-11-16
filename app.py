import re
import streamlit as st
from google import genai
from google.genai import types

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
MODEL_ID = "gemini-3.5-flash-lite"
TEMPERATURE = 0.3
MAX_OUTPUT_TOKENS = 280
MIN_WORDS = 15
MAX_WORDS = 120
MAX_REMEMBERED_QUESTIONS = 6

SYSTEM_INSTRUCTION = """
You are Paul von Hindenburg (born 2 October 1847 – died 2 August 1934), Prussian Generalfeldmarschall (Field Marshal) in World War I and second Reichspräsident (President) of Germany's Weimar Republic (1925–1934).

AUDIENCE & VOICE:
- Speak directly to U.S. 10th-grade high school students (ages 15–16).
- Sound like Paul von Hindenburg himself: formal, solemn, military, and disciplined—using the dignified cadence of an elder Prussian officer and head of state (e.g., "Young student," "As Field Marshal," "In service to the Reich").
- Keep sentences clear and understandable for a 10th-grade American reader. Briefly clarify German terms like "Reichstag" (German parliament), "Article 48" (emergency decree power), or "Reichskanzler" (Chancellor).

MANDATED HISTORICAL ANCHORS (BE FACT-DENSE):
Anchor every answer in specific dates, people, letters, laws, speeches, and court/parliamentary records—maintaining strict historical accuracy without excusing anti-democratic actions:
- WWI Command & Armistice (1914–1918): Battle of Tannenberg speech (31 Aug 1914); Third Supreme Command (OHL) with Erich Ludendorff (1916); Auxiliary Service Law (Hilfsdienstgesetz, Dec 1916); Spring Offensive (April 1918); Sept 29, 1918 Spa demand for an armistice; messages with Kaiser Wilhelm II on abdication (Nov 1918) and the Nov 11, 1918 Armistice with Allied & U.S. forces (Woodrow Wilson's notes).
- Weimar Parliamentary Testimony (1919): Nov 18, 1919 testimony before the Weimar National Assembly's 15th Investigative Committee (Reichstagsprotokolle), advancing the false "stab-in-the-back" (Dolchstoßlegende) claim drafted with Ludendorff.
- Presidency, Letters & Constitutional Courts (1925–1932): Elected President (1925; reelected 10 April 1932 over Adolf Hitler); letters with Chancellors Hermann Müller (March 1930) and Heinrich Brüning (Akten der Reichskanzlei) governing through Article 48 emergency decrees; July 20, 1932 Preußenschlag deposing Prussia's government, ruled partly unlawful by Germany's Staatsgerichtshof (constitutional court) in Preußen contra Reich (25 Oct 1932); Nov 1932 Industrialists' Petition (Industrielleneingabe) and letters from Ludendorff and Franz von Papen.
- 1933–1934 Dictatorship & Key Letters/Laws: Appointing Hitler Chancellor (30 Jan 1933); Potsdam Day speech (21 March 1933); signing the Reichstag Fire Decree (28 Feb 1933) and Enabling Act (24 March 1933); April 4, 1933 letter to Hitler protesting dismissal of Jewish WWI veterans before the Civil Service Law (7 April 1933), securing only a veteran exemption while accepting the purge; Reichskonkordat (20 July 1933); Law Concerning the Head of State (1 Aug 1934) merging the presidency into Hitler's office upon Hindenburg's death (2 Aug 1934), documented in Allied/U.S. Nuremberg & Congressional records.

MANDATORY SOURCES & CITATIONS:
Ground every response in primary and secondary sources—specifically letters written to and from Paul von Hindenburg, official government documents, speeches, Reichstag/parliamentary transcripts, U.S. Congressional/diplomatic records, court rulings (such as Preußen contra Reich or Nuremberg evidence), and legislation.
Every response MUST end with a brief citation naming the specific document/letter and at least one of these approved digital archives:
1. Library of Congress Digital Collections
2. HathiTrust Digital Library
3. German Federal Archives (Bundesarchiv - Invenio & Digitales Bildarchiv)
4. Akten der Reichskanzlei (Weimar Republic Edition)
5. Bavarian State Library (Bayerische Staatsbibliothek - Reichstagsprotokolle)
6. 100(0) Schlüsseldokumente zur deutschen Geschichte
7. 1914-1918-online (International Encyclopedia of the First World War)
8. Europeana 1914-1918
9. Filmportal.de
10. German Historical Institute Washington (GHDI)
11. Germany - The EHRI Portal

STRICT LENGTH & ACCURACY RULES:
- Every reply MUST be between 15 words minimum and 120 words maximum (target 65–105 words), INCLUDING the final "Source:" citation.
- Pack in real dates, names, and primary sources. Never invent facts or citations. Do not soften or justify authoritarian decrees—state honestly what the documents show. If asked about events after August 2, 1934, state plainly in character that your life ended on that date while citing an archive.
""".strip()


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
@st.cache_resource
def get_genai_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def enforce_word_limits(text: str, min_words: int = MIN_WORDS, max_words: int = MAX_WORDS) -> str:
    """Guarantees response stays strictly between 15 and 120 words."""
    cleaned = " ".join(text.split())
    words = cleaned.split()

    if len(words) > max_words:
        source_match = re.search(r"(Source:\s*.+)$", cleaned)
        source_tail = source_match.group(1) if source_match else ""
        source_words = source_tail.split() if source_tail else []

        if source_tail and len(source_words) < max_words - 15:
            body_text = cleaned[: source_match.start()].strip()
            budget = max_words - len(source_words)
            trimmed_body = " ".join(body_text.split()[:budget])
            last_punct = max(trimmed_body.rfind("."), trimmed_body.rfind("!"), trimmed_body.rfind("?"))
            if last_punct > 20:
                trimmed_body = trimmed_body[: last_punct + 1]
            cleaned = f"{trimmed_body} {source_tail}".strip()
        else:
            trimmed = " ".join(words[:max_words])
            last_punct = max(trimmed.rfind("."), trimmed.rfind("!"), trimmed.rfind("?"))
            cleaned = trimmed[: last_punct + 1] if last_punct > 20 else trimmed + "..."

    elif len(words) < min_words:
        cleaned = (
            f"{cleaned} The official Chancellery records and presidential decrees document these decisions clearly. "
            "Source: Akten der Reichskanzlei (Weimar Republic Edition)."
        ).strip()

    return cleaned


def build_limited_history(messages: list[dict], new_prompt: str) -> list[types.Content]:
    """
    Keeps ONLY the previous 6 user questions (and their 6 model answers)
    plus the current user question.
    """
    prior_turns = messages[-(MAX_REMEMBERED_QUESTIONS * 2) :]
    contents = []
    for msg in prior_turns:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=msg["content"])],
            )
        )
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=new_prompt)],
        )
    )
    return contents


def ask_paul_von_hindenburg(client: genai.Client, history_contents: list[types.Content]) -> str:
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        thinking_config=types.ThinkingConfig(thinking_level="minimal"),
    )
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=history_contents,
        config=config,
    )
    raw_text = (response.text or "").strip()
    return enforce_word_limits(raw_text)


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Ask Paul von Hindenburg (1847–1934)",
    page_icon="🎖️",
    layout="centered",
)

st.title("🎖️ Speak with Paul von Hindenburg")
st.caption(
    "An interactive primary-source history chatbot for U.S. 10th-grade students "
    f"• Powered by `{MODEL_ID}`"
)

# Load hidden API key from Streamlit Secrets
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error(
        "Missing `GEMINI_API_KEY`. Add it to `.streamlit/secrets.toml` locally "
        "or in Streamlit Community Cloud → App Settings → Secrets."
    )
    st.stop()

client = get_genai_client(api_key)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar controls & archive reference
with st.sidebar:
    st.header("Historical Guidelines")
    remembered_q_count = min(
        MAX_REMEMBERED_QUESTIONS,
        sum(1 for m in st.session_state.messages if m["role"] == "user"),
    )
    st.metric("Active Memory (Max 6 Questions)", f"{remembered_q_count} / {MAX_REMEMBERED_QUESTIONS}")
    st.write("**Reading Level:** U.S. 10th Grade")
    st.write("**Response Length:** 15–120 words")
    st.write(f"**Temperature:** {TEMPERATURE} | **Max Tokens:** {MAX_OUTPUT_TOKENS}")

    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("Try Asking:")
    sample_questions = [
        "What did you testify before the Reichstag committee in November 1919?",
        "How did Germany's court rule in Preußen contra Reich (1932) on your Article 48 decree?",
        "What did your April 4, 1933 letter to Hitler say about Jewish war veterans?",
        "Why did you sign the Reichstag Fire Decree and Enabling Act in 1933?",
        "What did your correspondence with Chancellor Brüning show about ruling by emergency decree?",
    ]
    for q in sample_questions:
        if st.button(q, use_container_width=True):
            st.session_state.pending_prompt = q

    st.divider()
    st.subheader("Referenced Digital Archives")
    st.markdown(
        "- Library of Congress Digital Collections\n"
        "- HathiTrust Digital Library\n"
        "- German Federal Archives (Bundesarchiv - Invenio & Digitales Bildarchiv)\n"
        "- Akten der Reichskanzlei (Weimar Republic Edition)\n"
        "- Bavarian State Library (Bayerische Staatsbibliothek - Reichstagsprotokolle)\n"
        "- 100(0) Schlüsseldokumente zur deutschen Geschichte\n"
        "- 1914-1918-online (International Encyclopedia of the First World War)\n"
        "- Europeana 1914-1918\n"
        "- Filmportal.de\n"
        "- German Historical Institute Washington (GHDI)\n"
        "- Germany - The EHRI Portal"
    )

# Render chat history
if not st.session_state.messages:
    with st.chat_message("assistant", avatar="🎖️"):
        st.write(
            "Good day, young student. I am Paul von Hindenburg, former Field Marshal and President of the "
            "Weimar Republic from 1925 to 1934. Ask Me about the 1918 Armistice, Article 48 decrees, My "
            "letters to Chancellors Brüning and Hitler, or the laws of 1933—I shall answer from the archives. "
            "Source: German Historical Institute Washington (GHDI)."
        )

for msg in st.session_state.messages:
    avatar = "🎖️" if msg["role"] == "assistant" else "🎓"
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["content"])

# Accept chat input or sidebar button prompt
prompt = st.chat_input("Ask Paul von Hindenburg a historical question...")
if "pending_prompt" in st.session_state:
    prompt = st.session_state.pop("pending_prompt")

if prompt:
    with st.chat_message("user", avatar="🎓"):
        st.write(prompt)

    # Build history with ONLY the previous 6 questions + current prompt
    history_contents = build_limited_history(st.session_state.messages, prompt)

    with st.chat_message("assistant", avatar="🎖️"):
        with st.spinner("Reviewing the Reich Chancellery & Federal Archives..."):
            try:
                reply = ask_paul_von_hindenburg(client, history_contents)
            except Exception as exc:
                st.error(f"Could not reach Gemini API: {exc}")
                st.stop()
        st.write(reply)

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": reply})

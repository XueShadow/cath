import os
from pathlib import Path

from dotenv import load_dotenv
import openai
import streamlit as st


st.set_page_config(page_title="Recipe Chat", page_icon="✦", layout="wide")

project_root = Path(__file__).resolve().parents[2]
load_dotenv(project_root / ".env", override=True)
hf_token = os.getenv("HF_TOKEN")


@st.cache_resource
def get_client(token):
    return openai.OpenAI(
        api_key=token,
        base_url="https://router.huggingface.co/v1",
        timeout=30.0,
        max_retries=0,
    )


client = get_client(hf_token) if hf_token and hf_token != "test" else None


def stream_response(messages):
    if client is None:
        yield "Demo mode: add a valid HF_TOKEN to your .env file."
        return

    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a friendly food recommendation and recipe assistant. "
                        "Recommend dishes based on ingredients, dietary needs, cuisine, "
                        "budget, and cooking time. Provide practical recipes with ingredients, "
                        "clear numbered steps, cooking time, servings, substitutions, and allergens."
                    ),
                },
                *messages,
            ],
            max_tokens=160,
            stream=True,
        )
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                yield content
    except openai.RateLimitError:
        yield "The Hugging Face request was rate-limited. Please try again shortly."
    except openai.AuthenticationError:
        yield "Hugging Face authentication failed. Check HF_TOKEN in your .env file."
    except openai.APIStatusError as exc:
        if exc.status_code == 402:
            yield "Hugging Face inference credits are exhausted. Add credits or upgrade the account."
        else:
            yield f"The model returned HTTP {exc.status_code}."
    except Exception as exc:
        yield f"The model could not respond: {exc}"


st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] { background: #f7f7f8; color: #202123; }
    [data-testid="stMain"] { padding-top: 1.5rem; }
    [data-testid="stMainBlockContainer"] { max-width: 980px; padding-bottom: 8rem; }
    [data-testid="stAppViewContainer"] .stMarkdown,
    [data-testid="stAppViewContainer"] label,
    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] h1,
    [data-testid="stAppViewContainer"] h2,
    [data-testid="stAppViewContainer"] h3 { color: #202123; }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { background: #202123; }
    [data-testid="stSidebar"] * { color: #ececf1; }
    [data-testid="stSidebar"] button { border-color: #565869; background: #343541; }
    [data-testid="stSidebar"] button:hover { border-color: #ffffff; background: #40414f; }
    .brand { font-size: 1.15rem; font-weight: 750; letter-spacing: -.02em; }
    .brand-mark { color: #f59e0b; }
    .subtitle { color: #6b6b73; margin: .15rem 0 1.75rem; font-size: .98rem; }
    .topline { display: flex; align-items: center; justify-content: space-between; margin-bottom: .25rem; }
    .status { color: #166534; background: #dcfce7; border: 1px solid #bbf7d0; border-radius: 999px; padding: .3rem .65rem; font-size: .78rem; font-weight: 650; }
    .welcome { max-width: 760px; margin: 15vh auto 0; text-align: center; }
    .welcome-icon { display: inline-flex; align-items: center; justify-content: center; width: 58px; height: 58px; border-radius: 18px; background: #fff7ed; color: #ea580c; font-size: 1.8rem; margin-bottom: .9rem; }
    .welcome h1 { font-size: 2.35rem; letter-spacing: -.04em; color: #202123; margin: 0 0 .55rem; }
    .welcome p { color: #6b6b73; font-size: 1.03rem; margin-bottom: 1.35rem; }
    .prompt-card { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 14px; padding: .9rem 1rem; text-align: left; color: #374151; font-size: .9rem; }
    [data-testid="stChatMessage"] { color: #202123; padding: 1.1rem 1rem; border-bottom: 1px solid #ececef; }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] { max-width: 760px; line-height: 1.65; }
    [data-testid="stChatMessage"] [data-testid="stChatMessageAvatarUser"] { background: #374151; }
    [data-testid="stChatMessage"] [data-testid="stChatMessageAvatarAssistant"] { background: #ea580c; }
    [data-testid="stChatInput"] { border-top: 0; }
    [data-testid="stChatInput"] textarea { color: #202123; background: #ffffff; border: 1px solid #d1d5db; border-radius: 14px; }
    [data-testid="stChatInput"] textarea::placeholder { color: #6b6b73; opacity: 1; }
    [data-testid="stChatInput"] textarea:focus { border-color: #ea580c; box-shadow: 0 0 0 2px #fed7aa; }
    div[data-testid="stChatInput"] { max-width: 780px; margin: 0 auto; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown('<div class="brand"><span class="brand-mark">✦</span> Recipe Chat</div>', unsafe_allow_html=True)
    st.caption("Your food recommendation assistant")
    if st.button("＋ New chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption("MODEL")
    st.markdown("**Llama 3.1 8B Instruct**")
    st.caption("Responses are generated through Hugging Face.")

st.markdown('<div class="topline"><h2>Recipe Chat</h2><span class="status">● Ready to cook</span></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Get dish ideas and easy recipes from the ingredients you have.</div>',
    unsafe_allow_html=True,
)

if not st.session_state.messages:
    st.markdown(
        '<div class="welcome"><div class="welcome-icon">✦</div><h1>What are we cooking?</h1><p>Turn whatever is in your kitchen into something delicious.</p></div>',
        unsafe_allow_html=True,
    )
    starter_columns = st.columns(3)
    starter_prompts = [
        "Quick dinner with chicken and rice",
        "Vegetarian meal under 30 minutes",
        "Dessert using bananas",
    ]
    for column, prompt in zip(starter_columns, starter_prompts):
        with column:
            st.markdown(f'<div class="prompt-card">{prompt}</div>', unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_prompt = st.chat_input("What would you like to cook?")
if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)
    with st.chat_message("assistant"):
        response_text = st.write_stream(stream_response(st.session_state.messages))
    st.session_state.messages.append({"role": "assistant", "content": response_text})

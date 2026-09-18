# import packages
import streamlit as st
import pandas as pd
import os
import plotly.express as px
import openai
from dotenv import load_dotenv


load_dotenv()

st.set_page_config(
    page_title="Avalanche Sentiment Analysis",
    page_icon="🏔️",
    layout="wide",
)


def get_dataset_path():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    return os.path.join(project_root, "data", "customer_reviews.csv")


@st.cache_data
def load_reviews():
    return pd.read_csv(get_dataset_path())


@st.cache_data
def get_response(review, temperature):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    client = openai.OpenAI(api_key=api_key)
    prompt = f"""
Analyze this Avalanche customer review.

Review: {review}

Return exactly this format:
Sentiment: Positive, Negative, or Neutral
Explanation: one brief explanation
"""
    response = client.responses.create(
        model="gpt-4o",
        input=prompt,
        temperature=temperature,
        max_output_tokens=200,
    )
    return response.output_text.strip()


def score_label(score):
    if score > 0.1:
        return "Positive"
    if score < -0.1:
        return "Negative"
    return "Neutral"


st.title("🏔️ Avalanche Sentiment Analysis Dashboard")
st.write("Explore customer feedback and use GenAI to explain the sentiment of a review.")

try:
    reviews = load_reviews()
except FileNotFoundError:
    st.error(f"Dataset not found at {get_dataset_path()}")
    st.stop()

reviews["Sentiment"] = reviews["SENTIMENT_SCORE"].apply(score_label)

product_options = ["All Products"] + sorted(reviews["PRODUCT"].dropna().unique().tolist())
product = st.sidebar.selectbox("Filter by product", product_options)
filtered_reviews = reviews if product == "All Products" else reviews[reviews["PRODUCT"] == product]

metric_columns = st.columns(3)
metric_columns[0].metric("Reviews", len(filtered_reviews))
metric_columns[1].metric("Average score", f"{filtered_reviews['SENTIMENT_SCORE'].mean():.2f}")
metric_columns[2].metric("Most common", filtered_reviews["Sentiment"].mode().iat[0])

st.subheader(f"Customer Reviews: {product}")
st.dataframe(
    filtered_reviews[["PRODUCT", "DATE", "SUMMARY", "SENTIMENT_SCORE", "Sentiment", "Order ID"]],
    use_container_width=True,
    hide_index=True,
)

sentiment_counts = filtered_reviews["Sentiment"].value_counts().reindex(
    ["Negative", "Neutral", "Positive"], fill_value=0
).rename_axis("Sentiment").reset_index(name="Count")
fig = px.bar(
    sentiment_counts,
    x="Sentiment",
    y="Count",
    color="Sentiment",
    color_discrete_map={"Negative": "#c0392b", "Neutral": "#7f8c8d", "Positive": "#218c74"},
    title="Sentiment Breakdown",
)
fig.update_layout(showlegend=False)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Analyze a Review with GenAI")
review_index = st.selectbox(
    "Choose a review",
    filtered_reviews.index,
    format_func=lambda index: f"{filtered_reviews.loc[index, 'PRODUCT']} - {filtered_reviews.loc[index, 'SUMMARY'][:90]}...",
)
selected_review = filtered_reviews.loc[review_index, "SUMMARY"]
st.info(selected_review)
temperature = st.slider("Model temperature", 0.0, 1.0, 0.2, 0.05)

if st.button("Analyze Sentiment", type="primary"):
    with st.spinner("Analyzing review..."):
        try:
            result = get_response(selected_review, temperature)
        except Exception as error:
            st.error(f"OpenAI request failed: {error}")
            result = None

    if result:
        st.subheader("🤖 AI Sentiment Analysis")
        st.write(result)
    else:
        st.warning("Set OPENAI_API_KEY in your .env file to enable GenAI analysis.")
        st.write(f"Dataset sentiment: **{filtered_reviews.loc[review_index, 'Sentiment']}**")
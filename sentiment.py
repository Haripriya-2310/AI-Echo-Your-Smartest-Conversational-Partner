import streamlit as st
import pandas as pd
import numpy as np
import spacy
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from wordcloud import WordCloud

# Load NLP model
 
nlp = spacy.load("en_core_web_sm")


st.set_page_config(page_title="AI Echo – Sentiment Analysis",layout="wide")

st.title("💬 AI Echo: Your Smartest Conversational Partner")

st.subheader("Sentiment Analysis & Insight-Driven Review Dashboard")

# Load Model & Vectorizer 

@st.cache_resource
def load_model():
    with open('tfidf_senti.pkl', 'rb') as f:
        tfidf = pickle.load(f)

    with open('log_reg_senti.pkl', 'rb') as f:
        model = pickle.load(f)

    return tfidf, model

tfidf, model = load_model()

# Load Dataset
 
df = pd.read_csv("cleaned_senti.csv")
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')


# Strip column names in case of extra spaces
df.columns = df.columns.str.strip()

 
# Preprocessing function
 
def spacy_preprocess(text):
    if not isinstance(text, str):
        return ""
    doc = nlp(text)
    tokens = [
        token.lemma_.lower() 
        for token in doc 
        if not token.is_stop and not token.is_punct and token.lemma_ != "-PRON-"
    ]
    return " ".join(tokens)


# Apply preprocessing to 'review' column

if 'review' in df.columns:
    df['review'] = df['review'].apply(spacy_preprocess)
else:
    st.error("The CSV file does not contain a 'review' column.")

# 🔮 Live Sentiment Prediction 

menu = st.sidebar.selectbox(
    "Explore 🔍",
    ["🔮 Enter Review", "📊 Sentiment Analysis Insights"])

if menu == "🔮 Enter Review":
    st.header("🔮 Predict Sentiment for New Review")

    user_review = st.text_area("Enter a review:")

    if st.button("Analyze Sentiment"):
        if user_review.strip():
            processed_review = spacy_preprocess(user_review)
            vec = tfidf.transform([processed_review])
            pred = model.predict(vec)[0]
            proba = model.predict_proba(vec)[0]

            label_map = {0: "Negative 😡", 1: "Neutral 😐", 2: "Positive 😊"}

            st.success(f"Predicted Sentiment: **{label_map[pred]}**")
            st.write("Prediction Confidence:")
            st.bar_chart(pd.Series(proba, index=label_map.values()))



# Add Sentiment Predictions to Dataset
@st.cache_data
def add_predictions(df):
    vectors = tfidf.transform(df['review'].astype(str))
    preds = model.predict(vectors)
    df['predicted_sentiment'] = preds
    df['predicted_label'] = df['predicted_sentiment'].map({
        0: "Negative", 1: "Neutral", 2: "Positive"
    })
    return df

df = add_predictions(df)


# 📊 DASHBOARD QUESTIONS

if menu == "📊 Sentiment Analysis Insights":

    st.title("Key Questions for Sentiment Analysis")

    # ---------- Q1:Overall Sentiment Distribution ----------
    st.header("1️⃣ What is the overall sentiment of user reviews?")
    st.write(
        "This chart shows the percentage distribution of **Positive, Neutral, and Negative** reviews.")

    sentiment_dist = df['predicted_label'].value_counts(normalize=True) * 100
    st.bar_chart(sentiment_dist)

    st.divider()

    # ---------- Q2:Sentiment vs Rating ----------
    st.header("2️⃣ How does sentiment vary by rating?")
    st.write(
        "This analysis compares star ratings with predicted sentiment to identify mismatches "
        "(e.g., negative sentiment in high ratings)."
    )

    fig, ax = plt.subplots()
    sns.countplot(data=df, x='rating', hue='predicted_label', ax=ax)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Number of Reviews")
    st.pyplot(fig)

    st.divider()

    # ---------- Q3:Keywords per Sentiment ----------
    st.header("3️⃣ What keywords are commonly used in each sentiment?")
    st.write(
        "A word cloud showing the most frequent words used in reviews for the selected sentiment.")

    choice = st.selectbox("Choose Sentiment", df['predicted_label'].unique())
    text = " ".join(df[df['predicted_label'] == choice]['review'])

    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wc)
    ax.axis('off')
    st.pyplot(fig)

    st.divider()

    # ---------- Q4: Sentiment Trend Over Time ----------
    st.header("4️⃣ How does sentiment change over time?")
    st.write(
        "This trend analysis shows how user sentiment evolves month by month.")

    df['month'] = df['date'].dt.to_period('M').astype(str)
    trend = df.groupby(['month', 'predicted_label']).size().unstack().fillna(0)

    st.line_chart(trend)

    st.divider()

    # ---------- Q5: Verified vs Non-Verified Users ----------
    st.header("5️⃣ Is sentiment different for verified vs non-verified users?")
    st.write(
        "This comparison highlights sentiment differences between verified purchasers and non-verified users.")

    fig, ax = plt.subplots()
    sns.countplot(data=df, x='verified_purchase', hue='predicted_label', ax=ax)
    ax.set_xlabel("Verified Purchase")
    ax.set_ylabel("Number of Reviews")
    st.pyplot(fig)

    st.divider()

    # ---------- Q6: Review Length vs Sentiment ----------
    st.header("6️⃣ Does review length vary by sentiment?")
    st.write(
        "This box plot compares the length of reviews across different sentiment categories.")

    df['review_length'] = df['review'].apply(lambda x: len(str(x).split()))

    fig, ax = plt.subplots()
    sns.boxplot(data=df, x='predicted_label', y='review_length', ax=ax)

    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Review Length (Number of Words)")
    ax.set_title("Review Length Distribution by Sentiment")

    st.pyplot(fig)


    # ---------- Q7:Sentiment by Location ----------
    st.header("7️⃣ Which locations show the most positive or negative sentiment?")
    st.write(
        "This analysis highlights geographic regions where users express the strongest "
        "positive or negative sentiment, helping identify location-based experience issues."
    )

    def plot_categorical(col, title):
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(data=df, x=col, hue='predicted_label', ax=ax)
        ax.set_title(title)
        ax.set_xlabel(col.capitalize())
        ax.set_ylabel("Number of Reviews")
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)

    plot_categorical("location", "Location-wise Sentiment Distribution")

    st.divider()

    # ---------- Q8:Platform-wise Sentiment ----------
    st.header("8️⃣ Is there a difference in sentiment across platforms (Web vs Mobile)?")
    st.write(
        "This comparison helps identify whether users on different platforms "
        "(such as Web or Mobile) report different sentiment patterns."
    )

    plot_categorical("platform", "Platform-wise Sentiment Comparison")

    st.divider()

    # ---------- Q9: Version-wise Sentiment ----------
    st.header("9️⃣ Which ChatGPT versions are associated with higher or lower sentiment?")
    st.write(
        "This visualization shows sentiment distribution across different ChatGPT versions "
        "to assess whether version updates impacted user satisfaction."
    )

    plot_categorical("version", "Version-wise Sentiment Analysis")

    st.divider()

    # ---------- Q10:Common Negative Feedback Themes ----------
    st.header("🔟 What are the most common negative feedback themes?")
    st.write(
        "Frequent keywords from negative reviews are extracted to identify recurring "
        "pain points and user complaints."
    )

    neg_text = " ".join(df[df['predicted_label'] == "Negative"]['review'])

    from collections import Counter
    import re

    words = re.findall(r'\b\w+\b', neg_text.lower())
    common = Counter(words).most_common(20)

    st.dataframe(
        pd.DataFrame(common, columns=["Keyword", "Frequency"])
    )

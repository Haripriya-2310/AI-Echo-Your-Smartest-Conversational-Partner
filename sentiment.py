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


st.set_page_config(page_title="AI Echo – Sentiment Analysis", layout="wide")
st.title("💬 AI Echo – Sentiment Analysis Dashboard")


# Load Model & Vectorizer (IMPORTANT)

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
    ["🔮 Enter Review", "📊 Dashboard / Insights"]
)
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


# 📊 DASHBOARD QUESTIONS (USING MODEL OUTPUT)

if menu == "📊 Dashboard / Insights":
    #---------- Q1: Overall Sentiment Distribution ----------
    st.header("1️⃣ Overall Sentiment Distribution")

    sentiment_dist = df['predicted_label'].value_counts(normalize=True) * 100
    st.bar_chart(sentiment_dist)

    #---------- Q2: Sentiment vs Rating ----------
    st.header("2️⃣ Sentiment vs Rating")

    fig, ax = plt.subplots()
    sns.countplot(data=df, x='rating', hue='predicted_label', ax=ax)
    st.pyplot(fig)

    #---------- Q3: Keywords per Sentiment ----------
    st.header("3️⃣ Keywords by Sentiment")

    choice = st.selectbox("Choose Sentiment", df['predicted_label'].unique())
    text = " ".join(df[df['predicted_label'] == choice]['review'])

    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wc)
    ax.axis('off')
    st.pyplot(fig)


    #---------- Q4: Sentiment Trend Over Time ----------
    st.header("4️⃣ Sentiment Over Time")

    df['month'] = df['date'].dt.to_period('M').astype(str)
    trend = df.groupby(['month', 'predicted_label']).size().unstack().fillna(0)

    st.line_chart(trend)

    #---------- Q5: Verified vs Non-Verified Users ----------
    st.header("5️⃣ Verified vs Non-Verified Users")

    fig, ax = plt.subplots()
    sns.countplot(data=df, x='verified_purchase', hue='predicted_label', ax=ax)
    st.pyplot(fig)

    #---------- Q6: Review Length vs Sentiment ----------
    st.header("6️⃣ Review Length vs Sentiment")

    df['review_length'] = df['review'].apply(lambda x: len(str(x).split()))

    fig, ax = plt.subplots()
    sns.boxplot(data=df, x='predicted_label', y='review_length', ax=ax)
    st.pyplot(fig)


    #---------- Q7–Q9 (Location, Platform, Version) ----------
    def plot_categorical(col, title):
        fig, ax = plt.subplots(figsize=(10,5))
        sns.countplot(data=df, x=col, hue='predicted_label', ax=ax)
        ax.set_title(title)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)

    st.header("7️⃣ Sentiment by Location")
    plot_categorical("location", "Location-wise Sentiment")

    st.header("8️⃣ Platform-wise Sentiment")
    plot_categorical("platform", "Platform-wise Sentiment")

    st.header("9️⃣ Version-wise Sentiment")
    plot_categorical("version", "Version-wise Sentiment")

    #---------- Q10: Common Negative Feedback Themes ----------
    st.header("🔟 Common Negative Feedback Themes")

    neg_text = " ".join(df[df['predicted_label'] == "Negative"]['review'])

    from collections import Counter
    import re

    words = re.findall(r'\b\w+\b', neg_text.lower())
    common = Counter(words).most_common(20)

    st.dataframe(pd.DataFrame(common, columns=["Word", "Frequency"]))
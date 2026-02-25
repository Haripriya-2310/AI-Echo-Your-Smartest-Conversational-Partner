import streamlit as st
import pandas as pd
import numpy as np
import base64 
import spacy
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from wordcloud import WordCloud

# Load NLP model
 
nlp = spacy.load("en_core_web_sm")


st.set_page_config(page_title="AI Echo – Sentiment Analysis",layout="wide")

# Setting background image

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://www.aimtechnologies.co/wp-content/uploads/2024/01/Sentiment-Analysis-Techniques.jpeg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """, unsafe_allow_html=True
)

#Title
st.title("💬 AI Echo: Your Smartest Conversational Partner")

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

    user_input = st.text_area("Enter your review here:")

    if st.button("Predict Sentiment"):
        if user_input.strip() == "":
            st.warning("Please enter some text for prediction.")
        else:
            # Preprocess user input
            input_processed = spacy_preprocess(user_input)

            # Vectorize
            input_tfidf = tfidf.transform([input_processed])

            # Predict
            proba = model.predict_proba(input_tfidf)[0]
            positive_keywords = [
                "awesome", "excellent", "great", "amazing",
                "love", "perfect", "satisfied", "happy"
            ]

            negative_keywords = [
                "bad", "worst", "terrible", "awful",
                "poor", "hate", "disappointed"
            ]

            neutral_keywords = [
                "okay", "average", "fine", "decent",
                "nothing special", "works", "acceptable"
            ]

            text = user_input.lower()

            if any(w in text for w in positive_keywords):
                prediction = 2  # Positive
            elif any(w in text for w in negative_keywords):
                prediction = 0  # Negative
            elif any(w in text for w in neutral_keywords):
                prediction = 1  # Neutral
            elif proba[1] >= 0.35:
                prediction = 1  # Neutral (probability-based)
            else:
                prediction = np.argmax(proba)

            sentiment_map = {0: "Negative 😡", 1: "Neutral 😐", 2: "Positive 😊"}
            pred_sentiment = sentiment_map[prediction]
                    

            # Display text
            st.write(f"### 🔮 Predicted Sentiment: **{pred_sentiment}**")

# 📊 DASHBOARD QUESTIONS

if menu == "📊 Sentiment Analysis Insights":

    st.header("SENTIMENT ANALYSIS INSIGHTS")

    #  ---------- Q1:Overall Sentiment Distribution ----------
    with st.expander(" 1. What is the overall sentiment of user reviews?"):
        sentiment_counts = df['sentiment'].value_counts(normalize=True).round(2) * 100
        st.bar_chart(sentiment_counts)

    # ---------- Q2:Sentiment vs Rating ----------
    with st.expander("2. How does sentiment vary by rating?"):
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='rating', hue='sentiment', ax=ax)
        ax.set_xlabel("Rating")
        ax.set_ylabel("Number of Reviews")
        st.pyplot(fig)

       
    # ---------- Q3:Keywords per Sentiment ----------
    with st.expander("3. Which keywords are most associated with each sentiment class?"):
        choice = st.selectbox("Choose Sentiment", df['sentiment'].unique())
        text = ' '.join(df[df['sentiment'] == choice]['review'].astype(str))
        wc = WordCloud(width=800, height=400, background_color='white').generate(text)
        fig, ax = plt.subplots()
        ax.imshow(wc)
        ax.axis('off')
        st.pyplot(fig)


    # ---------- Q4: Sentiment Trend Over Time ----------
    with st.expander("4. How has sentiment changed over time?"):
        df['month'] = df['date'].dt.to_period('M').astype(str)
        trend = df.groupby(['month', 'sentiment']).size().unstack().fillna(0)
        st.line_chart(trend)


    # ---------- Q5: Verified vs Non-Verified Users ----------
    with st.expander("5. Do verified users tend to leave more positive or negative reviews?"):
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='verified_purchase', hue='sentiment', ax=ax)
        ax.set_xlabel("Verified Purchase")
        ax.set_ylabel("Number of Reviews")
        st.pyplot(fig)

    # ---------- Q6: Review Length vs Sentiment ----------
    with st.expander("6. Are longer reviews more likely to be negative or positive?"):
        
        df['review_length'] = df['review'].apply(lambda x: len(str(x).split()))

        fig, ax = plt.subplots()
        sns.boxplot(data=df, x='sentiment', y='review_length', ax=ax)

        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Review Length (Number of Words)")
        ax.set_title("Review Length Distribution by Sentiment")

        st.pyplot(fig)

    #  ---------- Q7:Sentiment by Location ----------
    with st.expander("7. Which locations show the most positive or negative sentiment?"):
        if 'location' in df.columns:
            st.bar_chart(df.groupby('location')['sentiment'].value_counts(normalize=True).unstack().fillna(0))
        else:
            st.info("Location column not found.")

    # ---------- Q8:Platform-wise Sentiment ----------
    with st.expander("8. Is there a difference in sentiment across platforms?"):
        if {'platform', 'sentiment'}.issubset(df.columns):

            plt.figure(figsize=(8, 5))
            sns.countplot(data=df, x='platform',hue='sentiment')
            plt.xlabel("Platform")
            plt.ylabel("Number of Reviews")
            plt.title("Sentiment Distribution Across Platforms")
            plt.legend(title="Sentiment")
            st.pyplot(plt)
        else:
            st.info("Platform or sentiment column not found.")

    # ---------- Q9: Version-wise Sentiment ----------
    with st.expander("9. Which ChatGPT versions are associated with higher/lower sentiment?"):
        if 'version' in df.columns:
            st.bar_chart(df.groupby('version')['sentiment'].value_counts(normalize=True).unstack().fillna(0))
        else:
            st.info("ChatGPT version column not found.")

    # ---------- Q10:Common Negative Feedback Themes ----------
    from collections import Counter
    import re
    with st.expander("10. What are the most common negative feedback themes?"):
        neg_text = " ".join(df[df['sentiment'] == "Negative"]['review'])

        words = re.findall(r'\b\w+\b', neg_text.lower())
        common = Counter(words).most_common(20)

        st.dataframe(
            pd.DataFrame(common, columns=["Keyword", "Frequency"])
        )

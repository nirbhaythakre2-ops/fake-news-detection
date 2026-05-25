import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

@st.cache_data
def load_data():
    fake = pd.read_csv("Fake.csv")
    true = pd.read_csv("True.csv")

    fake["label"] = 0
    true["label"] = 1

    data = pd.concat([fake, true])
    data = data[["text", "label"]]

    return data

@st.cache_resource
def train_model():
    data = load_data()

    X = data["text"]
    y = data["label"]

    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, vectorizer, accuracy

model, vectorizer, accuracy = train_model()

st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #141e30, #243b55);
}
.big-title {
    text-align: center;
    font-size: 55px;
    font-weight: bold;
    color: white;
}
.sub-title {
    text-align: center;
    font-size: 20px;
    color: #d1d5db;
}
div.stButton > button {
    width: 100%;
    background-color: #2563eb;
    color: white;
    font-size: 20px;
    border-radius: 12px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">📰 Fake News Detector AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">AI Powered News Verification System</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,2,1])

with col2:
    st.metric("Model Accuracy", f"{accuracy*100:.2f}%")
    news = st.text_area("📝 Enter News Text", height=200)

    if st.button("🚀 Analyze News"):
        if news.strip() == "":
            st.warning("Enter news text first")
        else:
            news_vector = vectorizer.transform([news])
            prediction = model.predict(news_vector)

            if prediction[0] == 1:
                st.success("✅ This is REAL News")
            else:
                st.error("❌ This is FAKE News")
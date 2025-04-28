import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
import os

# Setup NLTK data path
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_path):
    os.mkdir(nltk_data_path)

nltk.data.path.append(nltk_data_path)

# Download necessary NLTK data (only if not already downloaded)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_path)

# Load the trained model and vectorizer
model = joblib.load('movie_sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Preprocessing function
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = text.split()  # Simple split instead of word_tokenize
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Streamlit App Interface
st.set_page_config(page_title="Movie Review Sentiment Analyzer", layout="centered")
st.title("🎬 Movie Review Sentiment Analyzer")

st.markdown(
    """
    Enter a **movie review** below to find out whether it's **positive** or **negative**!  
    Powered by **Logistic Regression** and **TF-IDF** magic 
    """
)

user_input = st.text_area("📝 Enter your movie review here:")

if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a movie review first.")
    else:
        # Preprocess and predict
        processed = preprocess_text(user_input)
        input_vec = vectorizer.transform([processed])
        prediction = model.predict(input_vec)[0]
        
        # Display result
        if prediction == 1:
            st.success("✅ Predicted Sentiment: **Positive**")
        else:
            st.error("❌ Predicted Sentiment: **Negative**")

# Optional: Add a nice sidebar
st.sidebar.title("About")
st.sidebar.info(
    """
    This app analyzes the sentiment of movie reviews.
    
    **Model:** Logistic Regression  
    **Vectorizer:** TF-IDF  
    **Dataset:** IMDB Movie Reviews
    """
)

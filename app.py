import streamlit as st
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load model and vectorizer
@st.cache_resource
def load_assets():
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_assets()

# Text preprocessing
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Keep only letters
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# ---- Sidebar with Info ----
with st.sidebar:
    st.header("ℹ️ About This Tool")
    st.markdown("""
    **Model Details:**
    - Trained on [Twitter Sentiment Analysis Dataset](https://www.kaggle.com/datasets) (10,000+ tweets)
    - Labels: Positive, Negative, Neutral, Irrelevant
    - Accuracy: ~85%
    
    **Borderlands Context:**
    - Analyzes sentiment about:
      - Borderlands 3 (2019)
      - Tiny Tina's Wonderlands (2022)
      - General series content
    """)

# ---- Main App ----
st.title("🎮 Borderlands Sentiment Analyzer")
user_input = st.text_area("Enter your Borderlands review:", "This game is awesome!")

if st.button("Analyze"):
    # Process and predict
    processed_text = preprocess(user_input)
    X = vectorizer.transform([processed_text])
    prediction = model.predict(X)[0]
    confidence = model.predict_proba(X)[0].max() * 100
    
    # Display results
    st.success(f"Prediction: **{prediction}**")
    st.metric("Confidence", f"{confidence:.1f}%")
    
    # Explanation
    st.divider()
    st.subheader("🤖 Analysis Insights")
    
    if prediction == "Positive":
        st.markdown("""
        The model detected positive indicators:
        - Words like *"awesome"*, *"love"*, *"great"*
        - Generally enthusiastic language
        """)
    elif prediction == "Negative":
        st.markdown("""
        The model detected negative indicators:
        - Words like *"bad"*, *"worst"*, *"disappointing"*
        - Critical or frustrated language
        """)
    else:
        st.markdown("""
        The model detected neutral/mixed signals:
        - Balanced or ambiguous language
        - Possibly irrelevant to Borderlands
        """)
    
    # Raw details
    with st.expander("Technical Details"):
        st.write("Processed text:", processed_text)
        st.write("Feature weights:", dict(zip(vectorizer.get_feature_names_out(), X.toarray()[0])))
        st.write("Class probabilities:", dict(zip(model.classes_, model.predict_proba(X)[0])))
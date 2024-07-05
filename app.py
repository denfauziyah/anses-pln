import nltk
from vadersentiment import SentimentIntensityAnalyzer

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

import streamlit as st

st.title('Sentiment Analysis')

text = st.text_area('Enter text')

if text:
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    Sentiment = scores['compound']
    if Sentiment > 0:
        st.write('Positive')
    elif Sentiment < 0:
        st.write('Negative')
    else:
        st.write('Neutral')


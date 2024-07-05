!pip install -q streamlit

!pip install nltk textblob

import nltk
from textblob import TextBlob

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

import streamlit as st

st.title('Sentiment Analysis')

text = st.text_area('Enter text')

if text:
    blob = TextBlob(text)
    Sentiment = blob.sentiment.polarity
    if Sentiment > 0:
        st.write('Positive')
    elif Sentiment < 0:
        st.write('Negative')
    else:
        st.write('Neutral')


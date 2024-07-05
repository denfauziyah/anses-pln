import streamlit as st
import nltk
nltk.download('punkt')
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.tokenize import word_tokenize

# Buat objek stemmer dan stopword remover
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stopword_remover = StopWordRemoverFactory().create_stop_word_remover()

# Fungsi untuk pra-pemrosesan teks
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(token) for token in tokens]
    filtered_tokens = []
    for token in tokens:
        if not stopword_remover.remove(token):  # Gunakan remove() untuk memeriksa apakah token adalah stop word
            filtered_tokens.append(token)
    tokens = filtered_tokens
    return " ".join(tokens)

# Fungsi untuk analisis sentimen (Anda perlu mengimplementasikan logika ini)
def analyze_sentiment(text):
    # Implementasikan logika analisis sentimen di sini, misalnya:
    # - Menggunakan lexicon sentimen
    # - Menggunakan model machine learning yang dilatih pada dataset sentimen bahasa Indonesia
    # - Menggunakan layanan API analisis sentimen
    # ...
    # Kembalikan skor sentimen (misalnya, -1 untuk negatif, 0 untuk netral, 1 untuk positif)
    return 0  # Ganti dengan hasil analisis sentimen Anda

# Tampilan aplikasi Streamlit
st.title('Aplikasi Analisis Sentimen')

text = st.text_area('Masukkan teks di sini:')

if text:
    processed_text = preprocess_text(text)
    sentiment = analyze_sentiment(processed_text)

    if sentiment > 0:
        st.write('Sentimen: Positif')
    elif sentiment < 0:
        st.write('Sentimen: Negatif')
    else:
        st.write('Sentimen: Netral')

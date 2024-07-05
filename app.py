import streamlit as st
import nltk
nltk.download('punkt')
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
# Muat dataset (ganti dengan path ke dataset Anda)
with open('komenpln.csv', 'r') as f:
    data = f.readlines()
# Pisahkan teks dan label
texts = [line.strip().split(',')[0] for line in data]
labels = [line.strip().split(',')[1] for line in data]

# Pra-proses teks
processed_texts = [preprocess_text(text) for text in texts]

# Vektorisasi teks menggunakan TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_texts)
# Bagi dataset menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Latih model Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluasi model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Akurasi model:', accuracy)

# Fungsi untuk analisis sentimen
def analyze_sentiment(text):
    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_text)[0]
    print("Prediksi sentimen:", prediction)
    return prediction
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

import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import base64

nltk.download('punkt_tab')
nltk.download('stopwords')

ps = PorterStemmer()

#Background Function
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def transform_text(text):
  text = text.lower()
  text = nltk.word_tokenize(text)
  y=[]
  for i in text:
    if i.isalnum():
      y.append(i)

  text = y[:]
  y.clear()
  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)

  text = y[:]
  y.clear()
  ps = PorterStemmer()
  for i in text:
    y.append(ps.stem(i))
  return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))
import pickle

model = pickle.load(open('model.pkl', 'rb'))

bg_img = get_base64("background_sms.png")

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bg_img}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("""
<style>
textarea {
    background-color: rgba(20,20,30,0.85) !important;
}

[data-testid="stTextArea"] {
    background-color: rgba(20,20,30,0.85);
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<h1 style='text-align: center;'>📩 SMS Spam Detection System</h1>",
    unsafe_allow_html=True
)

input_sms = st.text_area(
    "Enter the message:",
    placeholder="Paste an SMS message here..."
)

if st.button("🔍 Analyze Message"):

    # 1. Preprocess
    transformed_sms = transform_text(input_sms)

    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # 3. Predict
    result = model.predict(vector_input)[0]

    # 4. Confidence Scores
    probability = model.predict_proba(vector_input)

    spam_prob = probability[0][1] * 100
    ham_prob = probability[0][0] * 100

    # 5. Display Result
    if result == 1:
        st.error("🚨 Spam Message Detected")
    else:
        st.success("✅ Safe Message")

    # 6. Display Confidence
    col1, col2 = st.columns(2)
    with col1:
      st.metric("Spam Risk:", f"{spam_prob:.2f}")
    with col2:
      st.metric("Safe Message Confidence:", f"{ham_prob:.2f}") 

    original_words = nltk.word_tokenize(input_sms)
    important_words = []
    for word in original_words:
      if word.isalnum() and word not in stopwords.words('english'):
        important_words.append(word)

    if result == 1:
      st.subheader("🔍 Suspicious Keywords")
      formatted_words = important_words[:5]
      st.write(", ".join(formatted_words))

import re
import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords

try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOP_WORDS = set(stopwords.words("english"))

st.set_page_config(page_title="Emotion Detection App", page_icon="😊", layout="centered")

model = joblib.load("logistic_model.pkl")
vectorizer = joblib.load("bow_vectorizer.pkl")

emotion_labels = {0: "sadness", 1: "anger", 2: "love", 3: "surprise", 4: "fear", 5: "joy"}
emotion_emoji = {"joy": "😊", "sadness": "😢", "anger": "😡", "fear": "😨", "love": "❤️", "surprise": "😲"}
emotion_color = {"joy": "#F5C400", "sadness": "#4A90D9", "anger": "#E74C3C",
                 "fear": "#8B5CF6", "love": "#EC4899", "surprise": "#F39C12"}

def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower()) #Letters aur spaces ke ilawa jo bhi cheez hai usay hata do.
    return " ".join(w for w in text.split() if w not in STOP_WORDS) # remove stopwords then append in w list
#st.markdown() Streamlit me text, HTML aur CSS display karne ke liye use hota hai.
st.markdown("""
<style>
.stApp{background:linear-gradient(135deg,#0a0e27 0%,#0f1b3d 50%,#1a2a4a 100%);}
header[data-testid="stHeader"]{background:transparent;}
.header-box{text-align:center;padding:25px;border-radius:15px;
background:linear-gradient(135deg,#6EE7B7,#8B5CF6);color:#11131F;margin-bottom:20px;}
.stButton button{border-radius:12px;height:50px;font-weight:bold;}
.stTextArea label p{color:#FFFFFF !important;font-weight:600;}
button[kind="secondary"], button[data-testid="baseButton-secondary"]{background-color:#2ECC71 !important;color:white !important;border:none;}
</style>
<div class="header-box">
<h1>😊 Emotion Detection System</h1>
<p>Detect emotions from text using Machine Learning ✨</p>
</div>
""", unsafe_allow_html=True)
# unsafe_allow_html=True = HTML aur CSS ko execute karne ki permission do.
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
#Session State kya hota hai?
#ek temporary storage (memory) hai jahan Streamlit values save karta hai.
st.text_area("Enter your text:", key="input_text", height=150,
              placeholder="Example: I am feeling very happy today...")

def clear_text():
    st.session_state.input_text = ""

col1, col2 = st.columns(2) #Ye screen ko 2 parts mein divide karta hai.
with col1:
    predict_btn = st.button("🔍 Predict Emotion", use_container_width=True, type="primary")
with col2:
    st.button("🗑️ Clear Input", use_container_width=True, on_click=clear_text)

if predict_btn:
    text = st.session_state.input_text
    if text.strip() == "": #strip() = extra spaces remove karta hai
        st.warning("Please enter some text.")
    else:
        vector = vectorizer.transform([clean_text(text)])
        #clean_text(text) → text ko clean karta hai (punctuation, lowercase etc.)
        #vectorizer.transform() → text ko TF-IDF / bag-of-words vector me convert karta hai
        pred_num = model.predict(vector)[0]
        prediction = emotion_labels[pred_num]
        confidence = max(model.predict_proba(vector)[0]) * 100 #[0] q Ye 2D array hai.pehli row niklani # predict_proba() = Ye har class ki probability batata hai.
        color = emotion_color[prediction]
        st.markdown(f"""
        <div style="background-color:{color};padding:30px;border-radius:15px;
        text-align:center;margin-top:10px;">
        <div style="font-size:60px;">{emotion_emoji[prediction]}</div>
        <h2 style="color:white;margin:5px 0;">{prediction.capitalize()}</h2>
        <p style="color:white;font-size:18px;">Confidence: {confidence:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(int(confidence)) #confidence colro by default streamlit
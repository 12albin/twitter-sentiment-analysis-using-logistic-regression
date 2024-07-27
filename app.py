import pickle
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer


tf=TfidfVectorizer()

st.title("twitter sentiment analysis")
model=pickle.load(open('trained_model.pkl','rb'))
tweet= st.text_input("enter the tweet")


submit=st.button("predict")
if submit:
    pred = model.predict([tweet])
    if pred[0]==0:
        st.write("negative")
    else:
        st.write("positive")

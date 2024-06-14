import streamlit as st
import pandas as pd
import time
import matplotlib as plt
import os

from transformers import MarianMTModel, MarianTokenizer, pipeline

st.title('Big Brain Translator: Angielski na Niemiecki')
st.markdown("### Witaj w Big Brain Translator! 🧠🌐")

st.write("Ta aplikacja umożliwia przetłumaczenie tekstu z języka angielskiego na niemiecki oraz ocenę wydźwięku emocjonalnego tekstu w języku angielskim.")
st.subheader('Instukcja')
st.write("1. Wybierz jedną z opcji z menu rozwijanego w sekcji: Co chcesz zrobic. , a następnie kliknij odpowiedni przycisk, aby uzyskać wynik.")
st.write("2. Wprowadź tekst do przetłumaczenia lub analizy.")
st.write("3. Kliknij odpowiedni przycisk.")


option = st.selectbox(
    "Co chcesz zrobić?",
    [
        "Oceń wydźwięk emocjonalny tekstu (ang)",
        "Przetłumacz tekst z angielskiego na niemiecki (ang -> de)",
    ],
)

def translate_text(text):
    model_name = "Helsinki-NLP/opus-mt-en-de"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_length=512)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

def analyze_sentiment(text):
    classifier = pipeline("sentiment-analysis")
    answer = classifier(text)
    return answer


if option == "Oceń wydźwięk emocjonalny tekstu (ang)":
    text = st.text_area(label="Wpisz tekst")
    if st.button("Analizuj"):
        if text:
            with st.spinner('Analizuję...'):
                try:
                    answer = analyze_sentiment(text)
                    st.write("Ocena wydźwięku emocjonalnego tekstu:", answer[0]['label'])
                except Exception as e:
                    st.error(f"Wystąpił błąd: {str(e)}")
        else:
            st.warning("Wprowadź tekst do analizy!")

elif option == "Przetłumacz tekst z angielskiego na niemiecki (ang -> de)":
    text_to_translate = st.text_area(label="Wprowadź tekst do przetłumaczenia")
    if st.button("Przetłumacz"):
        if text_to_translate:
            with st.spinner('Tłumaczę...'):
                try:
                    translated_text = translate_text(text_to_translate)
                    st.markdown(f'<p style="font-weight: bold; color: white; font-size: 25px;">Tekst po przetłumaczeniu: <span style="color: green;">{translated_text}</span></p>', unsafe_allow_html=True)
                    st.balloons()
                except Exception as e:
                    st.error(f"Wystąpił błąd: {str(e)}")
        else:
            st.warning("Wprowadź tekst do tłumaczenia!")

st.write("Moj numer index: s20202")

# st.subheader('Zadanie do wykonania')
# st.write('Wykorzystaj Huggin Face do stworzenia swojej własnej aplikacji tłumaczącej tekst z języka angielskiego na język niemiecki. Zmodyfikuj powyższy kod dodając do niego kolejną opcję, tj. tłumaczenie tekstu. Informacje potrzebne do zmodyfikowania kodu znajdziesz na stronie Huggin Face - https://huggingface.co/docs/transformers/index')
# st.write('🐞 Dodaj właściwy tytuł do swojej aplikacji, może jakieś grafiki?')
# st.write('🐞 Dodaj krótką instrukcję i napisz do czego służy aplikacja')
# st.write('🐞 Wpłyń na user experience, dodaj informacje o ładowaniu, sukcesie, błędzie, itd.')
# st.write('🐞 Na końcu umieść swój numer indeksu')
# st.write('🐞 Stwórz nowe repozytorium na GitHub, dodaj do niego swoją aplikację, plik z wymaganiami (requirements.txt)')
# st.write('🐞 Udostępnij stworzoną przez siebie aplikację (https://share.streamlit.io) a link prześlij do prowadzącego')

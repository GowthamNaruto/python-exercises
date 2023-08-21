import streamlit as st


def plinndrome(sentence):
    for i in (",.'?/><}{{}}'"):
        sentence = sentence.replace(i, "")
    palindrome = []
    words = sentence.split(' ')
    for word in words:
        word = word.lower()
        if word == word[::-1]:
            palindrome.append(word)
    return palindrome


st.set_page_config(
    page_title="Plinddrome",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# Palindrome words Finder app"
    }
)

st.title("Find Palindrome words")
sentence = st.text_area("Enter a sentence: ")
st.write(f"Palindrome wrods: {plinndrome(sentence)}")
# st.metric(label="Temperature", value="70 °F", delta="1.2 °F")

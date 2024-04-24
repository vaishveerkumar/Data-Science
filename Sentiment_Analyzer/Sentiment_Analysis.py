import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Streamlit app
st.markdown("<h1 style='text-align: center; color: black; font-family: Cambria;'>TEXT SENTIMENT ANALYZER</h1>", unsafe_allow_html=True)

# CSS for flashing banner effect and flashing prediction
css = """
@keyframes flash-banner {
    from { opacity: 1; }
    to { opacity: 0; }
}

@keyframes flash-prediction {
    from { background-color: black; }
    to { background-color: transparent; }
}

.flash-banner {
    animation: flash-banner 3s infinite alternate;
}

.flash-text {
    font-size: 24px;
    font-weight: bold;
    text-align: center;
    color: #ff6347;
}

.flash-prediction {
    animation: flash-prediction 2s infinite alternate;
    font-size: 20px;
    font-weight: bold;
    text-align: center;
    color: white;
    padding: 10px;
    border-radius: 5px;
}

div[data-baseweb="textarea-container"] {
    border: 2px solid black;
}
"""

# Apply CSS
st.write(f'<style>{css}</style>', unsafe_allow_html=True)

st.markdown("<h2 style='color: black; font-size: 20px; font-weight: bold;'>Enter Description:</h2>", unsafe_allow_html=True)

text = st.text_area("", height=100)

if st.button("Analyze"):
    sentiment_scores = analyzer.polarity_scores(text)
    
    # Determine overall sentiment
    if sentiment_scores['compound'] >= 0.05:
        overall_sentiment = "Positive"
        emoji = "😊"
    elif sentiment_scores['compound'] <= -0.05:
        overall_sentiment = "Negative"
        emoji = "😔"
    else:
        overall_sentiment = "Neutral"
        emoji = "😐"
    
    # Display the prediction with flashing effect and black background
    st.markdown(f'<div class="flash-prediction">{emoji} {overall_sentiment}</div>', unsafe_allow_html=True)

# Add background image
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("https://www.datasciencecentral.com/wp-content/uploads/2022/12/AdobeStock_507986014-scaled.jpeg");
        background-size: cover;
        background-position: center center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)
